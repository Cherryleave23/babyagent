"""意图分类器与管线路由分发

基于 LLM 结构化输出对用户消息进行意图识别，并根据识别结果
将请求路由到正确的处理管线：产品推荐 / 健康膳食咨询 / 拒绝。

分类边界（R1）：
  - product_recommend: 包含明确产品询问词 + 产品品类信号
  - diet_advice: 食物/营养相关提问
  - health_consult: 健康症状、发育、疫苗类提问
  - out_of_scope: 超出 0-6 岁宝宝身体健康 + 营养领域
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 枚举 & 数据模型
# ---------------------------------------------------------------------------


class MessageIntent(str, Enum):
    """用户消息意图枚举

    分类依据 R1 规范，四个意图明确界定各自的范围。
    """

    PRODUCT_RECOMMEND = "product_recommend"   # 产品推荐：明确的"推荐/买/有没有/用什么"
    DIET_ADVICE = "diet_advice"               # 膳食咨询：食物、营养相关提问
    HEALTH_CONSULT = "health_consult"          # 健康咨询：症状、发育、疫苗类提问
    OUT_OF_SCOPE = "out_of_scope"              # 超出母婴范围：非 0-6 岁宝宝领域


class IntentResult(BaseModel):
    """意图分类结果

    Attributes:
        intent: 分类后的意图枚举值。
        baby_name_hint: 用户消息中可能提及的宝宝名字，用于自动切换上下文。
        confidence: 分类置信度 (0-1)。
    """

    intent: MessageIntent = Field(description="分类意图")
    baby_name_hint: Optional[str] = Field(default=None, description="消息中检测到的宝宝名字")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="分类置信度")


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = """你是一个母婴健康助手的意图分类器。请分析用户的消息，判断属于以下哪种意图：

1. product_recommend（产品推荐）：
   - 用户消息包含明确的"推荐"、"买"、"有没有"、"用什么"、"选哪个"等询问产品的措辞
   - 并且提到了产品品类（如奶粉、尿布、湿疹膏、辅食、奶瓶等）
   - 例如："推荐一款适合6个月宝宝的奶粉"、"宝宝红屁股用什么好"

2. diet_advice（膳食咨询）：
   - 用户询问有关食物、喂养、营养、辅食添加的问题
   - 例如："8个月宝宝可以吃蛋黄吗"、"辅食怎么添加"、"宝宝不爱喝奶怎么办"

3. health_consult（健康咨询）：
   - 用户询问宝宝的健康症状、身体发育、疫苗接种、常见病症护理等问题
   - 例如："宝宝发烧怎么办"、"湿疹怎么护理"、"疫苗后有什么反应"

4. out_of_scope（超出范围）：
   - 问题是关于0-6岁宝宝身体健康和营养之外的领域
   - 例如：育儿心理、成人健康、宠物、娱乐、政治、闲聊等
   - 注意：所有不属于母婴身体健康 + 营养领域的问题都归为 out_of_scope

另外，如果用户消息中提到了宝宝的名字（如"帮我家安安看看"、"小宝今天..."），请提取出来。

请以 JSON 格式回复，包含以下字段：
- intent: 意图分类，取值为 "product_recommend" / "diet_advice" / "health_consult" / "out_of_scope"
- baby_name_hint: 如果检测到宝宝名字则填写，否则为 null
- confidence: 置信度 (0.0-1.0)，表示你对分类的把握程度
- reasoning: 简短的中文分类理由（1-2句话）
"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _build_model_client(model_config: dict[str, Any]) -> Any:
    """根据配置构建 OpenAI SDK 兼容的客户端。

    支持 DeepSeek / GLM / ChatGPT 等所有 OpenAI 兼容 API。

    Args:
        model_config: 模型配置字典，须包含 api_key, base_url, model_name 等字段。

    Returns:
        OpenAI SDK 客户端实例。

    Raises:
        ImportError: openai 包未安装时。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    return OpenAI(
        api_key=model_config.get("api_key", ""),
        base_url=model_config.get("base_url", ""),
    )


def _extract_baby_name_from_text(message_text: str) -> Optional[str]:
    """用正则辅助从消息中提取可能的宝宝名字。

    仅作为冗余提取手段；主要提取由 LLM 完成。
    模式：紧接在称呼词后的中文名字。

    Args:
        message_text: 用户消息文本。

    Returns:
        提取到的名字，无法提取时返回 None。
    """
    patterns = [
        r"(?:我家|我们家|帮(?:我|我们)?)[的]?([\u4e00-\u9fff]{1,4})(?:宝宝)?(?:看看|查查|问问|怎么样|怎么办|最近)",
        r"([\u4e00-\u9fff]{1,4})(?:宝宝)?(?:今天|最近|现在)",
        r"帮(?:我|我们)?看看(.{1,4})[的]?",
    ]
    for pattern in patterns:
        m = re.search(pattern, message_text)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# 意图分类
# ---------------------------------------------------------------------------


async def classify_intent(
    message_text: str,
    active_baby_name: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
) -> IntentResult:
    """对用户消息进行意图分类。

    调用 LLM 进行结构化意图识别，同时探测消息中是否提及宝宝名字。
    若 LLM 调用失败，按 R1 规则降级为 health_consult（安全侧保守策略）。

    Args:
        message_text: 用户消息文本。
        active_baby_name: 当前活跃的宝宝名字（用于上下文提示）。
        model_config: 模型配置字典。为 None 时跳过 LLM 直接返回降级结果。

    Returns:
        IntentResult 包含意图分类结果、宝宝名字提示和置信度。
    """
    # 无模型配置时直接降级
    if not model_config or not model_config.get("api_key"):
        logger.warning("无可用模型配置，意图分类降级为 health_consult")
        return IntentResult(
            intent=MessageIntent.HEALTH_CONSULT,
            baby_name_hint=None,
            confidence=0.5,
        )

    # 构建用户提示上下文
    context_note = ""
    if active_baby_name:
        context_note = f"\n当前活跃的宝宝是「{active_baby_name}」。如果用户提到其他名字，可能是要切换宝宝。"

    user_prompt = f"用户消息：{message_text}{context_note}"

    try:
        client = _build_model_client(model_config)

        response = client.chat.completions.create(
            model=model_config.get("model_name", "deepseek-chat"),
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=model_config.get("max_tokens", 500),
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("LLM 返回空内容")

        parsed = json.loads(raw)

        intent_str = parsed.get("intent", "health_consult")
        try:
            intent = MessageIntent(intent_str)
        except ValueError:
            logger.warning("LLM 返回未知意图值: %s，回退为 health_consult", intent_str)
            intent = MessageIntent.HEALTH_CONSULT

        # 优先使用 LLM 提取的宝宝名，其次用正则辅助
        baby_name = parsed.get("baby_name_hint")
        if not baby_name:
            baby_name = _extract_baby_name_from_text(message_text)

        confidence = float(parsed.get("confidence", 0.8))
        # 确保 confidence 在合法范围
        confidence = max(0.0, min(1.0, confidence))

        reasoning = parsed.get("reasoning", "")
        logger.info(
            "意图分类完成: intent=%s, baby_name=%s, confidence=%.2f, reason=%s",
            intent.value, baby_name, confidence, reasoning,
        )

        return IntentResult(
            intent=intent,
            baby_name_hint=baby_name,
            confidence=confidence,
        )

    except Exception as exc:
        logger.error("意图分类 LLM 调用失败，降级为 health_consult: %s", exc)
        return IntentResult(
            intent=MessageIntent.HEALTH_CONSULT,
            baby_name_hint=_extract_baby_name_from_text(message_text),
            confidence=0.3,
        )


# ---------------------------------------------------------------------------
# 同步便捷封装
# ---------------------------------------------------------------------------


def classify_intent_sync(
    message_text: str,
    active_baby_name: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
) -> IntentResult:
    """同步版意图分类（内部使用 asyncio.run 封装）。

    方便在非 async 上下文中调用。

    Args:
        message_text: 用户消息文本。
        active_baby_name: 当前活跃的宝宝名字。
        model_config: 模型配置字典。

    Returns:
        IntentResult 分类结果。
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(classify_intent(message_text, active_baby_name, model_config))
    else:
        # 已在事件循环中运行，创建新的事件循环来执行
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                classify_intent(message_text, active_baby_name, model_config),
            )
            return future.result()


# ---------------------------------------------------------------------------
# 路由分发
# ---------------------------------------------------------------------------


def route_by_intent(
    intent_result: IntentResult,
    employee_msg: str = "",
    session_context: Optional[dict[str, Any]] = None,
) -> str:
    """根据意图分类结果路由到对应管线处理器。

    路由规则：
      - product_recommend → "product"
      - diet_advice     → "health_diet"
      - health_consult  → "health_diet"
      - out_of_scope    → "reject"

    Args:
        intent_result: 意图分类结果。
        employee_msg: 员工消息文本（用于紧急情况检测）。
        session_context: 会话上下文（预留扩展）。

    Returns:
        管线处理器标识: "product" / "health_diet" / "reject"
    """
    intent = intent_result.intent

    if intent == MessageIntent.OUT_OF_SCOPE:
        logger.info("路由决策: reject (out_of_scope)")
        return "reject"

    # 紧急症状检测（R6）：高热惊厥、呼吸困难等直接走 reject
    # 注意：如果意图不是 out_of_scope 但消息中包含紧急关键词，仍应由上层
    # 在调用 route_by_intent 前先检查 is_emergency_situation()
    # 这里仅做保底回退
    if intent == MessageIntent.PRODUCT_RECOMMEND:
        logger.info("路由决策: product (product_recommend, confidence=%.2f)", intent_result.confidence)
        return "product"

    if intent in (MessageIntent.DIET_ADVICE, MessageIntent.HEALTH_CONSULT):
        logger.info("路由决策: health_diet (%s, confidence=%.2f)", intent.value, intent_result.confidence)
        return "health_diet"

    # 保底：未知意图走健康膳食管线
    logger.warning("路由决策: 未知意图 %s，回退为 health_diet", intent)
    return "health_diet"
