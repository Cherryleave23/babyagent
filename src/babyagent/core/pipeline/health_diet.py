"""健康与膳食咨询管线 — 基于 RAG 的知识检索增强生成

对健康症状、发育、疫苗接种及膳食营养类问题，先检索知识库中的
相关专业内容，再结合宝宝档案上下文由 LLM 生成自然语言回复。

关键约束：
  - 必须包含医学免责声明
  - 知识来源：ChromaDB 三层知识库（种子层/企业层/公共层）
  - 不生成产品名称（I1 不变量在此管线同样生效）
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 免责声明
# ---------------------------------------------------------------------------

_MEDICAL_DISCLAIMER = "\n\n---\n*以上建议仅供参考，如症状持续或加重请及时就医。*"

# ---------------------------------------------------------------------------
# LLM 提示词
# ---------------------------------------------------------------------------

_CONSULTATION_SYSTEM_PROMPT = """你是一个专业的母婴健康顾问，专注于 0-6 岁宝宝的身体健康和营养领域。

请根据提供的知识库内容和宝宝档案信息，回答用户的问题。

回答要求：
1. 以知识库内容为主要依据，不要编造未经证实的信息
2. 结合宝宝的具体情况（月龄、过敏史、饮食限制等）给出个性化建议
3. 使用温暖、专业的语气，像一位有经验的母婴护理专家
4. 回答结构清晰、分点说明，便于阅读
5. 如果问题超出你的知识范围，请诚实说明并建议咨询专业医生
6. 绝对不要推荐具体的产品品牌或名称
7. 对于医疗类问题，回答末尾必须包含就医建议

回答格式建议：
- 先对用户关心的问题表示理解和共情
- 再给出基于知识的专业解答
- 最后提供实用的日常护理或预防建议
"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _build_model_client(model_config: dict[str, Any]) -> Any:
    """构建 OpenAI SDK 兼容客户端。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai 包: pip install openai")

    return OpenAI(
        api_key=model_config.get("api_key", ""),
        base_url=model_config.get("base_url", ""),
    )


def _format_knowledge_results(knowledge_results: list[dict[str, Any]]) -> str:
    """将知识库检索结果格式化为 LLM 可读的文本块。

    Args:
        knowledge_results: search_knowledge() 返回的结果列表。

    Returns:
        格式化的知识文本。
    """
    if not knowledge_results:
        return "（无相关知识库内容）"

    formatted_parts = []
    for i, item in enumerate(knowledge_results, 1):
        content = item.get("document", item.get("content", ""))
        metadata = item.get("metadata", {})
        source = metadata.get("source", metadata.get("title", ""))
        if content:
            if source:
                formatted_parts.append(f"[参考 {i}] (来源: {source})\n{content}")
            else:
                formatted_parts.append(f"[参考 {i}]\n{content}")

    return "\n\n---\n\n".join(formatted_parts)


def _format_baby_context(baby_profile: dict[str, Any]) -> str:
    """将宝宝档案格式化为上下文文本。

    Args:
        baby_profile: 宝宝档案字典。

    Returns:
        结构化的宝宝上下文文本。
    """
    parts = []

    name = baby_profile.get("name", "未命名")
    age = baby_profile.get("age_months", 0)
    gender = baby_profile.get("gender", "unknown")
    gender_cn = {"male": "男", "female": "女", "unknown": "未知"}.get(gender, "未知")

    parts.append(f"宝宝: {name}，{gender_cn}，{age}个月")

    restrictions = baby_profile.get("dietary_restrictions", [])
    if restrictions:
        parts.append(f"饮食限制: {', '.join(restrictions)}")

    allergies = baby_profile.get("allergies", [])
    if allergies:
        allergy_names = []
        for a in allergies:
            if isinstance(a, dict):
                name_a = a.get("allergen", "")
                if not a.get("removed", False) and name_a:
                    allergy_names.append(name_a)
            elif isinstance(a, str):
                allergy_names.append(a)
        if allergy_names:
            parts.append(f"过敏原: {', '.join(allergy_names)}")

    notes = baby_profile.get("notes_free_text", "")
    if notes:
        parts.append(f"喂养备注: {notes}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 主管线
# ---------------------------------------------------------------------------


async def consult_health_diet(
    employee_message: str,
    baby_profile: dict[str, Any],
    unified_store: Any,
    model_config: dict[str, Any],
) -> str:
    """健康/膳食咨询主管线 — RAG 检索增强生成。

    流程：
      1. 在知识库中搜索相关内容
      2. 构建包含宝宝档案 + 知识检索 + 用户消息的完整上下文
      3. 调用 LLM 生成自然语言回复
      4. 追加医学免责声明

    Args:
        employee_message: 员工/用户的提问消息。
        baby_profile: 宝宝档案字典。
        unified_store: UnifiedStore 实例，须提供 search_knowledge() 方法。
        model_config: LLM 模型配置字典。

    Returns:
        包含免责声明的中文回复文本。
    """
    logger.info("健康膳食咨询启动: message='%s...'", employee_message[:60])

    # ---- 步骤 1: 知识库检索 ----
    knowledge_results: list[dict[str, Any]] = []
    try:
        knowledge_results = unified_store.search_knowledge(
            query=employee_message,
            top_k=5,
        )
        logger.info("知识检索完成，返回 %d 条结果", len(knowledge_results))
    except Exception as exc:
        logger.warning("知识库检索失败，将在无知识背景的情况下生成回复: %s", exc)

    # ---- 步骤 2: 构建上下文 ----
    baby_context = _format_baby_context(baby_profile)
    knowledge_text = _format_knowledge_results(knowledge_results)

    user_prompt = (
        f"## 宝宝信息\n{baby_context}\n\n"
        f"## 知识库参考资料\n{knowledge_text}\n\n"
        f"## 用户问题\n{employee_message}\n\n"
        "请基于以上信息回答用户的问题。"
    )

    # ---- 步骤 3: LLM 生成回复 ----
    try:
        client = _build_model_client(model_config)

        response = client.chat.completions.create(
            model=model_config.get("model_name", "deepseek-chat"),
            messages=[
                {"role": "system", "content": _CONSULTATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=model_config.get("temperature", 0.6),
            max_tokens=model_config.get("max_tokens", 1500),
        )

        answer = response.choices[0].message.content
        if not answer:
            raise ValueError("LLM 返回空内容")

        logger.info("健康膳食咨询回复生成完成，长度: %d 字", len(answer))

    except Exception as exc:
        logger.error("LLM 调用失败: %s", exc)
        # 降级：生成模板化回复
        answer = _generate_fallback_response(employee_message, baby_profile, knowledge_results)
        logger.info("使用降级回复模板")

    # ---- 步骤 4: 追加免责声明 ----
    full_response = answer.rstrip() + _MEDICAL_DISCLAIMER
    return full_response


# ---------------------------------------------------------------------------
# 降级回复
# ---------------------------------------------------------------------------


def _generate_fallback_response(
    message: str,
    baby_profile: dict[str, Any],
    knowledge_results: list[dict[str, Any]],
) -> str:
    """LLM 不可用时的降级模板回复。

    Args:
        message: 用户消息。
        baby_profile: 宝宝档案。
        knowledge_results: 知识检索结果。

    Returns:
        中文模板回复。
    """
    age = baby_profile.get("age_months", 0)

    # 如果有知识库内容，简单引用
    if knowledge_results:
        knowledge_ref = "根据相关知识库信息，建议您关注以下几点：\n"
        for i, item in enumerate(knowledge_results[:3], 1):
            content = item.get("document", item.get("content", ""))[:200]
            if content:
                knowledge_ref += f"{i}. {content}...\n"
    else:
        knowledge_ref = ""

    return (
        f"感谢您的咨询！关于您提到的关于{age}个月宝宝的问题，我已收到。\n\n"
        f"{knowledge_ref}"
        f"\n由于当前系统服务受限，建议您稍后重试或直接咨询儿科医生获取专业意见。"
    )
