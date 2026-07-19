"""上下文压缩双输出 — Session 摘要 + 档案更新

R31/R39 上下文压缩流程：
  1. 提取宝宝档案更新（baby_profile_updates）—— 从 session 中识别结构化变更
  2. 生成 session 摘要（session_summary）—— 自然语言总结
  3. 仅 confidence='high' 的更新作为可操作输出返回

双输出模型（CompressionOutput）:
  - session_summary: 压缩后的 session 文本摘要
  - baby_profile_updates: 结构化档案更新列表
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 辅助数据结构
# ---------------------------------------------------------------------------


class BabyProfileUpdate(BaseModel):
    """单条宝宝档案更新建议"""
    baby_id: str = ""
    field: str = ""          # 字段名: dietary_restrictions, allergies, notes_free_text, name, gender, birth_date
    operation: str = ""      # 操作: append | replace | delete
    value: Any = None        # 值
    confidence: str = "low"  # 置信度: high | medium | low


class CompressionOutput(BaseModel):
    """压缩输出双结果模型

    Attributes:
        session_summary: 会话精华概要，替代原始 session 存入长期记忆。
        baby_profile_updates: 从 session 中提取的宝宝档案结构化更新列表。
    """
    session_summary: str = ""
    baby_profile_updates: list[BabyProfileUpdate] = Field(default_factory=list)

    def actionable_updates(self) -> list[BabyProfileUpdate]:
        """获取可自动执行的更新（仅 confidence='high'）。

        Returns:
            高置信度更新列表。
        """
        return [u for u in self.baby_profile_updates if u.confidence == "high"]

    def pending_updates(self) -> list[BabyProfileUpdate]:
        """获取待人工确认的更新（confidence != 'high'）。

        Returns:
            中/低置信度更新列表。
        """
        return [u for u in self.baby_profile_updates if u.confidence != "high"]


# ---------------------------------------------------------------------------
# 压缩 Prompt 模板
# ---------------------------------------------------------------------------

_COMPRESSION_SYSTEM_PROMPT = """你是一个专业的母婴顾问 AI，现在需要对一段聊天记录进行**双重压缩**。

你的任务：
1. 先从对话中提取关于宝宝档案的结构化变更
2. 再生成一段简洁的 session 摘要

**宝宝档案字段说明：**
- dietary_restrictions: 饮食限制/禁忌（列表）
- allergies: 过敏信息（列表，每项为 {allergen, notes, discovered_date}）
- notes_free_text: 喂养状态自由文本
- name: 宝宝姓名
- gender: 性别（male/female/unknown）
- birth_date: 出生日期（YYYY-MM-DD）

**操作类型：**
- append: 追加新条目到列表字段
- replace: 覆盖字段的当前值
- delete: 从列表字段移除条目

**置信度（confidence）：**
- high: 信息明确，可以直接应用
- medium: 信息可能存在歧义，建议确认
- low: 信息不确定，仅供参考

**重要规则：**
- 只输出 JSON，不要输出任何其他文本
- 如果对话中没有任何档案变更，updates 应为空列表 []
- session_summary 用中文，控制在 100~300 字以内
"""

_COMPRESSION_USER_TEMPLATE = """## 当前宝宝档案快照
{profile_snapshot}

## 本次 Session 对话历史
{session_history}

请按照上述规则进行双重压缩，输出 JSON 格式。"""

# ---------------------------------------------------------------------------
# 压缩生成函数
# ---------------------------------------------------------------------------


def generate_compression(
    session_history: str,
    current_baby_profile_snapshot: str,
    aux_model_config: dict[str, Any],
) -> CompressionOutput:
    """执行上下文压缩，生成双输出（R31/R39）。

    **流程（R39）：**
    1. 提取宝宝档案更新（baby_profile_updates）—— 优先从对话中识别结构化变更
    2. 生成 session 摘要（session_summary）—— 自然语言总结对话精华

    **置信度过滤：** 仅 confidence='high' 的更新视为可操作；
    其余放入 pending 队列等待人工确认。

    Args:
        session_history: 当前 session 的完整对话历史文本。
        current_baby_profile_snapshot: 当前宝宝的档案快照文本（通常 from to_context_summary()）。
        aux_model_config: 辅助 LLM 配置字典，包含 provider, model_name, api_key, base_url 等。

    Returns:
        CompressionOutput 包含 session_summary 和 baby_profile_updates。

    Raises:
        RuntimeError: 模型调用失败时。
        ValueError: 模型返回无法解析时。
    """
    prompt = _build_compression_prompt(session_history, current_baby_profile_snapshot)

    try:
        raw_output = _call_auxiliary_llm(prompt, aux_model_config)
        output = _parse_compression_output(raw_output)
        logger.info(
            "压缩完成: summary_chars=%d, updates=%d (high=%d)",
            len(output.session_summary),
            len(output.baby_profile_updates),
            len(output.actionable_updates()),
        )
        return output
    except Exception as exc:
        logger.error("压缩生成失败: %s", exc)
        # 降级：返回仅含 session_summary 的结果
        return _fallback_compression(session_history, str(exc))


def _build_compression_prompt(
    session_history: str,
    profile_snapshot: str,
) -> list[dict[str, str]]:
    """构建发送给辅助 LLM 的 messages 列表。

    Args:
        session_history: session 对话历史。
        profile_snapshot: 宝宝档案快照。

    Returns:
        OpenAI-compatible messages 列表。
    """
    user_content = _COMPRESSION_USER_TEMPLATE.format(
        profile_snapshot=profile_snapshot or "暂无档案信息",
        session_history=session_history,
    )
    return [
        {"role": "system", "content": _COMPRESSION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _call_auxiliary_llm(
    messages: list[dict[str, str]],
    config: dict[str, Any],
) -> str:
    """调用辅助 LLM 并返回原始文本响应。

    支持 OpenAI-compatible API。

    Args:
        messages: 消息列表。
        config: 模型配置字典。

    Returns:
        LLM 响应的文本内容。

    Raises:
        RuntimeError: 调用失败时。
    """
    provider = config.get("provider", "openai")
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    model_name = config.get("model_name", "gpt-4o-mini")

    # 对 deepseek 等需要 OpenAI-compatible 的 provider 走统一路径
    if not api_key:
        import os
        api_key = os.environ.get("BABYAGENT_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    if not api_key:
        raise RuntimeError("辅助模型 API key 未配置")

    try:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
            max_tokens=2048,
        )

        content = response.choices[0].message.content
        logger.debug("LLM 压缩响应: %s...", (content or "")[:200])
        return content or ""

    except ImportError:
        logger.warning("openai 库未安装，尝试使用 httpx 直连")
        return _call_via_httpx(messages, config)
    except Exception as exc:
        raise RuntimeError(f"辅助 LLM 调用失败: {exc}") from exc


def _call_via_httpx(
    messages: list[dict[str, str]],
    config: dict[str, Any],
) -> str:
    """通过 httpx 直连 OpenAI-compatible API 调用辅助 LLM。

    Args:
        messages: 消息列表。
        config: 模型配置。

    Returns:
        LLM 响应文本。

    Raises:
        RuntimeError: 调用失败时。
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx 库未安装，无法调用辅助 LLM")

    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "https://api.openai.com/v1")
    model_name = config.get("model_name", "gpt-4o-mini")

    if not api_key:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content or ""
    except Exception as exc:
        raise RuntimeError(f"httpx 直连调用失败: {exc}") from exc


def _parse_compression_output(raw: str) -> CompressionOutput:
    """解析 LLM 返回的压缩 JSON 输出。

    处理常见的格式化问题：
    - 前后可能存在的 markdown 代码块
    - LLM 可能输出的解释性文本
    - JSON 格式偏差

    Args:
        raw: LLM 原始响应文本。

    Returns:
        CompressionOutput 实例。

    Raises:
        ValueError: 解析失败时。
    """
    if not raw or not raw.strip():
        raise ValueError("LLM 返回为空")

    # 1) 尝试提取 JSON 代码块
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # 2) 尝试找最外层 { 到 }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start:end + 1]
        else:
            json_str = raw.strip()

    # 3) 解析
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # 尝试修复常见问题（如尾逗号）
        cleaned = re.sub(r',\s*}', '}', json_str)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 LLM 输出为 JSON: {raw[:300]}") from exc

    # 4) 构建 CompressionOutput
    session_summary = data.get("session_summary", "")
    updates_raw = data.get("baby_profile_updates", [])

    updates = []
    if isinstance(updates_raw, list):
        for u in updates_raw:
            if not isinstance(u, dict):
                continue
            updates.append(BabyProfileUpdate(
                baby_id=u.get("baby_id", ""),
                field=u.get("field", ""),
                operation=u.get("operation", ""),
                value=u.get("value"),
                confidence=u.get("confidence", "low"),
            ))

    return CompressionOutput(
        session_summary=session_summary,
        baby_profile_updates=updates,
    )


def _fallback_compression(
    session_history: str,
    error_msg: str = "",
) -> CompressionOutput:
    """压缩失败时的降级处理。

    当 LLM 调用失败时，生成一个基础的摘要来保留关键信息。

    Args:
        session_history: 原始 session 对话。
        error_msg: 失败原因。

    Returns:
        降级的 CompressionOutput（仅含 session_summary）。
    """
    # 简单截取前 300 字作为降级摘要
    truncated = session_history[:300]
    summary = f"[压缩失败{': ' + error_msg if error_msg else ''}] 原始对话截取: {truncated}..."
    logger.warning("压缩降级: %s", summary[:100])
    return CompressionOutput(
        session_summary=summary,
        baby_profile_updates=[],
    )


# ---------------------------------------------------------------------------
# 便捷工厂函数
# ---------------------------------------------------------------------------


def build_empty_output() -> CompressionOutput:
    """构建一个空的压缩输出（无变更场景）。

    Returns:
        空的 CompressionOutput。
    """
    return CompressionOutput(
        session_summary="(本次对话未产生新的档案变更)",
        baby_profile_updates=[],
    )
