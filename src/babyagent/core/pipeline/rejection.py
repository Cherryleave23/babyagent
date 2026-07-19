"""拒绝处理器 — 超出范围 & 紧急症状的预设回复模板

实现 R5（超出范围拒绝）和 R6（紧急就医转介）规则。
严格遵循 I2 不变量：拒绝场景绝不调用 LLM，全部使用硬编码模板。

拒绝类型：
  - out_of_scope: 非 0-6 岁宝宝领域的问题
  - emergency: 需立即就医的急症关键词命中
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .intent import IntentResult, MessageIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 预设拒绝模板（R5）
# ---------------------------------------------------------------------------

_REJECT_OUT_OF_SCOPE = (
    "抱歉，我是母婴健康助手，只能回答关于0-6岁宝宝身体健康和营养方面的问题。"
    "如果您有这方面的问题，我很乐意帮助您！"
)

# ---------------------------------------------------------------------------
# 紧急就医模板（R6）
# ---------------------------------------------------------------------------

_REJECT_EMERGENCY = (
    "您描述的症状属于紧急情况，请立即就医或拨打120急救电话！\n\n"
    "以下情况需要立即就医：\n"
    "- 高热惊厥（发烧伴随抽搐）\n"
    "- 呼吸困难或发绀\n"
    "- 意识模糊或昏迷\n"
    "- 严重外伤出血不止\n\n"
    "我不是医疗急救系统，无法处理紧急情况。请马上带宝宝去最近的医院急诊科！"
)

# ---------------------------------------------------------------------------
# 紧急症状关键词（R6）
# ---------------------------------------------------------------------------

_EMERGENCY_KEYWORDS: list[tuple[str, int]] = [
    # (关键词, 严重等级 1-3)
    ("高热惊厥", 3),
    ("呼吸困难", 3),
    ("意识模糊", 3),
    ("抽搐", 3),
    ("窒息", 3),
    ("昏迷", 3),
    ("发绀", 3),
    ("休克", 3),
    ("心脏骤停", 3),
    ("大出血", 3),
    ("严重过敏反应", 2),
    ("喉头水肿", 2),
    ("急性中毒", 2),
    ("误吞异物窒息", 3),
    ("口吐白沫", 2),
    ("翻白眼", 2),
    ("不省人事", 3),
    ("叫不醒", 2),
    ("嘴唇发紫", 2),
    ("喘不上气", 2),
    ("喘不过气", 2),
    ("呼吸急促伴", 2),
]


def is_emergency_situation(message_text: str) -> bool:
    """检测用户消息中是否包含紧急症状关键词。

    基于关键词匹配，识别需要立即就医的危急情况。
    匹配时忽略大小写。

    Args:
        message_text: 用户消息文本。

    Returns:
        True 表示检测到紧急情况，应触发就医转介模板。
    """
    if not message_text:
        return False

    text_lower = message_text.lower()

    for keyword, severity in _EMERGENCY_KEYWORDS:
        # 简单子串匹配即可覆盖中文关键词
        if keyword in text_lower or keyword in message_text:
            logger.warning(
                "检测到紧急症状关键词: '%s' (严重等级=%d), 消息片段: %s",
                keyword, severity, message_text[:80],
            )
            return True

    return False


def generate_rejection(
    intent_result: Optional[IntentResult] = None,
    message_text: str = "",
) -> str:
    """生成拒绝回复文本。

    根据拒绝类型返回对应的预设模板：
      - 紧急症状 → 就医转介（R6）
      - 超出范围 → 范围声明（R5）
      - 其他/未指定 → 范围声明（保底）

    绝不调用 LLM（I2 不变量）。

    Args:
        intent_result: 意图分类结果，为 None 时仅检查 message_text。
        message_text: 原始用户消息，用于紧急关键词检测。

    Returns:
        中文拒绝回复文本。
    """
    # 紧急情况优先检测（R6）
    if is_emergency_situation(message_text):
        logger.info("生成紧急就医转介回复")
        return _REJECT_EMERGENCY

    # 超出范围（R5）
    if intent_result is not None and intent_result.intent == MessageIntent.OUT_OF_SCOPE:
        logger.info("生成超出范围拒绝回复 (intent=%s)", intent_result.intent.value)
        return _REJECT_OUT_OF_SCOPE

    # 保底：任何未归类的拒绝请求均使用范围声明模板
    logger.info("生成保底拒绝回复")
    return _REJECT_OUT_OF_SCOPE


def get_rejection_message_by_type(rejection_type: str) -> str:
    """按拒绝类型返回对应的模板文本。

    方便在上层管线中按类型获取模板后进一步拼接上下文。

    Args:
        rejection_type: 拒绝类型标识：
            - "out_of_scope": 超出范围
            - "emergency": 紧急就医

    Returns:
        对应的预设模板文本。未知类型返回 out_of_scope 模板。
    """
    templates = {
        "out_of_scope": _REJECT_OUT_OF_SCOPE,
        "emergency": _REJECT_EMERGENCY,
    }
    return templates.get(rejection_type, _REJECT_OUT_OF_SCOPE)
