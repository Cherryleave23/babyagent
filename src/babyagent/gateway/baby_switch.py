"""宝宝切换机制 — 隐式+显式切换检测（R20-R23）

实现在对话中自动检测宝宝切换意图的逻辑：

规则覆盖：
  - R20: 切换成功时生成带档案摘要的确认消息
  - R21: 显式 @切换宝宝 / @宝宝 指令强制切换
  - R22: 消息中无宝宝名时返回 unchanged
  - R23: 多匹配冲突时返回候选列表要求用户确认
  - R38: 切换后首次 LLM 调用注入档案摘要（由上层编排器处理）

纯规则驱动，不调用 LLM。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 显式切换指令模式
_EXPLICIT_SWITCH_PATTERNS = [
    re.compile(r"@切换宝宝\s+(?P<name>.+)"),
    re.compile(r"@宝宝\s+(?P<name>.+)"),
    re.compile(r"切换(?:到|成)?(?P<name>[\u4e00-\u9fff]{1,6})(?:宝宝|小朋友)?"),
]

# 隐式宝宝名检测模式（常见称呼上下文）
_IMPLICIT_NAME_PATTERNS = [
    re.compile(r"(?:帮|给|替|为)(?:我|我们)?(?:看看|查查|问问|关心一下)[\u4e00-\u9fff]{1,6}(?:宝宝|小朋友)?"),
    re.compile(r"(?:我家|我们家|咱家)(?P<name>[\u4e00-\u9fff]{1,6})(?:宝宝|小朋友)"),
    re.compile(r"切换到?\s*(?P<name>[\u4e00-\u9fff]{1,6})(?:宝宝|小朋友)?"),
    re.compile(r"(?P<name>[\u4e00-\u9fff]{2,6})(?:宝宝|小朋友)(?:的|今天|最近|现在|怎么样|还好吗|又|拉|吃|喝|睡|发烧|感冒|咳嗽|湿疹|过敏|便秘|腹泻)"),
]


def _extract_name_from_message(message: str) -> Optional[str]:
    """从消息文本中提取可能的宝宝名字。

    优先级：
      1. 显式 @切换宝宝 / @宝宝 指令 → 强制切换
      2. 常见中文称呼上下文 → 隐式检测

    Args:
        message: 员工消息文本。

    Returns:
        提取到的宝宝名字，未检测到时返回 None。
    """
    if not message:
        return None

    # ---- 第一优先级：显式切换指令 ----
    for pattern in _EXPLICIT_SWITCH_PATTERNS:
        m = pattern.search(message)
        if m:
            name = m.group("name").strip()
            # 过滤过长的字符串
            if 1 <= len(name) <= 10:
                logger.debug("检测到显式切换指令: name=%s", name)
                return name

    # ---- 第二优先级：隐式称呼检测 ----
    for pattern in _IMPLICIT_NAME_PATTERNS:
        m = pattern.search(message)
        if m:
            name = m.group("name").strip()
            # 过滤常见误匹配（非名字的词汇）
            if _is_likely_name(name):
                logger.debug("检测到隐式宝宝名: name=%s", name)
                return name

    return None


def _is_likely_name(text: str) -> bool:
    """判断文本是否像一个宝宝名字（而非偶然匹配的普通词）。

    过滤规则：
      - 名字长度为 1-6 个汉字
      - 排除常见非名字词汇（如：这个、那个、什么、怎么等）

    Args:
        text: 待检查的文本。

    Returns:
        True 如果文本像人名。
    """
    if not text:
        return False

    # 长度过滤
    if len(text) < 1 or len(text) > 6:
        return False

    # 常见误匹配黑名单
    _COMMON_FALSE_POSITIVES = {
        "这个", "那个", "什么", "怎么", "一个", "每个", "哪个",
        "有没有", "是不是", "怎么样", "怎么办", "还好吗",
        "为什么", "能不能", "可不可以", "需要", "请问",
        "帮忙", "谢谢", "你好", "您好",
        "今天", "明天", "昨天", "最近",
        "一次", "一天", "一个", "几次",
        "牛奶", "鸡蛋", "花生", "海鲜",  # 常见过敏原，不是名字
        "上周", "下周", "本月",
    }

    if text in _COMMON_FALSE_POSITIVES:
        return False

    # 如果包含非中文常见字，降低信任度但不过滤
    return True


def _generate_switch_confirmation(baby_name: str, profile_summary: str) -> str:
    """生成切换确认消息（R20）。

    格式: "已切换到 {name} 的档案（{profile_summary}）"

    Args:
        baby_name: 宝宝名字。
        profile_summary: 档案摘要文本。

    Returns:
        带档案摘要的切换确认消息。
    """
    return f"\U0001f504 已切换到{baby_name}的档案（{profile_summary}）"


def _generate_conflict_message(baby_name: str, candidates: list[dict[str, Any]]) -> str:
    """生成重名冲突提示（R23）。

    当同一门店内有多个同名宝宝时，要求用户提供更多信息以做区分。

    Args:
        baby_name: 冲突的宝宝名字。
        candidates: 候选宝宝列表，每项含 id, gender, age_months 等。

    Returns:
        冲突提示文本。
    """
    lines = [f"找到 {len(candidates)} 个叫「{baby_name}」的宝宝，请确认您指的是哪一个："]
    for i, c in enumerate(candidates, 1):
        gender_cn = {"male": "男", "female": "女", "unknown": "未知"}.get(c.get("gender", "unknown"), "未知")
        age = c.get("age_months", 0)
        birth = c.get("birth_date", "未知")
        lines.append(f"{i}. {baby_name}，{gender_cn}，{age}个月（出生日期: {birth}）")

    lines.append("\n请回复数字编号来选择，或用 @Agent 建档 提供更详细的区分信息。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def detect_baby_switch(
    message: str,
    current_baby_name: Optional[str],
    profile_manager: Any,  # BabyProfileManager
    store_id: str,
) -> dict[str, Any]:
    """检测并处理宝宝切换意图。

    规则覆盖 R20-R23：
      - R20: 单匹配成功 → 切换并返回确认消息（含档案摘要）
      - R21: 显式 @切换宝宝/@宝宝 → 强制切换
      - R22: 无宝宝名 → unchanged
      - R23: 多匹配冲突 → 返回候选列表

    Args:
        message: 员工消息文本。
        current_baby_name: 当前活跃的宝宝名字（可为 None）。
        profile_manager: BabyProfileManager 实例，须提供 resolve_baby_by_name() 和 get_profile()。
        store_id: 门店 ID。

    Returns:
        字典包含:
          - switched: bool — 是否发生了切换
          - new_baby_id: Optional[int] — 新宝宝 ID
          - new_baby_name: Optional[str] — 新宝宝名字
          - confirmation_message: Optional[str] — 切换确认消息
          - conflict_babies: Optional[list] — R23 冲突候选列表
    """
    # ---- 步骤 1: 提取宝宝名 ----
    detected_name = _extract_name_from_message(message)

    # ---- 步骤 2: 无宝宝名 → unchanged（R22） ----
    if detected_name is None:
        logger.debug("消息中未检测到宝宝名: message='%s...'", message[:50])
        return {
            "switched": False,
            "new_baby_id": None,
            "new_baby_name": None,
            "confirmation_message": None,
            "conflict_babies": None,
        }

    # ---- 步骤 3: 同名 → unchanged（无需切换） ----
    if current_baby_name is not None and detected_name == current_baby_name:
        logger.debug("宝宝名与当前相同，无需切换: name=%s", detected_name)
        return {
            "switched": False,
            "new_baby_id": None,
            "new_baby_name": None,
            "confirmation_message": None,
            "conflict_babies": None,
        }

    # ---- 步骤 4: 调用 profile_manager 按名搜索 ----
    logger.info("检测到宝宝切换意图: detected=%s, current=%s", detected_name, current_baby_name)

    try:
        candidates = profile_manager.resolve_baby_by_name(
            name=detected_name,
            store_id=store_id,
        )
    except Exception as exc:
        logger.error("按名搜索宝宝失败: name=%s, error=%s", detected_name, exc)
        return {
            "switched": False,
            "new_baby_id": None,
            "new_baby_name": None,
            "confirmation_message": f"搜索宝宝「{detected_name}」时出错，请稍后重试。",
            "conflict_babies": None,
        }

    # ---- 步骤 5a: 无匹配 ----
    if not candidates:
        logger.info("未找到匹配的宝宝: name=%s", detected_name)
        return {
            "switched": False,
            "new_baby_id": None,
            "new_baby_name": None,
            "confirmation_message": f"未找到名为「{detected_name}」的宝宝档案。请先用 @Agent 建档 创建档案。",
            "conflict_babies": None,
        }

    # ---- 步骤 5b: 多匹配冲突（R23） ----
    if len(candidates) > 1:
        logger.info("宝宝重名冲突: name=%s, count=%d", detected_name, len(candidates))
        conflict_data = []
        for c in candidates:
            profile = c
            if hasattr(c, "model_dump"):
                profile = c.model_dump()
            elif hasattr(c, "dict"):
                profile = c.dict()
            conflict_data.append({
                "id": profile.get("id", ""),
                "name": profile.get("name", detected_name),
                "gender": profile.get("gender", "unknown"),
                "age_months": profile.get("age_months", 0),
                "birth_date": profile.get("birth_date", "未知"),
            })

        return {
            "switched": False,
            "new_baby_id": None,
            "new_baby_name": None,
            "confirmation_message": _generate_conflict_message(detected_name, conflict_data),
            "conflict_babies": conflict_data,
        }

    # ---- 步骤 5c: 单匹配成功 → 切换（R20） ----
    matched = candidates[0]
    # 兼容 Pydantic model 和 dict
    if hasattr(matched, "model_dump"):
        profile_data = matched.model_dump()
    elif hasattr(matched, "dict"):
        profile_data = matched.dict()
    else:
        profile_data = matched

    baby_id = profile_data.get("id", "")
    baby_name = profile_data.get("name", detected_name)

    # 生成档案摘要
    if hasattr(matched, "to_context_summary"):
        summary = matched.to_context_summary()
    else:
        gender_cn = {"male": "男", "female": "女", "unknown": "未知"}.get(
            profile_data.get("gender", "unknown"), "未知"
        )
        age = profile_data.get("age_months", 0)
        allergies = []
        for a in (profile_data.get("allergies") or []):
            if isinstance(a, dict):
                n = a.get("allergen", "")
                if n and not a.get("removed"):
                    allergies.append(n)
            elif hasattr(a, "allergen") and not getattr(a, "removed", False):
                allergies.append(a.allergen)
        summary_parts = [f"{baby_name}，{gender_cn}，{age}个月"]
        if allergies:
            summary_parts.append(f"过敏：{'、'.join(allergies)}")
        summary = "，".join(summary_parts)

    confirmation = _generate_switch_confirmation(baby_name, summary)

    logger.info("宝宝切换成功: name=%s, id=%s", baby_name, baby_id)
    return {
        "switched": True,
        "new_baby_id": baby_id,
        "new_baby_name": baby_name,
        "confirmation_message": confirmation,
        "conflict_babies": None,
    }
