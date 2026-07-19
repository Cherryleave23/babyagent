"""@Agent 指令解析器 — 基于正则的规则驱动指令解析

解析微信消息中的 @Agent 结构化指令，所有解析均为纯规则驱动，
绝不调用 LLM（I2 不变量）。

支持的指令格式：
  - @Agent 建档 {name} {gender} {birth_date}              → create_profile
  - @Agent {name} 过敏史添加 {allergen}                    → add_allergy
  - @Agent {name} 过敏史删除 {allergen}                    → delete_allergy
  - @Agent {name} 生长记录 {date} {height}cm {weight}kg    → add_growth
  - @宝宝 {name}                                            → switch_baby
  - @切换宝宝 {name}                                        → switch_baby

设计约束：
  - 全部基于 regex 解析，零 LLM 调用
  - 格式不匹配时返回对应的错误提示，帮助用户纠正输入
  - 中文分词按场景定制，不做通用 NLP
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """指令解析结果

    Attributes:
        cmd: 指令类型标识:
            - "create_profile": 建档
            - "add_allergy": 添加过敏记录
            - "delete_allergy": 删除过敏记录
            - "add_growth": 添加生长记录
            - "switch_baby": 切换宝宝
            - "unknown": 无法识别的指令
        args: 指令参数字典，字段因指令类型而异。
        error_message: 解析失败时的错误提示，成功时为 None。
        raw_message: 原始消息文本。
    """

    cmd: str = "unknown"
    args: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    raw_message: str = ""


# ---------------------------------------------------------------------------
# 正则模式定义
# ---------------------------------------------------------------------------

# 全局 @Agent 前缀识别
_AGENT_PREFIX_RE = re.compile(r"^\s*@Agent\s+", re.IGNORECASE)
_AT_BAOBAO_RE = re.compile(r"^\s*@宝宝\s+", re.IGNORECASE)
_AT_SWITCH_RE = re.compile(r"^\s*@切换宝宝\s+", re.IGNORECASE)


# 建档: @Agent 建档 小明 男 2024-01-15
_CREATE_PROFILE_RE = re.compile(
    r"建档\s+"
    r"(?P<name>[\u4e00-\u9fff\w]{1,20})\s+"
    r"(?P<gender>(?:男|女|male|female|未知|unknown))\s+"
    r"(?P<birth_date>\d{4}-\d{2}-\d{2})"
    r"(?:\s+(?P<notes>.+))?",
)

# 过敏添加: @Agent 小明 过敏史添加 牛奶蛋白
_ADD_ALLERGY_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fff\w]{1,20})\s+过敏史添加\s+(?P<allergen>.+)",
)

# 过敏删除: @Agent 小明 过敏史删除 牛奶蛋白
_DELETE_ALLERGY_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fff\w]{1,20})\s+过敏史删除\s+(?P<allergen>.+)",
)

# 生长记录: @Agent 小明 生长记录 2024-07-15 65cm 7.5kg
_ADD_GROWTH_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fff\w]{1,20})\s+"
    r"生长记录\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<height>[\d.]+)\s*cm\s+"
    r"(?P<weight>[\d.]+)\s*kg",
)

# 备选格式: 仅身高或仅体重
_ADD_GROWTH_HEIGHT_ONLY_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fff\w]{1,20})\s+"
    r"生长记录\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<height>[\d.]+)\s*cm",
)

_ADD_GROWTH_WEIGHT_ONLY_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fff\w]{1,20})\s+"
    r"生长记录\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<weight>[\d.]+)\s*kg",
)


# ---------------------------------------------------------------------------
# 公共入口
# ---------------------------------------------------------------------------


def parse_command(message_text: str) -> CommandResult:
    """解析微信消息中的 @Agent 指令。

    这是指令解析的公共入口。先检测前缀类型，再按模式匹配。
    不匹配时返回 cmd="unknown" 并附带错误提示。

    Args:
        message_text: 原始的微信消息文本。

    Returns:
        CommandResult 包含解析结果或错误信息。
    """
    original = message_text.strip()

    # ---- 1. 检测 @宝宝 / @切换宝宝 前缀 ----
    match_at_baobao = _AT_BAOBAO_RE.match(original)
    match_at_switch = _AT_SWITCH_RE.match(original)

    if match_at_baobao or match_at_switch:
        if match_at_baobao:
            body = original[match_at_baobao.end():].strip()
        else:
            body = original[match_at_switch.end():].strip()

        if not body:
            return CommandResult(
                cmd="switch_baby",
                args={},
                error_message="请指定要切换的宝宝名字，例如：@切换宝宝 小明",
                raw_message=original,
            )

        logger.info("解析切换宝宝指令: name=%s", body)
        return CommandResult(
            cmd="switch_baby",
            args={"name": body},
            raw_message=original,
        )

    # ---- 2. 检测 @Agent 前缀 ----
    match_agent = _AGENT_PREFIX_RE.match(original)
    if not match_agent:
        return CommandResult(
            cmd="unknown",
            args={},
            error_message=None,
            raw_message=original,
        )

    body = original[match_agent.end():].strip()

    if not body:
        return CommandResult(
            cmd="unknown",
            args={},
            error_message=(
                "@Agent 指令格式不正确。支持以下格式：\n"
                "  @Agent 建档 {姓名} {性别} {出生日期}\n"
                "  @Agent {姓名} 过敏史添加 {过敏原}\n"
                "  @Agent {姓名} 过敏史删除 {过敏原}\n"
                "  @Agent {姓名} 生长记录 {日期} {身高}cm {体重}kg"
            ),
            raw_message=original,
        )

    # ---- 3. 按模式顺序匹配 ----
    result = _try_create_profile(body)
    if result is not None:
        return CommandResult(cmd="create_profile", args=result, raw_message=original)

    result = _try_growth_record(body)
    if result is not None:
        return CommandResult(cmd="add_growth", args=result, raw_message=original)

    result = _try_add_allergy(body)
    if result is not None:
        return CommandResult(cmd="add_allergy", args=result, raw_message=original)

    result = _try_delete_allergy(body)
    if result is not None:
        return CommandResult(cmd="delete_allergy", args=result, raw_message=original)

    # ---- 4. 全部不匹配 ----
    logger.debug("无法识别的 @Agent 指令: body=%s", body)
    return CommandResult(
        cmd="unknown",
        args={},
        error_message=(
            f"无法识别的 @Agent 指令：「{body}」。\n"
            "支持以下格式：\n"
            "  @Agent 建档 {姓名} {性别} {出生日期}\n"
            "  @Agent {姓名} 过敏史添加 {过敏原}\n"
            "  @Agent {姓名} 过敏史删除 {过敏原}\n"
            "  @Agent {姓名} 生长记录 {日期} {身高}cm {体重}kg\n\n"
            "示例：@Agent 建档 小明 男 2024-01-15"
        ),
        raw_message=original,
    )


# ---------------------------------------------------------------------------
# 各子指令解析
# ---------------------------------------------------------------------------


def _try_create_profile(body: str) -> Optional[dict[str, Any]]:
    """尝试解析建档指令。

    格式：建档 {name} {gender} {birth_date} [notes]

    Args:
        body: @Agent 后的消息体。

    Returns:
        参数字典或 None。
    """
    m = _CREATE_PROFILE_RE.match(body)
    if not m:
        return None

    gender_map = {
        "男": "male",
        "male": "male",
        "女": "female",
        "female": "female",
        "未知": "unknown",
        "unknown": "unknown",
    }

    args = {
        "name": m.group("name").strip(),
        "gender": gender_map.get(m.group("gender"), "unknown"),
        "birth_date": m.group("birth_date"),
    }

    notes = m.group("notes")
    if notes:
        args["notes_free_text"] = notes.strip()

    logger.info("建档指令解析成功: name=%s, gender=%s, birth=%s", args["name"], args["gender"], args["birth_date"])
    return args


def _try_add_allergy(body: str) -> Optional[dict[str, Any]]:
    """尝试解析过敏添加指令。

    格式：{name} 过敏史添加 {allergen}

    Args:
        body: @Agent 后的消息体。

    Returns:
        参数字典或 None。
    """
    m = _ADD_ALLERGY_RE.match(body)
    if not m:
        return None

    allergen = m.group("allergen").strip()
    if not allergen:
        return None

    args = {
        "name": m.group("name").strip(),
        "allergen": allergen,
    }

    logger.info("过敏添加指令解析成功: name=%s, allergen=%s", args["name"], args["allergen"])
    return args


def _try_delete_allergy(body: str) -> Optional[dict[str, Any]]:
    """尝试解析过敏删除指令。

    格式：{name} 过敏史删除 {allergen}

    Args:
        body: @Agent 后的消息体。

    Returns:
        参数字典或 None。
    """
    m = _DELETE_ALLERGY_RE.match(body)
    if not m:
        return None

    allergen = m.group("allergen").strip()
    if not allergen:
        return None

    args = {
        "name": m.group("name").strip(),
        "allergen": allergen,
    }

    logger.info("过敏删除指令解析成功: name=%s, allergen=%s", args["name"], args["allergen"])
    return args


def _try_growth_record(body: str) -> Optional[dict[str, Any]]:
    """尝试解析生长记录指令。

    格式（完整）：{name} 生长记录 {date} {height}cm {weight}kg
    格式（仅身高）：{name} 生长记录 {date} {height}cm
    格式（仅体重）：{name} 生长记录 {date} {weight}kg

    Args:
        body: @Agent 后的消息体。

    Returns:
        参数字典或 None。
    """
    # 先尝试完整格式
    m = _ADD_GROWTH_RE.match(body)
    if m:
        args: dict[str, Any] = {
            "name": m.group("name").strip(),
            "record_date": m.group("date"),
            "height_cm": float(m.group("height")),
            "weight_kg": float(m.group("weight")),
        }
        logger.info("生长记录指令解析成功(完整): name=%s, height=%s, weight=%s",
                     args["name"], args["height_cm"], args["weight_kg"])
        return args

    # 仅身高
    m = _ADD_GROWTH_HEIGHT_ONLY_RE.match(body)
    if m:
        args = {
            "name": m.group("name").strip(),
            "record_date": m.group("date"),
            "height_cm": float(m.group("height")),
            "weight_kg": None,
        }
        logger.info("生长记录指令解析成功(仅身高): name=%s, height=%s", args["name"], args["height_cm"])
        return args

    # 仅体重
    m = _ADD_GROWTH_WEIGHT_ONLY_RE.match(body)
    if m:
        args = {
            "name": m.group("name").strip(),
            "record_date": m.group("date"),
            "height_cm": None,
            "weight_kg": float(m.group("weight")),
        }
        logger.info("生长记录指令解析成功(仅体重): name=%s, weight=%s", args["name"], args["weight_kg"])
        return args

    return None


# ---------------------------------------------------------------------------
# 错误提示生成
# ---------------------------------------------------------------------------


def get_help_message() -> str:
    """获取 @Agent 指令的帮助文本。

    Returns:
        格式化的帮助文本。
    """
    return (
        "@Agent 指令帮助：\n"
        "1. 建档：@Agent 建档 {姓名} {性别} {出生日期}\n"
        "   示例：@Agent 建档 小明 男 2024-01-15\n\n"
        "2. 过敏添加：@Agent {姓名} 过敏史添加 {过敏原}\n"
        "   示例：@Agent 小明 过敏史添加 牛奶蛋白\n\n"
        "3. 过敏删除：@Agent {姓名} 过敏史删除 {过敏原}\n"
        "   示例：@Agent 小明 过敏史删除 牛奶蛋白\n\n"
        "4. 生长记录：@Agent {姓名} 生长记录 {日期} {身高}cm {体重}kg\n"
        "   示例：@Agent 小明 生长记录 2024-07-15 65cm 7.5kg\n\n"
        "5. 切换宝宝：@切换宝宝 {姓名} 或 @宝宝 {姓名}\n"
        "   示例：@切换宝宝 小红"
    )
