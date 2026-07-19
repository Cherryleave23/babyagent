"""宝宝档案模型与管理器

BabyProfile: Pydantic 数据模型，表示一份完整的宝宝档案。
BabyProfileManager: 档案管理器，封装 UnifiedStore 的读写 + 业务规则。

关键设计约束：
  - R23: 按名字搜索宝宝，返回多个候选时要求用户确认
  - R32: 压缩输出合入档案，仅高置信度自动应用
  - R34: 过敏记录 append-only，仅 human 可标记移除
  - 数据作用域 (data_scope) 在列表查询中控制可见性
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, field_validator

from ..db.unified_store import UnifiedStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BabyProfile — 宝宝档案数据模型
# ---------------------------------------------------------------------------


class GrowthRecord(BaseModel):
    """生长发育记录"""
    id: str = ""
    record_date: str
    height_cm: float | None = None
    weight_kg: float | None = None


class AllergyEntry(BaseModel):
    """过敏条目"""
    allergen: str = ""
    source: str = "human"  # human | llm_append
    discovered_date: str | None = None
    notes: str = ""
    removed: bool = False


class NoteEntry(BaseModel):
    """LLM 备注条目"""
    session_id: str = ""
    content: str = ""
    created_at: str = ""


class BabyProfile(BaseModel):
    """宝宝完整档案模型

    包含宝宝基本信息、过敏、生长记录、LLM 备注等全部字段，
    使用 Pydantic 提供序列化与验证。
    """

    id: str = ""
    store_id: str = ""
    name: str = ""
    gender: str = "unknown"  # male | female | unknown
    birth_date: str = ""      # YYYY-MM-DD

    dietary_restrictions: list[str] = Field(default_factory=list)
    notes_free_text: str = ""

    allergies: list[AllergyEntry] = Field(default_factory=list)
    growth_records: list[GrowthRecord] = Field(default_factory=list)
    notes: list[NoteEntry] = Field(default_factory=list)

    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    @field_validator("gender")
    @classmethod
    def _validate_gender(cls, v: str) -> str:
        if v not in ("male", "female", "unknown"):
            raise ValueError(f"性别无效: {v}，允许值: male, female, unknown")
        return v

    @computed_field
    @property
    def age_months(self) -> int:
        """计算月龄（从出生日期到今天的月份数）。

        Returns:
            月龄整数，出生日期未设置时返回 0。
        """
        if not self.birth_date:
            return 0
        try:
            birth = date.fromisoformat(self.birth_date)
            today = date.today()
            months = (today.year - birth.year) * 12 + (today.month - birth.month)
            if today.day < birth.day:
                months -= 1
            return max(0, months)
        except (ValueError, TypeError):
            return 0

    @computed_field
    @property
    def active_allergies(self) -> list[AllergyEntry]:
        """当前活跃的过敏列表（排除已移除的）。"""
        return [a for a in self.allergies if not a.removed]

    @computed_field
    @property
    def allergy_list(self) -> list[str]:
        """活跃过敏原名称列表。"""
        return [a.allergen for a in self.active_allergies]

    def to_context_summary(self) -> str:
        """生成简洁的"宝宝当前状态"摘要文本。

        格式示例:
            "安安，女，8个月。已知过敏：米粉（2026-07-19添加后起疹，已停）。当前喂养：配方奶为主，辅食暂停。"

        Returns:
            单行或少数几行的中文状态摘要。
        """
        name = self.name or "未命名宝宝"
        gender_cn = {"male": "男", "female": "女", "unknown": "未知"}.get(self.gender, "未知")
        age = self.age_months

        parts = [f"{name}，{gender_cn}，{age}个月。"]

        # 过敏信息
        if self.active_allergies:
            allergy_strs = []
            for a in self.active_allergies:
                detail = a.allergen
                if a.discovered_date:
                    detail += f"（{a.discovered_date}"
                    if a.notes:
                        detail += f"添加后{a.notes}"
                    if a.removed:
                        detail += "，已停"
                    detail += "）"
                allergy_strs.append(detail)
            parts.append(f"已知过敏：{'、'.join(allergy_strs)}。")

        # 饮食限制
        if self.dietary_restrictions:
            parts.append(f"饮食限制：{'、'.join(self.dietary_restrictions)}。")

        # 自由文本备注用于喂养状态
        if self.notes_free_text:
            parts.append(f"当前喂养：{self.notes_free_text}。")

        summary = "".join(parts)
        logger.debug("生成宝宝状态摘要: %s", summary)
        return summary

    @classmethod
    def from_sql_row(
        cls,
        row: dict[str, Any],
        allergies: list[dict[str, Any]] | None = None,
        growth: list[dict[str, Any]] | None = None,
        notes: list[dict[str, Any]] | None = None,
    ) -> "BabyProfile":
        """从 SQLite 行数据 + 关联表数据构建 BabyProfile 实例。

        Args:
            row: babies 表的行字典。
            allergies: baby_allergy_history 表的相关行列表。
            growth: baby_growth_records 表的相关行列表。
            notes: baby_notes 表的相关行列表。

        Returns:
            完整的 BabyProfile 实例。
        """
        dietary = row.get("dietary_restrictions", "[]")
        if isinstance(dietary, str):
            try:
                dietary = json.loads(dietary)
            except (json.JSONDecodeError, TypeError):
                dietary = []

        return cls(
            id=row.get("id", ""),
            store_id=row.get("store_id", ""),
            name=row.get("name", ""),
            gender=row.get("gender", "unknown"),
            birth_date=row.get("birth_date", ""),
            dietary_restrictions=dietary if isinstance(dietary, list) else [],
            notes_free_text=row.get("notes_free_text", ""),
            allergies=[
                AllergyEntry(
                    allergen=a.get("allergen", ""),
                    source=a.get("source", "human"),
                    discovered_date=a.get("discovered_date"),
                    notes=a.get("notes", ""),
                    removed=bool(a.get("removed", 0)),
                )
                for a in (allergies or [])
            ],
            growth_records=[
                GrowthRecord(
                    id=g.get("id", ""),
                    record_date=g.get("record_date", ""),
                    height_cm=g.get("height_cm"),
                    weight_kg=g.get("weight_kg"),
                )
                for g in (growth or [])
            ],
            notes=[
                NoteEntry(
                    session_id=n.get("session_id", ""),
                    content=n.get("content", ""),
                    created_at=n.get("created_at", ""),
                )
                for n in (notes or [])
            ],
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
            version=row.get("version", 1),
        )


# ---------------------------------------------------------------------------
# BabyProfileManager — 档案管理器
# ---------------------------------------------------------------------------


class BabyProfileManager:
    """宝宝档案管理器

    封装 UnifiedStore 操作，提供高级档案管理接口。
    负责数据作用域控制、过敏规则执行、冲突检测等业务逻辑。

    Attributes:
        store: UnifiedStore 实例。
    """

    def __init__(self, store: UnifiedStore) -> None:
        """初始化管理器。

        Args:
            store: 已初始化的 UnifiedStore 实例。
        """
        self.store = store

    # ---- 创建 ----

    def create_profile(
        self,
        store_id: str,
        name: str,
        gender: str,
        birth_date: str,
        dietary_restrictions: list[str] | None = None,
        notes_free_text: str = "",
    ) -> BabyProfile:
        """创建新宝宝档案。

        Args:
            store_id: 门店 ID。
            name: 宝宝姓名/昵称。
            gender: 性别 ('male' / 'female' / 'unknown')。
            birth_date: 出生日期 (YYYY-MM-DD)。
            dietary_restrictions: 初始饮食限制列表。
            notes_free_text: 自由文本备注。

        Returns:
            新创建的 BabyProfile 实例。
        """
        row = self.store.create_baby(
            store_id=store_id,
            name=name,
            gender=gender,
            birth_date=birth_date,
            dietary_restrictions=dietary_restrictions,
            notes_free_text=notes_free_text,
        )
        return BabyProfile.from_sql_row(row)

    # ---- 读取 ----

    def get_profile(self, baby_id: str) -> BabyProfile | None:
        """按 ID 加载完整宝宝档案（含过敏、生长、备注）。

        Args:
            baby_id: 宝宝 ID。

        Returns:
            BabyProfile 实例，不存在时返回 None。
        """
        row = self.store.get_baby(baby_id)
        if row is None:
            return None

        allergies = self.store.get_allergies_for_baby(baby_id)
        growth = self.store.get_growth_records(baby_id)
        notes = self.store.get_notes_for_baby(baby_id)

        return BabyProfile.from_sql_row(row, allergies=allergies, growth=growth, notes=notes)

    def list_profiles(
        self,
        store_id: str,
        data_scope: str = "store",
        filter_store_id: str | None = None,
    ) -> list[BabyProfile]:
        """列出宝宝档案，遵守 data_scope 作用域规则。

        Args:
            store_id: 当前操作的门店 ID。
            data_scope: 数据作用域 ('enterprise' / 'store' / 'store_strict')。
                - enterprise: 返回全部门店的宝宝（跨店可见）
                - store: 仅返回当前门店的宝宝
                - store_strict: 仅返回当前门店，且严格隔离
            filter_store_id: data_scope='enterprise' 时可选指定门店过滤。

        Returns:
            BabyProfile 列表。
        """
        if data_scope in ("store", "store_strict"):
            effective_store = store_id
        elif data_scope == "enterprise" and filter_store_id:
            effective_store = filter_store_id
        else:
            effective_store = None

        rows = self.store.list_babies(store_id=effective_store)

        profiles = []
        for row in rows:
            baby_id = row["id"]
            allergies = self.store.get_allergies_for_baby(baby_id)
            growth = self.store.get_growth_records(baby_id)
            notes = self.store.get_notes_for_baby(baby_id)
            profiles.append(BabyProfile.from_sql_row(row, allergies=allergies, growth=growth, notes=notes))

        return profiles

    def resolve_baby_by_name(
        self,
        name: str,
        store_id: str,
    ) -> list[BabyProfile]:
        """按宝宝姓名搜索（R23: 支持重名冲突处理）。

        在指定门店内搜索，返回所有匹配的宝宝。
        当返回多个时，调用方应要求用户确认选择。

        Args:
            name: 宝宝姓名（精确匹配或模糊匹配由实现决定）。
            store_id: 门店 ID。

        Returns:
            匹配的 BabyProfile 列表。空列表表示无匹配。
        """
        rows = self.store.list_babies(store_id=store_id)
        matching = [r for r in rows if r.get("name") == name]

        profiles = []
        for row in matching:
            baby_id = row["id"]
            allergies = self.store.get_allergies_for_baby(baby_id)
            growth = self.store.get_growth_records(baby_id)
            notes = self.store.get_notes_for_baby(baby_id)
            profiles.append(BabyProfile.from_sql_row(row, allergies=allergies, growth=growth, notes=notes))

        logger.info("按姓名 '%s' 搜索: 找到 %d 个匹配", name, len(profiles))
        return profiles

    # ---- 过敏管理（R34） ----

    def update_allergy(
        self,
        baby_id: str,
        allergen: str,
        operation: str,
        source: str,
        session_id: str | None = None,
        notes: str = "",
    ) -> AllergyEntry:
        """更新宝宝的过敏信息（R34: append-only 规则）。

        Args:
            baby_id: 宝宝 ID。
            allergen: 过敏原名称。
            operation: 操作类型：
                - 'append': 追加新过敏记录（任意 source 均可）
                - 'delete': 标记过敏为已移除（仅 source='human' 允许）
            source: 来源 ('human' / 'llm_append')。
            session_id: 关联会话 ID。
            notes: 备注说明。

        Returns:
            操作产生的 AllergyEntry。

        Raises:
            ValueError: 操作规则冲突时。
        """
        if operation == "append":
            record = self.store.record_baby_allergy(
                baby_id=baby_id,
                allergen=allergen,
                source=source,
                session_id=session_id,
                notes=notes,
            )
            return AllergyEntry(
                allergen=record.get("allergen", allergen),
                source=record.get("source", source),
                discovered_date=record.get("discovered_date"),
                notes=record.get("notes", notes),
                removed=bool(record.get("removed", 0)),
            )

        elif operation == "delete":
            if source != "human":
                raise ValueError(
                    "过敏删除仅允许人工操作 (source='human')，当前来源: %s" % source
                )
            # 找到未移除的同名过敏记录并标记
            existing = self.store.get_allergies_for_baby(baby_id)
            for a in existing:
                if a["allergen"] == allergen and not a.get("removed"):
                    self.store.mark_allergy_removed(a["id"], by=source)
                    return AllergyEntry(
                        allergen=allergen,
                        source=source,
                        discovered_date=a.get("discovered_date"),
                        notes=a.get("notes", ""),
                        removed=True,
                    )
            raise ValueError(f"未找到活跃的过敏记录: baby_id={baby_id}, allergen={allergen}")

        else:
            raise ValueError(f"不支持的过敏操作: {operation}，允许 'append' 或 'delete'")

    # ---- 生长记录 ----

    def add_growth_record(
        self,
        baby_id: str,
        record_date: str,
        height_cm: float | None = None,
        weight_kg: float | None = None,
    ) -> GrowthRecord:
        """添加宝宝的生长发育记录。

        Args:
            baby_id: 宝宝 ID。
            record_date: 记录日期。
            height_cm: 身高（厘米）。
            weight_kg: 体重（千克）。

        Returns:
            新建的 GrowthRecord。
        """
        record = self.store.add_growth_record(
            baby_id=baby_id,
            record_date=record_date,
            height_cm=height_cm,
            weight_kg=weight_kg,
        )
        return GrowthRecord(
            id=record.get("id", ""),
            record_date=record.get("record_date", ""),
            height_cm=record.get("height_cm"),
            weight_kg=record.get("weight_kg"),
        )

    # ---- 档案更新（R32: 压缩合入） ----

    def merge_baby_updates(
        self,
        baby_id: str,
        updates: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """处理压缩输出的结构化更新（R32）。

        每个更新条目格式:
            {baby_id, field, operation, value, confidence}

        规则:
            - confidence='high': 自动应用到档案
            - confidence='medium' / 'low': 放入 pending 队列，等待人工确认
            - operation='append': 追加到列表字段
            - operation='replace': 覆盖字段
            - operation='delete': 从列表字段移除

        Args:
            baby_id: 目标宝宝 ID。
            updates: 结构化更新列表。

        Returns:
            {'applied': [...], 'pending': [...]} 区分已应用和待确认的更新。
        """
        applied = []
        pending = []
        profile = self.get_profile(baby_id)

        if profile is None:
            logger.warning("merge_baby_updates: 未找到宝宝 baby_id=%s", baby_id)
            return {"applied": [], "pending": updates}

        for upd in updates:
            upd_baby_id = upd.get("baby_id", "")
            if upd_baby_id and upd_baby_id != baby_id:
                pending.append(upd)
                continue

            field = upd.get("field", "")
            operation = upd.get("operation", "")
            value = upd.get("value")
            confidence = upd.get("confidence", "low")

            if confidence != "high":
                pending.append(upd)
                continue

            try:
                self._apply_single_update(baby_id, profile, field, operation, value)
                applied.append(upd)
                logger.info("档案更新已应用: baby=%s, field=%s, op=%s", baby_id, field, operation)
            except Exception as exc:
                logger.warning("应用更新失败(已移入pending): %s", exc)
                pending.append(upd)

        return {"applied": applied, "pending": pending}

    def _apply_single_update(
        self,
        baby_id: str,
        profile: BabyProfile,
        field: str,
        operation: str,
        value: Any,
    ) -> None:
        """执行单条更新操作到档案。

        Args:
            baby_id: 宝宝 ID。
            profile: 当前 BabyProfile。
            field: 字段名。
            operation: 操作类型。
            value: 值。

        Raises:
            ValueError: 操作无效时。
        """
        if field == "dietary_restrictions":
            if operation == "append":
                current = list(profile.dietary_restrictions)
                if isinstance(value, str) and value not in current:
                    current.append(value)
                elif isinstance(value, list):
                    for v in value:
                        if v not in current:
                            current.append(v)
                self.store.update_baby(baby_id, {"dietary_restrictions": current})
            elif operation == "delete":
                current = list(profile.dietary_restrictions)
                if isinstance(value, str) and value in current:
                    current.remove(value)
                elif isinstance(value, list):
                    current = [v for v in current if v not in value]
                self.store.update_baby(baby_id, {"dietary_restrictions": current})
            elif operation == "replace":
                new_val = value if isinstance(value, list) else [value]
                self.store.update_baby(baby_id, {"dietary_restrictions": new_val})

        elif field == "notes_free_text":
            if operation in ("replace", "append"):
                # append 模式: 追加文本
                if operation == "append":
                    new_text = profile.notes_free_text
                    if new_text:
                        new_text += "\n" + str(value)
                    else:
                        new_text = str(value)
                else:
                    new_text = str(value)
                self.store.update_baby(baby_id, {"notes_free_text": new_text})

        elif field == "allergies":
            if operation == "append":
                if isinstance(value, str):
                    self.store.record_baby_allergy(
                        baby_id=baby_id,
                        allergen=value,
                        source="llm_append",
                    )
                elif isinstance(value, dict):
                    self.store.record_baby_allergy(
                        baby_id=baby_id,
                        allergen=value.get("allergen", str(value)),
                        source="llm_append",
                        notes=value.get("notes", ""),
                    )

        elif field in ("name", "gender", "birth_date"):
            if operation == "replace":
                self.store.update_baby(baby_id, {field: value})

        else:
            logger.debug("未处理的字段: field=%s, operation=%s", field, operation)
