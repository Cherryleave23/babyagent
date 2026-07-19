"""统一数据存储层 — SQLite + ChromaDB

UnifiedStore 是数据库操作的唯一入口，同时管理：
 - SQLite 结构化数据（10 张基表）
 - ChromaDB 向量知识库（三层集合）
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from .schema import (
    ChromaDBWrapper,
    KnowledgeLayer,
    init_sqlite,
    _row_to_dict,
)

logger = logging.getLogger(__name__)


class UnifiedStore:
    """统一数据存储层

    封装 SQLite 与 ChromaDB 的所有读写操作，对外提供一致的接口。
    采用分层设计：schema.py 定义表结构和向量集合，本类负责业务操作。

    Attributes:
        db_path: SQLite 文件路径。
        vector_path: ChromaDB 持久化目录。
        conn: 活跃的 SQLite 连接。
        vectors: ChromaDBWrapper 实例。
    """

    def __init__(
        self,
        db_path: str = "data/babyagent.db",
        vector_path: str = "data/vectors/",
    ) -> None:
        """初始化统一存储。

        Args:
            db_path: SQLite 数据库文件路径。
            vector_path: ChromaDB 向量持久化目录。
        """
        self.db_path = db_path
        self.vector_path = vector_path
        self.conn: sqlite3.Connection | None = None
        self.vectors: ChromaDBWrapper | None = None

    # ---- 初始化 ----

    def init_db(self) -> None:
        """初始化所有数据库资源：创建 SQLite 表 + ChromaDB 集合。

        幂等操作，重复调用不会破坏已有数据。
        """
        self.conn = init_sqlite(self.db_path)
        self.vectors = ChromaDBWrapper(persist_path=self.vector_path)
        logger.info("UnifiedStore 已初始化: sqlite=%s, vectors=%s", self.db_path, self.vector_path)

    def _ensure_init(self) -> None:
        """确保已初始化，否则自动初始化。"""
        if self.conn is None or self.vectors is None:
            self.init_db()

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("SQLite 连接已关闭")

    # ---- 向量知识搜索 ----

    def search_knowledge(
        self,
        query: str,
        layers: list[KnowledgeLayer] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """跨层级搜索知识向量（按 R29 不返回层级标签）。

        Args:
            query: 搜索查询文本。
            layers: 目标层级。None 表示全部。
            top_k: 结果数量上限。

        Returns:
            结果列表，每项不含 layer 字段。
        """
        self._ensure_init()
        return self.vectors.search(query=query, layers=layers, top_k=top_k)

    # ---- 产品检索 ----

    def get_products_by_category(
        self,
        category: str,
        store_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """按分类查询产品，可选按门店过滤。

        Args:
            category: 产品分类名称。
            store_id: 门店 ID，为 None 则不限制。

        Returns:
            产品字典列表。
        """
        self._ensure_init()
        if store_id:
            rows = self.conn.execute(
                "SELECT * FROM products WHERE category = ? AND store_id = ? ORDER BY name",
                (category, store_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM products WHERE category = ? ORDER BY name",
                (category,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search_products(
        self,
        query: str,
        top_k: int = 5,
        baby_profile: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量搜索产品描述，可选按宝宝月龄过滤。

        流程：
        1. 向量搜索产品描述文本
        2. 若提供了 baby_profile，按 suitable_age_min_months / suitable_age_max_months
           过滤出适合宝宝月龄的产品

        Args:
            query: 产品搜索查询文本。
            top_k: 返回数量上限。
            baby_profile: 宝宝档案（含 age_months）。为 None 则不按年龄过滤。

        Returns:
            过滤后的产品字典列表。
        """
        self._ensure_init()

        # 1) 借助向量搜索找到匹配的产品描述
        vector_results = self.vectors.search(query=query, layers=[KnowledgeLayer.ENTERPRISE], top_k=top_k * 3)

        # 2) 收集产品 ID，批量查询
        if not vector_results:
            return []

        product_ids = [r["id"] for r in vector_results if r.get("id")]
        if not product_ids:
            return []

        placeholders = ",".join(["?" for _ in product_ids])
        rows = self.conn.execute(
            f"SELECT * FROM products WHERE id IN ({placeholders})",
            product_ids,
        ).fetchall()

        products = [_row_to_dict(r) for r in rows]

        # 3) 按月龄过滤
        if baby_profile and baby_profile.get("age_months") is not None:
            age = baby_profile["age_months"]
            products = [
                p for p in products
                if (
                    (p.get("suitable_age_min_months") is None or age >= p["suitable_age_min_months"])
                    and (p.get("suitable_age_max_months") is None or age <= p["suitable_age_max_months"])
                )
            ]

        # 保持向量搜索的顺序
        id_order = {pid: idx for idx, pid in enumerate(product_ids)}
        products.sort(key=lambda p: id_order.get(p["id"], len(products)))

        return products[:top_k]

    # ---- 宝宝 CRUD ----

    def create_baby(
        self,
        store_id: str,
        name: str,
        gender: str,
        birth_date: str,
        dietary_restrictions: list[str] | None = None,
        notes_free_text: str = "",
    ) -> dict[str, Any]:
        """新建宝宝档案。

        Args:
            store_id: 门店 ID。
            name: 宝宝姓名/昵称。
            gender: 性别 ('male' / 'female' / 'unknown')。
            birth_date: 出生日期，格式 YYYY-MM-DD。
            dietary_restrictions: 饮食限制列表。
            notes_free_text: 自由文本备注。

        Returns:
            新创建的宝宝记录字典。
        """
        self._ensure_init()
        baby_id = str(uuid.uuid4())
        restrictions_json = json.dumps(dietary_restrictions or [], ensure_ascii=False)

        self.conn.execute(
            """INSERT INTO babies (id, store_id, name, gender, birth_date, dietary_restrictions, notes_free_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (baby_id, store_id, name, gender, birth_date, restrictions_json, notes_free_text),
        )
        self.conn.commit()
        logger.info("宝宝已创建: id=%s, name=%s, store=%s", baby_id, name, store_id)
        return self.get_baby(baby_id)

    def get_baby(self, baby_id: str) -> dict[str, Any] | None:
        """按 ID 获取宝宝档案。

        Args:
            baby_id: 宝宝 ID。

        Returns:
            宝宝记录字典，不存在时返回 None。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT * FROM babies WHERE id = ?",
            (baby_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def list_babies(
        self,
        store_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出宝宝列表，可选按门店过滤。

        Args:
            store_id: 门店 ID，为 None 则返回全部。

        Returns:
            宝宝记录列表。
        """
        self._ensure_init()
        if store_id:
            rows = self.conn.execute(
                "SELECT * FROM babies WHERE store_id = ? ORDER BY created_at DESC",
                (store_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM babies ORDER BY created_at DESC",
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_baby(
        self,
        baby_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        """乐观锁更新宝宝档案。

        Args:
            baby_id: 宝宝 ID。
            updates: 要更新的字段字典。可包含 name, gender, birth_date,
                     dietary_restrictions, notes_free_text。

        Returns:
            更新后的宝宝记录，不存在时返回 None。

        Raises:
            ValueError: 版本冲突（并发写）时抛出。
        """
        self._ensure_init()
        current = self.get_baby(baby_id)
        if current is None:
            return None

        # 乐观锁检查
        expected_version = updates.get("_expected_version", current["version"])
        if expected_version != current["version"]:
            raise ValueError(
                f"版本冲突: baby_id={baby_id}, expected={expected_version}, actual={current['version']}"
            )

        # 构建 SET 子句
        set_parts = []
        params = []

        allowed_fields = {"name", "gender", "birth_date", "dietary_restrictions", "notes_free_text"}
        for field in allowed_fields:
            if field in updates:
                value = updates[field]
                if field == "dietary_restrictions" and isinstance(value, list):
                    value = json.dumps(value, ensure_ascii=False)
                set_parts.append(f"{field} = ?")
                params.append(value)

        if not set_parts:
            return current

        set_parts.append("version = version + 1")
        set_parts.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        params.extend([baby_id, expected_version])

        cursor = self.conn.execute(
            f"""UPDATE babies SET {', '.join(set_parts)}
               WHERE id = ? AND version = ?""",
            params,
        )
        self.conn.commit()

        if cursor.rowcount == 0:
            raise ValueError(f"更新失败: baby_id={baby_id} 版本冲突或记录不存在")

        logger.info("宝宝已更新: id=%s, fields=%s", baby_id, list(updates.keys()))
        return self.get_baby(baby_id)

    def delete_baby(self, baby_id: str) -> bool:
        """软删除宝宝档案（实际做标记处理）。

        Args:
            baby_id: 宝宝 ID。

        Returns:
            是否成功删除。
        """
        self._ensure_init()
        cursor = self.conn.execute(
            "DELETE FROM babies WHERE id = ?",
            (baby_id,),
        )
        self.conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("宝宝已删除: id=%s", baby_id)
        return deleted

    # ---- 过敏记录（R34: append-only） ----

    def record_baby_allergy(
        self,
        baby_id: str,
        allergen: str,
        source: str,
        session_id: str | None = None,
        discovered_date: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """追加宝宝过敏记录（R34: 只允许追加，不允许删除）。

        Args:
            baby_id: 宝宝 ID。
            allergen: 过敏原名称。
            source: 来源 ('human' 或 'llm_append')。
            session_id: 关联的会话 ID。
            discovered_date: 发现日期，默认今天。
            notes: 备注。

        Returns:
            新创建的过敏记录字典。

        Raises:
            ValueError: source 无效或尝试删除。
        """
        self._ensure_init()

        if source not in ("human", "llm_append"):
            raise ValueError(f"过敏来源无效: {source}，必须是 'human' 或 'llm_append'")

        record_id = str(uuid.uuid4())
        discovered = discovered_date or date.today().isoformat()

        self.conn.execute(
            """INSERT INTO baby_allergy_history (id, baby_id, allergen, source, discovered_date, notes, session_id_ref)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (record_id, baby_id, allergen, source, discovered, notes, session_id),
        )
        self.conn.commit()
        logger.info("过敏记录已追加: baby_id=%s, allergen=%s, source=%s", baby_id, allergen, source)
        return self._get_allergy_record(record_id)

    def mark_allergy_removed(self, allergy_id: str, by: str) -> dict[str, Any] | None:
        """标记过敏记录为已移除（仅 human 源可操作，R34）。

        Args:
            allergy_id: 过敏记录 ID。
            by: 操作者标识 ('human' / 'llm_append')。

        Returns:
            更新后的记录；未找到时返回 None。

        Raises:
            ValueError: 非 human 源尝试移除时抛出。
        """
        self._ensure_init()

        if by != "human":
            raise ValueError(
                "仅人工操作 (source='human') 可标记过敏记录为已移除，"
                f"当前来源: {by}"
            )

        self.conn.execute(
            """UPDATE baby_allergy_history SET removed = 1 WHERE id = ? AND source = 'human'""",
            (allergy_id,),
        )
        self.conn.commit()
        logger.info("过敏记录已标记为移除: id=%s", allergy_id)
        return self._get_allergy_record(allergy_id)

    def get_allergies_for_baby(
        self,
        baby_id: str,
        include_removed: bool = False,
    ) -> list[dict[str, Any]]:
        """获取某宝宝的所有过敏记录。

        Args:
            baby_id: 宝宝 ID。
            include_removed: 是否包含已移除的记录。

        Returns:
            过敏记录列表。
        """
        self._ensure_init()
        if include_removed:
            rows = self.conn.execute(
                "SELECT * FROM baby_allergy_history WHERE baby_id = ? ORDER BY discovered_date DESC",
                (baby_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM baby_allergy_history WHERE baby_id = ? AND removed = 0 ORDER BY discovered_date DESC",
                (baby_id,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def _get_allergy_record(self, record_id: str) -> dict[str, Any] | None:
        """内部：按 ID 获取过敏记录。"""
        row = self.conn.execute(
            "SELECT * FROM baby_allergy_history WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    # ---- 备注 ----

    def get_notes_for_baby(self, baby_id: str) -> list[dict[str, Any]]:
        """获取宝宝的所有 LLM 备注记录。

        Args:
            baby_id: 宝宝 ID。

        Returns:
            备注列表（按时间升序）。
        """
        self._ensure_init()
        rows = self.conn.execute(
            "SELECT * FROM baby_notes WHERE baby_id = ? ORDER BY created_at ASC",
            (baby_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def add_note(self, baby_id: str, session_id: str, content: str) -> dict[str, Any]:
        """为宝宝添加一条 LLM 生成的备注。

        Args:
            baby_id: 宝宝 ID。
            session_id: 关联的会话 ID。
            content: 备注内容。

        Returns:
            新创建的备注记录。
        """
        self._ensure_init()
        note_id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT INTO baby_notes (id, baby_id, session_id, content)
               VALUES (?, ?, ?, ?)""",
            (note_id, baby_id, session_id, content),
        )
        self.conn.commit()
        logger.info("备注已添加: baby_id=%s, session=%s", baby_id, session_id)

        row = self.conn.execute(
            "SELECT * FROM baby_notes WHERE id = ?", (note_id,)
        ).fetchone()
        return _row_to_dict(row)

    # ---- 生长发育记录 ----

    def add_growth_record(
        self,
        baby_id: str,
        record_date: str,
        height_cm: float | None = None,
        weight_kg: float | None = None,
    ) -> dict[str, Any]:
        """添加宝宝生长发育记录。

        Args:
            baby_id: 宝宝 ID。
            record_date: 记录日期 (YYYY-MM-DD)。
            height_cm: 身高（厘米）。
            weight_kg: 体重（千克）。

        Returns:
            新建的记录字典。
        """
        self._ensure_init()
        record_id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT INTO baby_growth_records (id, baby_id, record_date, height_cm, weight_kg)
               VALUES (?, ?, ?, ?, ?)""",
            (record_id, baby_id, record_date, height_cm, weight_kg),
        )
        self.conn.commit()
        logger.info("生长记录已添加: baby=%s, date=%s", baby_id, record_date)

        row = self.conn.execute(
            "SELECT * FROM baby_growth_records WHERE id = ?", (record_id,)
        ).fetchone()
        return _row_to_dict(row)

    def get_growth_records(self, baby_id: str) -> list[dict[str, Any]]:
        """获取宝宝的生长发育记录（按日期倒序）。

        Args:
            baby_id: 宝宝 ID。

        Returns:
            生长记录列表。
        """
        self._ensure_init()
        rows = self.conn.execute(
            "SELECT * FROM baby_growth_records WHERE baby_id = ? ORDER BY record_date DESC",
            (baby_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ---- 产品 CRUD ----

    def create_product(
        self,
        store_id: str,
        name: str,
        category: str = "",
        description: str = "",
        suitable_age_min_months: int | None = None,
        suitable_age_max_months: int | None = None,
        ingredients: str = "",
        price: float | None = None,
    ) -> dict[str, Any]:
        """创建新产品记录。

        Args:
            store_id: 门店 ID。
            name: 产品名称。
            category: 产品分类。
            description: 产品描述。
            suitable_age_min_months: 适用最小月龄。
            suitable_age_max_months: 适用最大月龄。
            ingredients: 成分说明。
            price: 价格。

        Returns:
            新建的产品字典。
        """
        self._ensure_init()
        product_id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT INTO products (id, store_id, name, category, description,
               suitable_age_min_months, suitable_age_max_months, ingredients, price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (product_id, store_id, name, category, description,
             suitable_age_min_months, suitable_age_max_months, ingredients, price),
        )
        self.conn.commit()
        logger.info("产品已创建: id=%s, name=%s", product_id, name)

        row = self.conn.execute(
            "SELECT * FROM products WHERE id = ?", (product_id,)
        ).fetchone()
        return _row_to_dict(row)

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        """按 ID 获取产品。

        Args:
            product_id: 产品 ID。

        Returns:
            产品字典，不存在时为 None。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT * FROM products WHERE id = ?",
            (product_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    # ---- 员工 CRUD ----

    def create_employee(
        self,
        store_id: str,
        weixin_user_id: str,
        name: str,
        role: str = "staff",
    ) -> dict[str, Any]:
        """创建新员工记录。

        Args:
            store_id: 门店 ID。
            weixin_user_id: 微信用户 ID。
            name: 员工姓名。
            role: 角色 ('admin' / 'manager' / 'staff')。

        Returns:
            新建的员工字典。
        """
        self._ensure_init()
        emp_id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT INTO employees (id, store_id, weixin_user_id, name, role)
               VALUES (?, ?, ?, ?, ?)""",
            (emp_id, store_id, weixin_user_id, name, role),
        )
        self.conn.commit()
        logger.info("员工已创建: id=%s, name=%s", emp_id, name)

        row = self.conn.execute(
            "SELECT * FROM employees WHERE id = ?", (emp_id,)
        ).fetchone()
        return _row_to_dict(row)

    def get_employee(self, employee_id: str) -> dict[str, Any] | None:
        """按 ID 获取员工信息。

        Args:
            employee_id: 员工 ID。

        Returns:
            员工字典，不存在时为 None。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT * FROM employees WHERE id = ?",
            (employee_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_employee_by_weixin(self, weixin_user_id: str) -> dict[str, Any] | None:
        """按微信用户 ID 获取员工信息。

        Args:
            weixin_user_id: 微信用户 ID。

        Returns:
            员工字典，不存在时为 None。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT * FROM employees WHERE weixin_user_id = ?",
            (weixin_user_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    # ---- 种子同步（R26） ----

    def seed_sync(
        self,
        new_version: int,
        deltas: list[dict[str, Any]],
        full_rebuild: bool = False,
    ) -> None:
        """处理种子知识同步（R26）。

        根据同步策略执行增量或全量更新：
        - full_rebuild=True: 清空种子层并全量重建
        - full_rebuild=False: 按 deltas 列表执行增量更新

        Args:
            new_version: 新种子版本号。
            deltas: 变更数据列表。每项为 {'id', 'document', 'metadata', 'operation'}，
                    operation 为 'add' / 'update' / 'remove'。
            full_rebuild: 是否全量重建。
        """
        self._ensure_init()

        # 更新版本号
        self.conn.execute(
            "INSERT OR REPLACE INTO seed_metadata (key, value) VALUES (?, ?)",
            ("seed_version", str(new_version)),
        )
        self.conn.commit()

        if full_rebuild:
            docs = [d["document"] for d in deltas]
            ids_list = [d.get("id", str(uuid.uuid4())) for d in deltas]
            metadatas = [d.get("metadata", {}) for d in deltas]
            self.vectors.rebuild_from_docs(KnowledgeLayer.SEED, docs, metadatas=metadatas, ids=ids_list)
            logger.info("种子同步(全量): version=%d, docs=%d", new_version, len(docs))
            return

        # 增量更新
        for delta in deltas:
            op = delta.get("operation", "add")
            if op == "remove":
                # 单个删除非 remove 的文档
                _id = delta.get("id")
                if _id:
                    try:
                        collection = self.vectors._collections.get(KnowledgeLayer.SEED)
                        if collection:
                            collection.delete(ids=[_id])
                    except Exception:
                        pass
            elif op in ("add", "update"):
                _id = delta.get("id", str(uuid.uuid4()))
                doc = delta.get("document", "")
                meta = delta.get("metadata", {})
                if op == "update":
                    # 先删后加
                    try:
                        collection = self.vectors._collections.get(KnowledgeLayer.SEED)
                        if collection:
                            collection.delete(ids=[_id])
                    except Exception:
                        pass
                self.vectors.add_documents(KnowledgeLayer.SEED, [doc], metadatas=[meta], ids=[_id])

        logger.info("种子同步(增量): version=%d, deltas=%d", new_version, len(deltas))

    def get_seed_version(self) -> int:
        """获取当前种子数据版本号。

        Returns:
            版本号整数，未设置时返回 0。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT value FROM seed_metadata WHERE key = 'seed_version'",
        ).fetchone()
        if row is None:
            return 0
        return int(row["value"])

    # ---- 企业/门店基础操作 ----

    def create_enterprise(self, name: str, data_scope: str = "store") -> dict[str, Any]:
        """创建企业记录。

        Args:
            name: 企业名称。
            data_scope: 数据作用域 ('enterprise'/'store'/'store_strict')。

        Returns:
            新建的企业字典。
        """
        self._ensure_init()
        ent_id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT INTO enterprises (id, name, data_scope)
               VALUES (?, ?, ?)""",
            (ent_id, name, data_scope),
        )
        self.conn.commit()
        logger.info("企业已创建: id=%s, name=%s", ent_id, name)

        row = self.conn.execute(
            "SELECT * FROM enterprises WHERE id = ?", (ent_id,)
        ).fetchone()
        return _row_to_dict(row)

    def create_store(self, enterprise_id: str, name: str) -> dict[str, Any]:
        """创建门店记录。

        Args:
            enterprise_id: 所属企业 ID。
            name: 门店名称。

        Returns:
            新建的门店字典。
        """
        self._ensure_init()
        store_id = str(uuid.uuid4())

        self.conn.execute(
            """INSERT INTO stores (id, enterprise_id, name)
               VALUES (?, ?, ?)""",
            (store_id, enterprise_id, name),
        )
        self.conn.commit()
        logger.info("门店已创建: id=%s, name=%s", store_id, name)

        row = self.conn.execute(
            "SELECT * FROM stores WHERE id = ?", (store_id,)
        ).fetchone()
        return _row_to_dict(row)

    def get_store(self, store_id: str) -> dict[str, Any] | None:
        """按 ID 获取门店信息。"""
        self._ensure_init()
        row = self.conn.execute(
            "SELECT * FROM stores WHERE id = ?", (store_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def get_enterprise(self, enterprise_id: str) -> dict[str, Any] | None:
        """按 ID 获取企业信息。"""
        self._ensure_init()
        row = self.conn.execute(
            "SELECT * FROM enterprises WHERE id = ?", (enterprise_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
