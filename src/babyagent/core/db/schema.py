"""SQLite 数据库 Schema + ChromaDB 向量检索封装

三层知识向量模型：
  层A (SEED)       — 总部种子知识库，只读，由 seed_sync 刷入
  层B (ENTERPRISE)  — 企业/门店拓展知识库，由管理员维护
  层C (RUNTIME)     — 运行时动态积累，LLM 学习与压缩产生

SQLite 基表：
  enterprises, stores, employees, babies, baby_growth_records,
  baby_allergy_history, baby_notes, products, product_categories, seed_metadata

ChromaDB 集合：
  seed_layer, enterprise_layer, runtime_layer
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局枚举
# ---------------------------------------------------------------------------


class KnowledgeLayer(IntEnum):
    """知识向量的三个层级"""
    SEED = 0        # 总部种子库（只读）
    ENTERPRISE = 1  # 企业/门店拓展库
    RUNTIME = 2     # 运行时积累


class AllergySource(str):
    """过敏记录来源枚举"""
    HUMAN = "human"       # 人工手动录入
    LLM_APPEND = "llm_append"  # LLM 自动追加


# ---------------------------------------------------------------------------
# 常量：ChromaDB 集合名映射
# ---------------------------------------------------------------------------

_LAYER_COLLECTION_MAP: dict[KnowledgeLayer, str] = {
    KnowledgeLayer.SEED: "seed_layer",
    KnowledgeLayer.ENTERPRISE: "enterprise_layer",
    KnowledgeLayer.RUNTIME: "runtime_layer",
}

# 反向查找
_COLLECTION_LAYER_MAP: dict[str, KnowledgeLayer] = {v: k for k, v in _LAYER_COLLECTION_MAP.items()}

# ---------------------------------------------------------------------------
# SQLite DDL
# ---------------------------------------------------------------------------

SQL_CREATE_ENTERPRISES = """
CREATE TABLE IF NOT EXISTS enterprises (
    id          TEXT PRIMARY KEY,
    name        TEXT    NOT NULL,
    data_scope  TEXT    NOT NULL DEFAULT 'store' CHECK(data_scope IN ('enterprise','store','store_strict')),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SQL_CREATE_STORES = """
CREATE TABLE IF NOT EXISTS stores (
    id              TEXT PRIMARY KEY,
    enterprise_id   TEXT    NOT NULL REFERENCES enterprises(id),
    name            TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SQL_CREATE_EMPLOYEES = """
CREATE TABLE IF NOT EXISTS employees (
    id              TEXT PRIMARY KEY,
    store_id        TEXT    NOT NULL REFERENCES stores(id),
    weixin_user_id  TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    role            TEXT    NOT NULL DEFAULT 'staff' CHECK(role IN ('admin','manager','staff')),
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SQL_CREATE_BABIES = """
CREATE TABLE IF NOT EXISTS babies (
    id                      TEXT PRIMARY KEY,
    store_id                TEXT    NOT NULL REFERENCES stores(id),
    name                    TEXT    NOT NULL,
    gender                  TEXT    NOT NULL CHECK(gender IN ('male','female','unknown')),
    birth_date              TEXT    NOT NULL,
    dietary_restrictions    TEXT    NOT NULL DEFAULT '[]',
    notes_free_text         TEXT    NOT NULL DEFAULT '',
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    version                 INTEGER NOT NULL DEFAULT 1
);
"""

SQL_CREATE_BABY_GROWTH_RECORDS = """
CREATE TABLE IF NOT EXISTS baby_growth_records (
    id          TEXT PRIMARY KEY,
    baby_id     TEXT    NOT NULL REFERENCES babies(id),
    record_date TEXT    NOT NULL,
    height_cm   REAL,
    weight_kg   REAL
);
"""

SQL_CREATE_BABY_ALLERGY_HISTORY = """
CREATE TABLE IF NOT EXISTS baby_allergy_history (
    id              TEXT PRIMARY KEY,
    baby_id         TEXT    NOT NULL REFERENCES babies(id),
    allergen        TEXT    NOT NULL,
    source          TEXT    NOT NULL CHECK(source IN ('human','llm_append')),
    discovered_date TEXT    NOT NULL DEFAULT (date('now')),
    notes           TEXT    NOT NULL DEFAULT '',
    session_id_ref  TEXT,
    removed         INTEGER NOT NULL DEFAULT 0 CHECK(removed IN (0,1))
);
"""

SQL_CREATE_BABY_NOTES = """
CREATE TABLE IF NOT EXISTS baby_notes (
    id          TEXT PRIMARY KEY,
    baby_id     TEXT    NOT NULL REFERENCES babies(id),
    session_id  TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SQL_CREATE_PRODUCTS = """
CREATE TABLE IF NOT EXISTS products (
    id                      TEXT PRIMARY KEY,
    store_id                TEXT    NOT NULL REFERENCES stores(id),
    name                    TEXT    NOT NULL,
    category                TEXT    NOT NULL DEFAULT '',
    description             TEXT    NOT NULL DEFAULT '',
    suitable_age_min_months INTEGER,
    suitable_age_max_months INTEGER,
    ingredients             TEXT    NOT NULL DEFAULT '',
    price                   REAL
);
"""

SQL_CREATE_PRODUCT_CATEGORIES = """
CREATE TABLE IF NOT EXISTS product_categories (
    id          TEXT PRIMARY KEY,
    name        TEXT    NOT NULL,
    parent_id   TEXT    REFERENCES product_categories(id)
);
"""

SQL_CREATE_SEED_METADATA = """
CREATE TABLE IF NOT EXISTS seed_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# 所有建表语句列表
ALL_CREATE_STMTS = [
    SQL_CREATE_ENTERPRISES,
    SQL_CREATE_STORES,
    SQL_CREATE_EMPLOYEES,
    SQL_CREATE_BABIES,
    SQL_CREATE_BABY_GROWTH_RECORDS,
    SQL_CREATE_BABY_ALLERGY_HISTORY,
    SQL_CREATE_BABY_NOTES,
    SQL_CREATE_PRODUCTS,
    SQL_CREATE_PRODUCT_CATEGORIES,
    SQL_CREATE_SEED_METADATA,
]

# ---------------------------------------------------------------------------
# ChromaDB 包装器
# ---------------------------------------------------------------------------


class ChromaDBWrapper:
    """ChromaDB 向量检索包装器

    使用 sentence-transformers 中文嵌入模型，提供三个知识层级的增删查能力。
    默认模型: shibing624/text2vec-base-chinese（轻量中文语义嵌入）。

    Attributes:
        client: ChromaDB PersistentClient 实例
        collections: 三层集合的字典映射
        embedding_fn: 嵌入函数
        model_name: 当前加载的嵌入模型名
    """

    DEFAULT_MODEL = "shibing624/text2vec-base-chinese"
    ALTERNATE_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(
        self,
        persist_path: str = "data/vectors/",
        model_name: str | None = None,
    ) -> None:
        """初始化 ChromaDB 客户端及嵌入模型。

        Args:
            persist_path: ChromaDB 本地持久化目录路径。
            model_name: 嵌入模型名称，为 None 时使用默认模型。
        """
        self._persist_path = persist_path
        self._model_name = model_name or self.DEFAULT_MODEL
        self._client: chromadb.PersistentClient | None = None
        self._collections: dict[KnowledgeLayer, Any] = {}
        self._embedding_fn = None

        self._init_client()

    # ---- 内部初始化 ----

    def _init_client(self) -> None:
        """初始化 ChromaDB 持久化客户端并加载嵌入模型。"""
        Path(self._persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self._persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._init_embedding()
        self._init_collections()
        logger.info("ChromaDBWrapper 初始化完成: persist=%s, model=%s", self._persist_path, self._model_name)

    def _init_embedding(self) -> None:
        """加载 sentence-transformers 嵌入模型。"""
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=self._model_name,
            )
            logger.debug("嵌入模型已加载: %s", self._model_name)
        except Exception as exc:
            logger.error("无法加载嵌入模型 %s: %s", self._model_name, exc)
            raise RuntimeError(f"ChromaDB 嵌入模型加载失败: {exc}") from exc

    def _init_collections(self) -> None:
        """创建或获取 ChromaDB 三个层级的集合。"""
        if self._client is None:
            raise RuntimeError("ChromaDB 客户端未初始化")

        for layer in KnowledgeLayer:
            coll_name = _LAYER_COLLECTION_MAP[layer]
            try:
                collection = self._client.get_or_create_collection(
                    name=coll_name,
                    embedding_function=self._embedding_fn,
                )
                self._collections[layer] = collection
                logger.debug("ChromaDB 集合已就绪: %s", coll_name)
            except Exception as exc:
                logger.error("创建集合 %s 失败: %s", coll_name, exc)
                raise

    # ---- 公开方法 ----

    def add_documents(
        self,
        layer: KnowledgeLayer,
        docs: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """向指定层级添加文档向量。

        Args:
            layer: 知识层级 (SEED / ENTERPRISE / RUNTIME)。
            docs: 要添加的文档文本列表。
            metadatas: 每条文档的元数据字典列表，长度需与 docs 一致。
            ids: 自定义文档 ID 列表。省略时自动生成 UUID。

        Raises:
            KeyError: layer 无效时抛出。
        """
        collection = self._collections.get(layer)
        if collection is None:
            raise KeyError(f"无效的知识层级: {layer}")

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in docs]

        if metadatas is None:
            metadatas = [{} for _ in docs]

        try:
            collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("层级 %s 已添加 %d 条文档", layer.name, len(docs))
        except Exception as exc:
            logger.error("层级 %s 添加文档失败: %s", layer.name, exc)
            raise

    def search(
        self,
        query: str,
        layers: list[KnowledgeLayer] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """跨层级搜索知识向量（按 R29 不返回层级标签）。

        Args:
            query: 搜索查询文本。
            layers: 要搜索的层级列表。为 None 时搜索全部三层。
            top_k: 返回结果数量上限。

        Returns:
            每项包含 {'id', 'document', 'metadata', 'distance'} 的结果列表，
            按相似度降序排列。
        """
        if layers is None:
            layers = list(KnowledgeLayer)

        results: dict[str, tuple[float, dict[str, Any]]] = {}

        for layer in layers:
            collection = self._collections.get(layer)
            if collection is None:
                continue

            try:
                layer_results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                )
                if layer_results and layer_results.get("ids") and layer_results["ids"][0]:
                    for idx, doc_id in enumerate(layer_results["ids"][0]):
                        distance = (
                            float(layer_results["distances"][0][idx])
                            if layer_results.get("distances") and layer_results["distances"][0]
                            else 0.0
                        )
                        document = (
                            layer_results["documents"][0][idx]
                            if layer_results.get("documents") and layer_results["documents"][0]
                            else ""
                        )
                        metadata = (
                            layer_results["metadatas"][0][idx]
                            if layer_results.get("metadatas") and layer_results["metadatas"][0]
                            else {}
                        )

                        # 保留每个 doc_id 的最佳匹配
                        if doc_id not in results or distance < results[doc_id][0]:
                            results[doc_id] = (distance, {
                                "id": doc_id,
                                "document": document,
                                "metadata": metadata,
                                "distance": distance,
                            })

            except Exception as exc:
                logger.warning("层级 %s 搜索异常: %s", layer.name, exc)
                continue

        # 按 distance 排序（越小越相似）并截取 top_k
        sorted_results = sorted(results.values(), key=lambda x: x[0])
        return [item[1] for item in sorted_results[:top_k]]

    def rebuild_from_docs(
        self,
        layer: KnowledgeLayer,
        docs: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """重建指定层级：删除全部后重新添加。

        Args:
            layer: 要重建的层级。
            docs: 新文档列表。
            metadatas: 可选元数据列表。
            ids: 可选 ID 列表。
        """
        self.delete_layer(layer)
        self.add_documents(layer, docs, metadatas=metadatas, ids=ids)
        logger.info("层级 %s 已重建，共 %d 条文档", layer.name, len(docs))

    def delete_layer(self, layer: KnowledgeLayer) -> None:
        """清空指定层级的所有向量数据。

        Args:
            layer: 要清空的层级。

        Raises:
            KeyError: layer 无效时抛出。
        """
        collection = self._collections.get(layer)
        if collection is None:
            raise KeyError(f"无效的知识层级: {layer}")

        try:
            # 获取所有 ID 后批量删除
            existing = collection.get()
            if existing and existing.get("ids") and len(existing["ids"]) > 0:
                collection.delete(ids=existing["ids"])
                logger.info("层级 %s 已清空 %d 条文档", layer.name, len(existing["ids"]))
            else:
                logger.debug("层级 %s 为空，无需清空", layer.name)
        except Exception as exc:
            logger.error("清空层级 %s 失败: %s", layer.name, exc)
            raise

    def count(self, layer: KnowledgeLayer | None = None) -> int:
        """获取指定层级或全部层级的文档总数。

        Args:
            layer: 目标层级。为 None 时返回全部。

        Returns:
            文档数量。
        """
        if layer is not None:
            collection = self._collections.get(layer)
            if collection is None:
                return 0
            return collection.count()
        return sum(c.count() for c in self._collections.values())


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def init_sqlite(db_path: str) -> sqlite3.Connection:
    """初始化 SQLite 数据库并创建所有基表。

    Args:
        db_path: SQLite 数据库文件路径。

    Returns:
        已就绪的 sqlite3.Connection，启用 WAL 模式和 foreign_keys。
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    for stmt in ALL_CREATE_STMTS:
        conn.execute(stmt)

    conn.commit()
    logger.info("SQLite 数据库已初始化: %s (共 %d 张表)", db_path, len(ALL_CREATE_STMTS))
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """将 sqlite3.Row 转换为普通 dict。"""
    if row is None:
        return {}
    return dict(row)
