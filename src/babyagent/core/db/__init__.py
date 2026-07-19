"""端侧统一数据库

三层合并模型：层A(总部种子库) + 层B(企业拓展库) + 层C(运行时数据)。
SQLite + ChromaDB 向量检索。
"""

from .schema import (
    ChromaDBWrapper,
    KnowledgeLayer,
    init_sqlite,
    _row_to_dict,
    AllergySource,
)

from .unified_store import UnifiedStore

__all__ = [
    "ChromaDBWrapper",
    "KnowledgeLayer",
    "AllergySource",
    "init_sqlite",
    "_row_to_dict",
    "UnifiedStore",
]
