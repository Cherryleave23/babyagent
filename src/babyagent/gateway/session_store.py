"""BabyAgent 会话存储 — 轻量 SQLite 会话隔离

基于 SQLite 的简化会话存储，用于管理微信员工的对话状态与宝宝上下文。
不使用 Hermes 的复杂 SessionStore 架构，而是专为 BabyAgent 场景定制。

设计约束：
  - 按 employee_wxid 隔离，一个员工一个 session（R41）
  - Session key 格式: "babyagent:weixin:dm:{employee_wxid}"
  - 支持宝宝切换、会话历史、上下文令牌管理
  - 纯 SQLite 实现，无额外依赖
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 会话 key 前缀（R41）
_SESSION_KEY_PREFIX = "babyagent:weixin:dm:"

# 默认保留的最近对话轮数
_DEFAULT_MAX_HISTORY_ROUNDS = 20

# 会话表 DDL
_CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_key   TEXT PRIMARY KEY,
    employee_wxid TEXT NOT NULL,
    store_id      TEXT NOT NULL DEFAULT '',
    active_baby_id    INTEGER,
    active_baby_name  TEXT,
    current_token     TEXT,
    conversation_history_json TEXT DEFAULT '[]',
    first_baby_message_sent  INTEGER DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# 索引
_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_wxid ON sessions(employee_wxid);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_store ON sessions(store_id);",
]


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class SessionContext:
    """会话上下文 — 封装一个员工的当前对话状态

    Attributes:
        session_key: 会话唯一标识，格式 "babyagent:weixin:dm:{employee_wxid}"。
        employee_wxid: 微信用户 ID。
        store_id: 门店 ID。
        active_baby_id: 当前活跃的宝宝 ID，无活跃宝宝时为 None。
        active_baby_name: 当前活跃的宝宝名字。
        conversation_history: 对话历史，格式 [(role, content), ...]。
        context_token: 微信上下文令牌，用于回复关联。
        first_baby_message_sent: 当前宝宝在本 session 中的首条消息是否已发送。
    """

    session_key: str = ""
    employee_wxid: str = ""
    store_id: str = ""
    active_baby_id: Optional[int] = None
    active_baby_name: Optional[str] = None
    conversation_history: list[tuple[str, str]] = field(default_factory=list)
    context_token: Optional[str] = None
    first_baby_message_sent: bool = False


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """轻量 SQLite 会话存储

    为 BabyAgent 场景定制的会话管理，不支持 Hermes 的多租户/复杂SessionStore。
    每个员工（employee_wxid）在同一门店下最多一个活跃 session。

    Attributes:
        db_path: SQLite 数据库文件路径。
        conn: 活跃的 SQLite 连接。
    """

    def __init__(self, db_path: str = "data/babyagent_sessions.db") -> None:
        """初始化会话存储。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    # ---- 初始化 ----

    def init(self) -> None:
        """创建会话表及索引。幂等操作，重复调用不会破坏已有数据。"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute(_CREATE_SESSIONS_TABLE)
        for idx_sql in _CREATE_INDEXES:
            self.conn.execute(idx_sql)
        self.conn.commit()
        logger.info("SessionStore 已初始化: db=%s", self.db_path)

    def _ensure_init(self) -> None:
        """确保数据库已初始化，否则自动初始化。"""
        if self.conn is None:
            self.init()

    def close(self) -> None:
        """关闭数据库连接。"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("SessionStore 连接已关闭")

    # ---- Session Key 构造 ----

    @staticmethod
    def _make_session_key(employee_wxid: str) -> str:
        """生成标准 session key（R41）。

        Args:
            employee_wxid: 微信用户 ID。

        Returns:
            格式化的 session key。
        """
        return f"{_SESSION_KEY_PREFIX}{employee_wxid}"

    # ---- CRUD ----

    def get_or_create_session(
        self,
        employee_wxid: str,
        store_id: str,
    ) -> SessionContext:
        """获取或创建会话上下文。

        如果该员工已有 session，加载并返回；否则创建新 session。
        创建 session 时会同时设置 active_baby_id 为 NULL（无默认宝宝）。

        Args:
            employee_wxid: 微信用户 ID。
            store_id: 门店 ID。

        Returns:
            SessionContext 实例。
        """
        self._ensure_init()
        session_key = self._make_session_key(employee_wxid)

        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_key = ?",
            (session_key,),
        ).fetchone()

        if row is not None:
            logger.debug("加载已有会话: key=%s", session_key)
            return _row_to_context(row)

        # 创建新 session
        now = datetime.now().isoformat()
        self.conn.execute(
            """INSERT INTO sessions (session_key, employee_wxid, store_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_key, employee_wxid, store_id, now, now),
        )
        self.conn.commit()
        logger.info("新会话已创建: key=%s, wxid=%s, store=%s", session_key, employee_wxid, store_id)

        return SessionContext(
            session_key=session_key,
            employee_wxid=employee_wxid,
            store_id=store_id,
        )

    def set_active_baby(
        self,
        session_key: str,
        baby_id: int,
        baby_name: str,
    ) -> None:
        """设置当前活跃的宝宝。

        切换宝宝时重置 first_baby_message_sent 标记（R38），
        确保下次 LLM 调用时重新注入档案摘要。

        Args:
            session_key: 会话 key。
            baby_id: 宝宝 ID。
            baby_name: 宝宝名字。
        """
        self._ensure_init()
        now = datetime.now().isoformat()
        self.conn.execute(
            """UPDATE sessions
               SET active_baby_id = ?, active_baby_name = ?, first_baby_message_sent = 0, updated_at = ?
               WHERE session_key = ?""",
            (baby_id, baby_name, now, session_key),
        )
        self.conn.commit()
        logger.info("活跃宝宝已设置: session=%s, baby_id=%s, name=%s", session_key, baby_id, baby_name)

    def get_active_baby(self, session_key: str) -> Optional[tuple[int, str]]:
        """获取当前活跃的宝宝信息。

        Args:
            session_key: 会话 key。

        Returns:
            (baby_id, baby_name) 元组，无活跃宝宝时返回 None。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT active_baby_id, active_baby_name FROM sessions WHERE session_key = ?",
            (session_key,),
        ).fetchone()

        if row is None or row["active_baby_id"] is None:
            return None

        return (row["active_baby_id"], row["active_baby_name"] or "")

    def is_first_baby_message(self, session_key: str) -> bool:
        """检查当前宝宝在本 session 中是否尚未发送过首条消息（R38）。

        用于 R37/R38：首次接触某宝宝时需注入档案摘要。

        Args:
            session_key: 会话 key。

        Returns:
            True 表示这是首次与当前宝宝对话（需注入档案摘要）。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT first_baby_message_sent FROM sessions WHERE session_key = ?",
            (session_key,),
        ).fetchone()

        if row is None:
            return True

        return not bool(row["first_baby_message_sent"])

    def mark_first_baby_message_sent(self, session_key: str) -> None:
        """标记当前宝宝的首次摘要已发送（R37/R38）。

        调用后 is_first_baby_message() 将返回 False，直到下一次切换宝宝。

        Args:
            session_key: 会话 key。
        """
        self._ensure_init()
        self.conn.execute(
            "UPDATE sessions SET first_baby_message_sent = 1 WHERE session_key = ?",
            (session_key,),
        )
        self.conn.commit()
        logger.debug("标记首条宝宝消息已发送: session=%s", session_key)

    # ---- 对话历史 ----

    def append_to_history(
        self,
        session_key: str,
        role: str,
        content: str,
    ) -> None:
        """向对话历史追加一条消息。

        Args:
            session_key: 会话 key。
            role: 角色（'user' 或 'assistant'）。
            content: 消息内容。
        """
        self._ensure_init()
        history = self._load_history(session_key)
        history.append((role, content))

        # 限制最大轮数
        if len(history) > _DEFAULT_MAX_HISTORY_ROUNDS * 2:
            history = history[-(_DEFAULT_MAX_HISTORY_ROUNDS * 2):]

        self._save_history(session_key, history)
        logger.debug("历史追加: session=%s, role=%s, len=%d chars", session_key, role, len(content))

    def get_history(self, session_key: str) -> list[tuple[str, str]]:
        """获取当前会话的对话历史。

        Args:
            session_key: 会话 key。

        Returns:
            [(role, content), ...] 列表。
        """
        self._ensure_init()
        return self._load_history(session_key)

    def clear_history(self, session_key: str) -> None:
        """清空当前会话的对话历史。

        通常在上下文压缩后调用，用压缩摘要替代原始历史。

        Args:
            session_key: 会话 key。
        """
        self._ensure_init()
        self._save_history(session_key, [])
        logger.info("历史已清空: session=%s", session_key)

    def replace_history(self, session_key: str, new_history: list[tuple[str, str]]) -> None:
        """用压缩后的摘要历史替换原始对话历史。

        在上下文压缩（R31）后调用，用于存放 session_summary。

        Args:
            session_key: 会话 key。
            new_history: 新的对话历史（通常为 [("system", summary)] 格式）。
        """
        self._ensure_init()
        self._save_history(session_key, new_history)
        logger.info("历史已替换为压缩摘要: session=%s, entries=%d", session_key, len(new_history))

    # ---- 上下文令牌 ----

    def update_token(self, session_key: str, context_token: str) -> None:
        """更新微信上下文令牌（用于消息回复关联）。

        Args:
            session_key: 会话 key。
            context_token: 微信上下文令牌。
        """
        self._ensure_init()
        now = datetime.now().isoformat()
        self.conn.execute(
            "UPDATE sessions SET current_token = ?, updated_at = ? WHERE session_key = ?",
            (context_token, now, session_key),
        )
        self.conn.commit()

    def get_token(self, session_key: str) -> Optional[str]:
        """获取当前微信上下文令牌。

        Args:
            session_key: 会话 key。

        Returns:
            上下文令牌字符串，不存在时为 None。
        """
        self._ensure_init()
        row = self.conn.execute(
            "SELECT current_token FROM sessions WHERE session_key = ?",
            (session_key,),
        ).fetchone()

        if row is None:
            return None
        return row["current_token"]

    # ---- 内部辅助 ----

    def _load_history(self, session_key: str) -> list[tuple[str, str]]:
        """从数据库加载对话历史 JSON。"""
        row = self.conn.execute(
            "SELECT conversation_history_json FROM sessions WHERE session_key = ?",
            (session_key,),
        ).fetchone()

        if row is None or not row["conversation_history_json"]:
            return []

        try:
            raw = json.loads(row["conversation_history_json"])
            if isinstance(raw, list):
                return [(item[0], item[1]) for item in raw if isinstance(item, list) and len(item) >= 2]
        except (json.JSONDecodeError, TypeError, IndexError) as exc:
            logger.warning("历史 JSON 解析失败: %s", exc)

        return []

    def _save_history(self, session_key: str, history: list[tuple[str, str]]) -> None:
        """将对话历史序列化为 JSON 并写入数据库。"""
        now = datetime.now().isoformat()
        json_data = json.dumps(history, ensure_ascii=False)
        self.conn.execute(
            "UPDATE sessions SET conversation_history_json = ?, updated_at = ? WHERE session_key = ?",
            (json_data, now, session_key),
        )
        self.conn.commit()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _row_to_context(row: sqlite3.Row) -> SessionContext:
    """将 SQLite 行转换为 SessionContext 实例。

    Args:
        row: sqlite3.Row 对象。

    Returns:
        SessionContext 实例。
    """
    history: list[tuple[str, str]] = []
    history_json = row["conversation_history_json"]
    if history_json:
        try:
            raw = json.loads(history_json)
            if isinstance(raw, list):
                history = [(item[0], item[1]) for item in raw if isinstance(item, list) and len(item) >= 2]
        except (json.JSONDecodeError, TypeError):
            pass

    return SessionContext(
        session_key=row["session_key"],
        employee_wxid=row["employee_wxid"],
        store_id=row["store_id"] or "",
        active_baby_id=row["active_baby_id"],
        active_baby_name=row["active_baby_name"],
        conversation_history=history,
        context_token=row["current_token"],
        first_baby_message_sent=bool(row["first_baby_message_sent"]),
    )
