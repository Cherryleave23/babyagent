"""网关层 — WeChat Clawbot 适配器、会话隔离、宝宝切换、消息编排

提供 BabyAgent 与外部系统（微信企微机器人）对接的所有接口：

- SessionStore: 轻量 SQLite 会话存储，按员工微信 ID 隔离
- SessionContext: 会话上下文数据模型
- detect_baby_switch: 隐式/显式宝宝切换检测（R20-R23）
- process_message: 主消息处理编排器，串联全部管线
- parse_command: 解析 @Agent 结构化指令（规则驱动，非 LLM）
"""

from babyagent.gateway.session_store import SessionStore, SessionContext
from babyagent.gateway.baby_switch import detect_baby_switch
from babyagent.gateway.command_parser import parse_command, CommandResult
from babyagent.gateway.orchestrator import process_message

__all__ = [
    "SessionStore",
    "SessionContext",
    "detect_baby_switch",
    "parse_command",
    "CommandResult",
    "process_message",
]
