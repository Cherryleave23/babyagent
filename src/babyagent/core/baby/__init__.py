"""宝宝档案管理

指令驱动(规则)写入 + LLM 只读。
备注机制 + 跨 Session 档案同步 + 宝宝快速切换。
"""

from .profile import BabyProfile, BabyProfileManager, GrowthRecord, AllergyEntry, NoteEntry
from .compression import (
    CompressionOutput,
    BabyProfileUpdate,
    generate_compression,
    build_empty_output,
)

__all__ = [
    "BabyProfile",
    "BabyProfileManager",
    "GrowthRecord",
    "AllergyEntry",
    "NoteEntry",
    "CompressionOutput",
    "BabyProfileUpdate",
    "generate_compression",
    "build_empty_output",
]
