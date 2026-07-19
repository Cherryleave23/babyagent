"""BabyAgent Web 管理控制台 — REST API 路由

提供以下 API 端点：
  - CRUD: employees, babies, products
  - 系统状态与健康检查
  - 配置读写
  - 微信绑定 QR 码发起（占位）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# ---------- 全局依赖注入（由 create_app 设置） ----------

_unified_store: Any = None
_config: Any = None
_start_time: float = time.time()


def inject(store, cfg) -> None:
    """由 app.create_app 调用，注入 UnifiedStore 与 AppConfig 实例。"""
    global _unified_store, _config, _start_time
    _unified_store = store
    _config = cfg
    _start_time = time.time()


def _store():
    if _unified_store is None:
        raise HTTPException(500, "存储层未初始化")
    _unified_store._ensure_init()
    return _unified_store


def _cfg():
    if _config is None:
        raise HTTPException(500, "配置未加载")
    return _config


# ============================
# 系统状态 & 健康检查
# ============================

@router.get("/status")
def system_status(request=None) -> dict[str, Any]:
    """返回系统运行状态 JSON。"""
    try:
        store = _store()
        cfg = _cfg()
        uptime = time.time() - _start_time

        babies = store.list_babies() or []
        employees = store.conn.execute("SELECT COUNT(*) as cnt FROM employees").fetchone()
        products = store.conn.execute("SELECT COUNT(*) as cnt FROM products").fetchone()
        enterprises = store.conn.execute("SELECT COUNT(*) as cnt FROM enterprises").fetchone()

        seed_version = store.get_seed_version()

        return {
            "status": "ok",
            "uptime_seconds": round(uptime, 1),
            "uptime_human": _format_uptime(uptime),
            "timestamps": {
                "current": time.time(),
                "start": _start_time,
            },
            "counts": {
                "enterprises": enterprises["cnt"] if enterprises else 0,
                "employees": employees["cnt"] if employees else 0,
                "babies": len(babies),
                "products": products["cnt"] if products else 0,
                "seed_version": seed_version,
            },
            "config": {
                "enterprise_name": cfg.enterprise.name,
                "data_scope": cfg.data_scope,
                "model_provider": cfg.model.provider,
                "weixin_enabled": cfg.weixin.enabled,
                "seed_sync_enabled": cfg.seed_sync.enabled,
            },
            "session_count": _session_count(request),
        }
    except Exception as e:
        logger.exception("获取系统状态失败")
        return {"status": "error", "message": str(e)}


@router.get("/health")
def health() -> dict[str, str]:
    """简单健康检查。"""
    return {"status": "ok"}


# ============================
# 员工 CRUD
# ============================

@router.get("/employees")
def list_employees(store_id: Optional[str] = Query(None)) -> list[dict[str, Any]]:
    """列出所有员工，可选按门店过滤。"""
    store = _store()
    rows = None
    if store_id:
        rows = store.conn.execute(
            "SELECT e.*, s.name AS store_name FROM employees e LEFT JOIN stores s ON e.store_id = s.id WHERE e.store_id = ? ORDER BY e.created_at DESC",
            (store_id,),
        ).fetchall()
    else:
        rows = store.conn.execute(
            "SELECT e.*, s.name AS store_name FROM employees e LEFT JOIN stores s ON e.store_id = s.id ORDER BY e.created_at DESC",
        ).fetchall()
    return [_row_dict(r) for r in rows]


@router.post("/employees")
def create_employee(payload: dict[str, Any]) -> dict[str, Any]:
    """新增员工。"""
    store = _store()

    store_id = payload.get("store_id")
    weixin = payload.get("weixin_user_id", "")
    name = payload.get("name", "").strip()
    role = payload.get("role", "staff")

    if not name or not store_id:
        raise HTTPException(400, "员工姓名和门店 ID 为必填项")
    if role not in ("admin", "manager", "staff"):
        raise HTTPException(400, "角色无效")

    return store.create_employee(store_id=store_id, weixin_user_id=weixin, name=name, role=role)


@router.get("/employees/{employee_id}")
def get_employee(employee_id: str) -> dict[str, Any]:
    """按 ID 获取员工详情。"""
    store = _store()
    emp = store.get_employee(employee_id)
    if emp is None:
        raise HTTPException(404, "员工不存在")
    return emp


# ============================
# 宝宝 CRUD
# ============================

@router.get("/babies")
def list_babies(
    store_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    """列出宝宝档案，支持门店过滤和姓名搜索。"""
    store = _store()

    if store_id:
        rows = store.conn.execute(
            "SELECT b.*, s.name AS store_name FROM babies b LEFT JOIN stores s ON b.store_id = s.id WHERE b.store_id = ? ORDER BY b.created_at DESC",
            (store_id,),
        ).fetchall()
    else:
        rows = store.conn.execute(
            "SELECT b.*, s.name AS store_name FROM babies b LEFT JOIN stores s ON b.store_id = s.id ORDER BY b.created_at DESC",
        ).fetchall()

    results = [_row_dict(r) for r in rows]

    if search:
        search_l = search.lower()
        results = [r for r in results if search_l in (r.get("name", "") or "").lower()]

    return results


@router.post("/babies")
def create_baby(payload: dict[str, Any]) -> dict[str, Any]:
    """新增宝宝档案。"""
    store = _store()

    store_id = payload.get("store_id")
    name = payload.get("name", "").strip()
    gender = payload.get("gender", "unknown")
    birth_date = payload.get("birth_date", "")
    dietary = payload.get("dietary_restrictions", [])
    notes = payload.get("notes_free_text", "")

    if not name or not store_id or not birth_date:
        raise HTTPException(400, "宝宝姓名、门店 ID 和出生日期为必填项")
    if gender not in ("male", "female", "unknown"):
        raise HTTPException(400, "性别无效")

    return store.create_baby(
        store_id=store_id, name=name, gender=gender,
        birth_date=birth_date, dietary_restrictions=dietary, notes_free_text=notes,
    )


@router.get("/babies/{baby_id}")
def get_baby(baby_id: str) -> dict[str, Any]:
    """获取宝宝档案详情（含过敏记录和生长记录）。"""
    store = _store()
    baby = store.get_baby(baby_id)
    if baby is None:
        raise HTTPException(404, "宝宝档案不存在")

    baby["allergies"] = store.get_allergies_for_baby(baby_id)
    baby["growth_records"] = store.get_growth_records(baby_id)
    baby["notes"] = store.get_notes_for_baby(baby_id)

    # 计算月龄
    from datetime import date
    try:
        birth = date.fromisoformat(baby["birth_date"])
        today = date.today()
        months = (today.year - birth.year) * 12 + (today.month - birth.month)
        if today.day < birth.day:
            months -= 1
        baby["age_months"] = max(0, months)
    except Exception:
        baby["age_months"] = None

    return baby


# ============================
# 产品 CRUD
# ============================

@router.get("/products")
def list_products(category: Optional[str] = Query(None)) -> list[dict[str, Any]]:
    """列出产品，支持按分类过滤。"""
    store = _store()

    if category:
        rows = store.conn.execute(
            "SELECT p.*, s.name AS store_name FROM products p LEFT JOIN stores s ON p.store_id = s.id WHERE p.category = ? ORDER BY p.name",
            (category,),
        ).fetchall()
    else:
        rows = store.conn.execute(
            "SELECT p.*, s.name AS store_name FROM products p LEFT JOIN stores s ON p.store_id = s.id ORDER BY p.name",
        ).fetchall()
    return [_row_dict(r) for r in rows]


@router.post("/products")
def create_product(payload: dict[str, Any]) -> dict[str, Any]:
    """新增产品。"""
    store = _store()

    store_id = payload.get("store_id")
    name = payload.get("name", "").strip()
    category = payload.get("category", "")

    if not name or not store_id:
        raise HTTPException(400, "产品名称和门店 ID 为必填项")

    return store.create_product(
        store_id=store_id,
        name=name,
        category=category,
        description=payload.get("description", ""),
        suitable_age_min_months=payload.get("suitable_age_min_months"),
        suitable_age_max_months=payload.get("suitable_age_max_months"),
        ingredients=payload.get("ingredients", ""),
        price=payload.get("price"),
    )


@router.get("/products/categories")
def list_categories() -> list[dict[str, Any]]:
    """列出所有产品分类。"""
    store = _store()
    rows = store.conn.execute("SELECT * FROM product_categories ORDER BY name").fetchall()
    return [_row_dict(r) for r in rows]


# ============================
# 企业 / 门店
# ============================

@router.get("/enterprises")
def list_enterprises() -> list[dict[str, Any]]:
    """列出所有企业。"""
    store = _store()
    rows = store.conn.execute("SELECT * FROM enterprises ORDER BY created_at DESC").fetchall()
    return [_row_dict(r) for r in rows]


@router.get("/stores")
def list_stores(enterprise_id: Optional[str] = Query(None)) -> list[dict[str, Any]]:
    """列出门店，可选按企业过滤。"""
    store = _store()
    if enterprise_id:
        rows = store.conn.execute(
            "SELECT * FROM stores WHERE enterprise_id = ? ORDER BY name",
            (enterprise_id,),
        ).fetchall()
    else:
        rows = store.conn.execute("SELECT * FROM stores ORDER BY name").fetchall()
    return [_row_dict(r) for r in rows]


# ============================
# 配置读写
# ============================

@router.post("/settings")
def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """更新系统配置（运行时生效，不持久化到文件）。"""
    cfg = _cfg()
    changes = []

    if "data_scope" in payload:
        scope = payload["data_scope"]
        if scope in ("enterprise", "store", "store_strict"):
            cfg.data_scope = scope
            changes.append(f"data_scope={scope}")

    if "compression" in payload:
        comp = payload["compression"]
        if "max_context_tokens" in comp:
            cfg.compression.max_context_tokens = int(comp["max_context_tokens"])
            changes.append("max_context_tokens")
        if "compression_threshold" in comp:
            cfg.compression.compression_threshold = float(comp["compression_threshold"])
            changes.append("compression_threshold")
        if "keep_recent_turns" in comp:
            cfg.compression.keep_recent_turns = int(comp["keep_recent_turns"])
            changes.append("keep_recent_turns")

    if "seed_sync" in payload:
        ss = payload["seed_sync"]
        if "enabled" in ss:
            cfg.seed_sync.enabled = bool(ss["enabled"])
            changes.append("seed_sync_enabled")
        if "endpoint" in ss:
            cfg.seed_sync.endpoint = ss["endpoint"]
            changes.append("seed_sync_endpoint")

    if "model" in payload:
        m = payload["model"]
        if "provider" in m:
            cfg.model.provider = m["provider"]
            changes.append("model_provider")
        if "api_key" in m and m["api_key"]:
            cfg.model.api_key = m["api_key"]
            changes.append("model_api_key")
        if "model_name" in m:
            cfg.model.model_name = m["model_name"]
            changes.append("model_name")

    if "weixin" in payload:
        wx = payload["weixin"]
        if "bot_id" in wx:
            cfg.weixin.bot_id = wx["bot_id"]
        if "bot_token" in wx:
            cfg.weixin.bot_token = wx["bot_token"]
        changes.append("weixin")

    logger.info("配置已更新: %s", changes)
    return {"status": "ok", "changes": changes}


# ============================
# 微信 QR 码
# ============================

@router.post("/weixin/init-qr")
def init_weixin_qr() -> dict[str, Any]:
    """发起微信二维码绑定请求（占位实现）。"""
    cfg = _cfg()

    return {
        "status": "pending",
        "message": "二维码绑定功能将在 Gateway Platform 对接后实现",
        "bot_id": cfg.weixin.bot_id or "(未设置)",
        "enabled": cfg.weixin.enabled,
    }


# ============================
# 种子同步触发
# ============================

@router.post("/seed-sync/trigger")
def trigger_seed_sync() -> dict[str, Any]:
    """手动触发种子同步（占位）。"""
    cfg = _cfg()
    if not cfg.seed_sync.enabled:
        raise HTTPException(400, "种子同步未启用")

    return {
        "status": "triggered",
        "message": "种子同步将在后台执行",
        "endpoint": cfg.seed_sync.endpoint,
    }


# ============================
# 辅助函数
# ============================


def _row_dict(row: Any) -> dict[str, Any]:
    """将 sqlite3.Row 转换为 dict。"""
    if row is None:
        return {}
    return dict(row)


def _format_uptime(seconds: float) -> str:
    """将秒数格式为人类可读的运转时间字符串。"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d:
        parts.append(f"{d}天")
    if h:
        parts.append(f"{h}小时")
    if m:
        parts.append(f"{m}分钟")
    parts.append(f"{s}秒")
    return "".join(parts)


def _session_count(request=None) -> int:
    """返回当前活跃的会话数。"""
    # 需要从 app 的 _sessions 字典获取
    # 这里通过全局变量方式注入
    try:
        import babyagent.web.app as app_module
        return len(getattr(app_module, "_sessions", {}))
    except Exception:
        return 0
