"""BabyAgent Web 管理控制台 — FastAPI 主应用

使用工厂模式 create_app(unified_store, config) 注入依赖，
基于 Jinja2 模板渲染管理后台页面。
"""

from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response

# API 路由
from babyagent.web.api import router as api_router, inject as api_inject

logger = logging.getLogger(__name__)

# ---------- 全局会话存储 ----------

_sessions: dict[str, dict[str, Any]] = {}
SESSION_TIMEOUT = 3600  # 1 小时超时

# ---------- 模板路径 ----------

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------- Jinja2 ----------

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------- 全局注入 ----------

_unified_store: Any = None
_config: Any = None


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
# 会话管理
# ============================


def _get_session(request: Request) -> Optional[dict[str, Any]]:
    """从 Cookie 获取会话，过期则清除。"""
    sid = request.cookies.get("ba_session")
    if not sid or sid not in _sessions:
        return None
    sess = _sessions[sid]
    if time.time() - sess["created"] > SESSION_TIMEOUT:
        _sessions.pop(sid, None)
        return None
    return sess


def _require_auth(request: Request) -> dict[str, Any]:
    """要求登录，否则重定向。"""
    sess = _get_session(request)
    if sess is None:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return sess


# ============================
# 工厂函数
# ============================


def create_app(unified_store: Any, config: Any) -> FastAPI:
    """创建 FastAPI 应用实例，注入 UnifiedStore 与 AppConfig。

    Args:
        unified_store: UnifiedStore 实例。
        config: AppConfig 实例。

    Returns:
        已配置的 FastAPI 应用。
    """
    global _unified_store, _config
    _unified_store = unified_store
    _config = config

    # 注入到 api 模块
    api_inject(unified_store, config)

    app = FastAPI(title="BabyAgent 管理控制台", version="2.0")

    # 注册 API 路由
    app.include_router(api_router)

    # ===========================================
    # 路由定义
    # ===========================================

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """首页重定向到登录页。"""
        return RedirectResponse(url="/login", status_code=302)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        """登录页面 (GET)。"""
        # 已登录则跳转到仪表盘
        if _get_session(request):
            return RedirectResponse(url="/dashboard", status_code=302)

        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": None,
        })

    @app.post("/login", response_class=HTMLResponse)
    async def login_action(
        request: Request,
        username: str = Form(""),
        password: str = Form(""),
    ):
        """登录操作 (POST)。"""
        cfg = _cfg()
        web_cfg = cfg.web

        if username == web_cfg.username and password == web_cfg.password:
            sid = secrets.token_hex(32)
            _sessions[sid] = {"username": username, "created": time.time()}
            logger.info("用户登录: %s", username)
            resp = RedirectResponse(url="/dashboard", status_code=302)
            resp.set_cookie(key="ba_session", value=sid, httponly=True, max_age=SESSION_TIMEOUT)
            return resp

        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "用户名或密码错误",
        })

    @app.get("/logout")
    async def logout(request: Request):
        """退出登录。"""
        sid = request.cookies.get("ba_session")
        if sid:
            _sessions.pop(sid, None)
        resp = RedirectResponse(url="/login", status_code=302)
        resp.delete_cookie("ba_session")
        return resp

    # ---------- 仪表盘 ----------

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """管理仪表盘首页。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess
        return _render_dashboard(request)

    # ---------- 微信绑定 ----------

    @app.get("/weixin", response_class=HTMLResponse)
    async def weixin_page(request: Request):
        """微信 Clawbot 配置页面。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        cfg = _cfg()
        wx = cfg.weixin

        bot_id_display = wx.bot_id or "(未设置)"
        token_display = _mask_token(wx.bot_token) if wx.bot_token else "(未设置)"

        return templates.TemplateResponse("weixin.html", {
            "request": request,
            "current_page": "weixin",
            "bot_id": bot_id_display,
            "bot_id_raw": wx.bot_id,
            "bot_token": token_display,
            "bot_token_raw": wx.bot_token,
            "enabled": wx.enabled,
            "qr_status": "pending",
            "qr_message": "",
        })

    # ---------- 模型配置 ----------

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        """模型配置页面。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        cfg = _cfg()
        m = cfg.model
        am = cfg.aux_model

        providers = ["deepseek", "zai", "openai-codex", "anthropic"]

        return templates.TemplateResponse("models.html", {
            "request": request,
            "current_page": "models",
            "providers": providers,
            "model": m,
            "aux_model": am,
            "test_result": None,
        })

    @app.post("/models", response_class=HTMLResponse)
    async def models_save(
        request: Request,
        provider: str = Form("deepseek"),
        model_name: str = Form(""),
        api_key: str = Form(""),
        aux_provider: str = Form("deepseek"),
        aux_model_name: str = Form(""),
    ):
        """保存模型配置。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        cfg = _cfg()
        cfg.model.provider = provider
        cfg.model.model_name = model_name
        if api_key:
            cfg.model.api_key = api_key
        cfg.aux_model.provider = aux_provider
        cfg.aux_model.model_name = aux_model_name

        providers = ["deepseek", "zai", "openai-codex", "anthropic"]

        return templates.TemplateResponse("models.html", {
            "request": request,
            "current_page": "models",
            "providers": providers,
            "model": cfg.model,
            "aux_model": cfg.aux_model,
            "test_result": "配置已保存",
        })

    # ---------- 员工管理 ----------

    @app.get("/employees", response_class=HTMLResponse)
    async def employees_page(request: Request):
        """员工列表页面。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        try:
            store = _store()
            rows = store.conn.execute(
                "SELECT e.*, s.name AS store_name FROM employees e LEFT JOIN stores s ON e.store_id = s.id ORDER BY e.created_at DESC"
            ).fetchall()
            employees_list = [dict(r) for r in rows]

            stores = store.conn.execute("SELECT * FROM stores ORDER BY name").fetchall()
            stores_list = [dict(r) for r in stores]
        except Exception as e:
            logger.warning("数据库查询失败: %s", e)
            employees_list = []
            stores_list = []

        return templates.TemplateResponse("employees.html", {
            "request": request,
            "current_page": "employees",
            "employees": employees_list,
            "stores": stores_list,
        })

    # ---------- 宝宝档案 ----------

    @app.get("/babies", response_class=HTMLResponse)
    async def babies_page(
        request: Request,
        search: str = "",
        store_id: str = "",
    ):
        """宝宝档案列表。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        try:
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

            babies_list = [dict(r) for r in rows]
            if search:
                search_l = search.lower()
                babies_list = [b for b in babies_list if search_l in (b.get("name", "") or "").lower()]

            stores_rows = store.conn.execute("SELECT * FROM stores ORDER BY name").fetchall()
            stores_list = [dict(r) for r in stores_rows]
        except Exception as e:
            logger.warning("数据库查询失败: %s", e)
            babies_list = []
            stores_list = []

        return templates.TemplateResponse("babies.html", {
            "request": request,
            "current_page": "babies",
            "babies": babies_list,
            "stores": stores_list,
            "search": search,
            "store_id": store_id,
        })

    @app.get("/babies/{baby_id}", response_class=HTMLResponse)
    async def baby_detail(request: Request, baby_id: str):
        """宝宝档案详情页。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        try:
            store = _store()
            baby = store.get_baby(baby_id)
            if baby is None:
                raise HTTPException(404, "宝宝档案不存在")

            baby["allergies"] = store.get_allergies_for_baby(baby_id)
            baby["growth_records"] = store.get_growth_records(baby_id)
            baby["notes"] = store.get_notes_for_baby(baby_id)

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

            # 获取门店名称
            store_row = store.conn.execute(
                "SELECT name FROM stores WHERE id = ?", (baby.get("store_id"),)
            ).fetchone()
            store_name = store_row["name"] if store_row else "未知门店"
        except Exception as e:
            logger.exception("获取宝宝详情失败")
            raise HTTPException(500, str(e))

        return templates.TemplateResponse("baby_detail.html", {
            "request": request,
            "current_page": "babies",
            "baby": baby,
            "store_name": store_name,
        })

    # ---------- 产品管理 ----------

    @app.get("/products", response_class=HTMLResponse)
    async def products_page(
        request: Request,
        category: str = "",
    ):
        """产品管理列表。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        try:
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
            products_list = [dict(r) for r in rows]

            cats = store.conn.execute("SELECT DISTINCT category FROM products WHERE category != '' ORDER BY category").fetchall()
            categories_list = [r["category"] for r in cats]

            stores_rows = store.conn.execute("SELECT * FROM stores ORDER BY name").fetchall()
            stores_list = [dict(r) for r in stores_rows]
        except Exception as e:
            logger.warning("数据库查询失败: %s", e)
            products_list = []
            categories_list = []
            stores_list = []

        return templates.TemplateResponse("products.html", {
            "request": request,
            "current_page": "products",
            "products": products_list,
            "categories": categories_list,
            "stores": stores_list,
            "selected_category": category,
        })

    # ---------- 系统设置 ----------

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        """系统设置页面。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        cfg = _cfg()

        return templates.TemplateResponse("settings.html", {
            "request": request,
            "current_page": "settings",
            "data_scope": cfg.data_scope,
            "compression": cfg.compression,
            "seed_sync": cfg.seed_sync,
            "enterprise": cfg.enterprise,
            "message": None,
        })

    @app.post("/settings", response_class=HTMLResponse)
    async def settings_save(
        request: Request,
        data_scope: str = Form("enterprise"),
        max_context_tokens: int = Form(64000),
        compression_threshold: float = Form(0.8),
        keep_recent_turns: int = Form(6),
        seed_sync_enabled: str = Form("off"),
        seed_sync_endpoint: str = Form(""),
    ):
        """保存系统设置。"""
        sess = _require_auth(request)
        if isinstance(sess, HTTPException):
            raise sess

        cfg = _cfg()
        cfg.data_scope = data_scope
        cfg.compression.max_context_tokens = max_context_tokens
        cfg.compression.compression_threshold = compression_threshold
        cfg.compression.keep_recent_turns = keep_recent_turns
        cfg.seed_sync.enabled = seed_sync_enabled == "on"
        cfg.seed_sync.endpoint = seed_sync_endpoint

        logger.info("系统设置已保存: data_scope=%s, seed_sync=%s", data_scope, seed_sync_enabled)

        return templates.TemplateResponse("settings.html", {
            "request": request,
            "current_page": "settings",
            "data_scope": cfg.data_scope,
            "compression": cfg.compression,
            "seed_sync": cfg.seed_sync,
            "enterprise": cfg.enterprise,
            "message": "设置已保存",
        })

    return app


# ============================
# 辅助函数
# ============================


def _render_dashboard(request: Request) -> Response:
    """渲染仪表盘页面。"""
    try:
        store = _store()
        cfg = _cfg()

        employees = store.conn.execute("SELECT COUNT(*) as cnt FROM employees").fetchone()
        babies = store.list_babies()
        products = store.conn.execute("SELECT COUNT(*) as cnt FROM products").fetchone()
        enterprises = store.conn.execute("SELECT COUNT(*) as cnt FROM enterprises").fetchone()
        seed_version = store.get_seed_version()

        stats = {
            "enterprise_name": cfg.enterprise.name,
            "data_scope": cfg.data_scope,
            "data_scope_label": _scope_label(cfg.data_scope),
            "employees_count": employees["cnt"] if employees else 0,
            "babies_count": len(babies) if babies else 0,
            "products_count": products["cnt"] if products else 0,
            "enterprises_count": enterprises["cnt"] if enterprises else 0,
            "seed_version": seed_version,
            "model_provider": cfg.model.provider,
            "weixin_enabled": cfg.weixin.enabled,
            "seed_sync_enabled": cfg.seed_sync.enabled,
        }
    except Exception as e:
        logger.exception("仪表盘数据加载失败")
        stats = {
            "enterprise_name": "加载失败",
            "data_scope": "unknown",
            "data_scope_label": "未知",
            "employees_count": 0,
            "babies_count": 0,
            "products_count": 0,
            "enterprises_count": 0,
            "seed_version": 0,
            "model_provider": "未知",
            "weixin_enabled": False,
            "seed_sync_enabled": False,
        }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "current_page": "dashboard",
        "stats": stats,
    })


def _scope_label(scope: str) -> str:
    """数据隔离策略的中文标签。"""
    return {
        "enterprise": "全企业共享",
        "store": "产品共享+客户隔离",
        "store_strict": "完全隔离",
    }.get(scope, scope)


def _mask_token(token: str) -> str:
    """脱敏显示 Token（前 4 位 + **** + 后 4 位）。"""
    if len(token) <= 8:
        return token[:2] + "****" if len(token) >= 2 else "****"
    return token[:4] + "****" + token[-4:]
