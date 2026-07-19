"""BabyAgent 配置加载器"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    provider: str = "deepseek"
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class WeixinConfig:
    enabled: bool = True
    bot_id: str = ""
    bot_token: str = ""
    long_poll_timeout_ms: int = 35000
    dedup_ttl_seconds: int = 300


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8800
    username: str = "admin"
    password: str = ""


@dataclass
class DatabaseConfig:
    path: str = "./data/babyagent.db"
    vector_path: str = "./data/vectors/"
    seed_version: int = 1


@dataclass
class SeedSyncConfig:
    enabled: bool = True
    endpoint: str = ""
    cron: str = "0 3 * * *"
    delta_threshold_pct: int = 30


@dataclass
class CompressionConfig:
    max_context_tokens: int = 64000
    compression_threshold: float = 0.8
    keep_recent_turns: int = 6


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./data/babyagent.log"


@dataclass
class EnterpriseConfig:
    name: str = "未命名企业"
    store_id: str = "default"


@dataclass
class AppConfig:
    enterprise: EnterpriseConfig = field(default_factory=EnterpriseConfig)
    data_scope: str = "enterprise"  # enterprise | store | store_strict
    model: ModelConfig = field(default_factory=ModelConfig)
    aux_model: ModelConfig = field(default_factory=lambda: ModelConfig(provider="deepseek"))
    weixin: WeixinConfig = field(default_factory=WeixinConfig)
    web: WebConfig = field(default_factory=WebConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    seed_sync: SeedSyncConfig = field(default_factory=SeedSyncConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _nested_get(d: dict, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k, {})
    return d if d != {} else None


def load_config(path: Optional[str] = None) -> AppConfig:
    """加载配置：优先指定文件 > 环境变量 BABYAGENT_CONFIG > 默认 search paths"""
    if path:
        return _load_from_file(path)

    env_path = os.environ.get("BABYAGENT_CONFIG")
    if env_path and os.path.isfile(env_path):
        return _load_from_file(env_path)

    # 默认搜索路径
    search_paths = [
        Path("config.yaml"),
        Path("config/config.yaml"),
        Path.home() / ".babyagent" / "config.yaml",
    ]
    for p in search_paths:
        if p.is_file():
            return _load_from_file(str(p))

    # 无配置文件，返回默认
    return AppConfig()


def _load_from_file(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def _get(*keys):
        return _nested_get(raw, *keys)

    cfg = AppConfig()

    # enterprise
    ent = _get("enterprise") or {}
    cfg.enterprise = EnterpriseConfig(
        name=ent.get("name", cfg.enterprise.name),
        store_id=str(ent.get("store_id", cfg.enterprise.store_id)),
    )
    cfg.data_scope = raw.get("data_scope", cfg.data_scope)

    # model
    m = _get("model") or {}
    cfg.model = ModelConfig(
        provider=m.get("provider", cfg.model.provider),
        model_name=m.get("model_name", cfg.model.model_name),
        api_key=m.get("api_key", os.environ.get("BABYAGENT_API_KEY", cfg.model.api_key)),
        base_url=m.get("base_url", cfg.model.base_url),
        max_tokens=m.get("max_tokens", cfg.model.max_tokens),
        temperature=m.get("temperature", cfg.model.temperature),
    )

    # aux model
    am = _get("aux_model") or {}
    cfg.aux_model = ModelConfig(
        provider=am.get("provider", cfg.aux_model.provider),
        model_name=am.get("model_name", cfg.aux_model.model_name),
        api_key=am.get("api_key", cfg.aux_model.api_key),
        base_url=am.get("base_url", cfg.aux_model.base_url),
    )

    # weixin
    wx = _get("weixin") or {}
    cfg.weixin = WeixinConfig(
        enabled=wx.get("enabled", cfg.weixin.enabled),
        bot_id=wx.get("bot_id", cfg.weixin.bot_id),
        bot_token=wx.get("bot_token", cfg.weixin.bot_token),
        long_poll_timeout_ms=wx.get("long_poll_timeout_ms", cfg.weixin.long_poll_timeout_ms),
        dedup_ttl_seconds=wx.get("dedup_ttl_seconds", cfg.weixin.dedup_ttl_seconds),
    )

    # web
    w = _get("web") or {}
    auth = w.get("auth", {})
    cfg.web = WebConfig(
        host=w.get("host", cfg.web.host),
        port=w.get("port", cfg.web.port),
        username=auth.get("username", cfg.web.username),
        password=auth.get("password", cfg.web.password),
    )

    # database
    db = _get("database") or {}
    cfg.database = DatabaseConfig(
        path=db.get("path", cfg.database.path),
        vector_path=db.get("vector_path", cfg.database.vector_path),
        seed_version=db.get("seed_version", cfg.database.seed_version),
    )

    # seed sync
    ss = _get("seed_sync") or {}
    cfg.seed_sync = SeedSyncConfig(
        enabled=ss.get("enabled", cfg.seed_sync.enabled),
        endpoint=ss.get("endpoint", cfg.seed_sync.endpoint),
        cron=ss.get("cron", cfg.seed_sync.cron),
        delta_threshold_pct=ss.get("delta_threshold_pct", cfg.seed_sync.delta_threshold_pct),
    )

    # compression
    comp = _get("compression") or {}
    cfg.compression = CompressionConfig(
        max_context_tokens=comp.get("max_context_tokens", cfg.compression.max_context_tokens),
        compression_threshold=comp.get("compression_threshold", cfg.compression.compression_threshold),
        keep_recent_turns=comp.get("keep_recent_turns", cfg.compression.keep_recent_turns),
    )

    # logging
    log = _get("logging") or {}
    cfg.logging = LoggingConfig(
        level=log.get("level", cfg.logging.level),
        file=log.get("file", cfg.logging.file),
    )

    return cfg
