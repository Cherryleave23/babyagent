"""BabyAgent Web 管理控制台

提供 FastAPI + Jinja2 的管理/运维界面（R15-R17），包括：
  - 仪表盘概览
  - 微信绑定配置
  - 模型配置管理
  - 员工/宝宝/产品 CRUD
  - 系统设置与状态监控

用法:
    from babyagent.web import create_app
    app = create_app(unified_store, config)
"""

from babyagent.web.app import create_app

__all__ = ["create_app"]
