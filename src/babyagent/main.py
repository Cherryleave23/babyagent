"""BabyAgent 主入口

启动 Gateway（微信 Clawbot 接入）和 Web 管理控制台。
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BabyAgent — 母婴垂类 B2B 智能 Agent",
    )
    sub = parser.add_subparsers(dest="command")

    # gateway 模式
    gw = sub.add_parser("gateway", help="启动 Gateway（微信 Clawbot + Agent 核心）")
    gw.add_argument("--port", type=int, default=18789, help="Gateway WebSocket 端口")
    gw.add_argument("--config", type=str, default=None, help="配置文件路径")
    gw.add_argument("--verbose", action="store_true", help="详细日志")

    # web 模式
    web = sub.add_parser("web", help="启动 Web 管理控制台（运维配置）")
    web.add_argument("--port", type=int, default=8800, help="Web 服务端口")
    web.add_argument("--host", type=str, default="127.0.0.1", help="绑定地址")

    # 全量模式
    all_ = sub.add_parser("serve", help="同时启动 Gateway + Web 控制台")
    all_.add_argument("--gw-port", type=int, default=18789, help="Gateway 端口")
    all_.add_argument("--web-port", type=int, default=8800, help="Web 端口")
    all_.add_argument("--config", type=str, default=None, help="配置文件路径")

    # setup 模式
    setup = sub.add_parser("setup", help="初始化部署：扫码绑定微信 + 配置模型 API")
    setup.add_argument("--config", type=str, default=None, help="配置文件路径")

    return parser.parse_args()


def run_gateway(config_path: str = None, port: int = 18789, verbose: bool = False):
    """启动 Gateway 服务"""
    from babyagent.gateway import run as gateway_run
    logger.info("启动 BabyAgent Gateway (port=%d)...", port)
    # TODO: 接入 gateway/run.py
    logger.info("Gateway 运行中（框架就绪，待 Iter4 接入微信 Clawbot）")


def run_web(host: str = "127.0.0.1", port: int = 8800):
    """启动 Web 管理控制台"""
    import uvicorn
    logger.info("启动 Web 管理控制台 http://%s:%d ...", host, port)
    # TODO: 接入 FastAPI app
    logger.info("Web 控制台就绪（框架就绪，待 Iter5 实现管理界面）")


def run_all(gw_port: int = 18789, web_port: int = 8800, config_path: str = None):
    """同时启动 Gateway + Web"""
    import asyncio
    logger.info("启动 BabyAgent 全量模式...")
    # TODO: 并行启动
    logger.info("全量模式框架就绪")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "gateway":
        run_gateway(config_path=args.config, port=args.port, verbose=args.verbose)
    elif args.command == "web":
        run_web(host=args.host, port=args.port)
    elif args.command == "serve":
        run_all(gw_port=args.gw_port, web_port=args.web_port, config_path=args.config)
    elif args.command == "setup":
        logger.info("初始化向导...")
    else:
        logger.error("未知命令: %s", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
