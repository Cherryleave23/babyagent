# 🍼 BabyAgent — 母婴垂类 B2B 智能 Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Pre--release-orange.svg)]()

面向母婴企业门店员工的 AI 助手，充当 **"智能儿科医生 + 儿童营养师"** 角色。

通过 **微信 Clawbot（iLink 官方协议）** 与门店员工交互，为 0-6 岁宝宝量身定制膳食方案、推荐企业产品（零幻觉）、提供儿科健康咨询。

> 基于 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 内核魔改，保留 Gateway + Session + Model Provider + Skills 自进化。

---

## ✨ 核心特性

| 特性 | 说明 |
|---|---|
| 🧠 **不绑定单一模型** | 支持 DeepSeek / GLM(z.ai) / ChatGPT / Anthropic 等 33 种基座模型，运行时切换 |
| 🔒 **产品推荐零幻觉** | 产品名从企业库取值，LLM 仅解释推荐理由——硬管线杜绝编造 |
| 📋 **宝宝档案端侧保留** | 指令驱动（规则）写入，核心字段严禁 LLM 自由修改；过敏信息 append-only |
| 💬 **微信 Clawbot 官方通道** | 基于腾讯 iLink 协议，零封号风险，长轮询实时收发 |
| 👥 **多员工会话隔离** | 每个员工独立 Session，上下文互不可见；宝宝档案跨 Session 自动同步 |
| 🔄 **宝宝快速切换** | 支持隐式（消息中识别宝宝名）和显式（@切换宝宝 指令）双轨切换 |
| 🏢 **一户一实例** | 每个企业独立部署，多门店共享产品库，数据隔离策略 WebUI 可调配 |
| 📚 **三层知识库** | 总部种子库（只读）+ 企业拓展库 + 运行时数据 → 统一向量检索空间 |
| 🎛️ **Web 管理控制台** | 微信绑定、API Key 配置、员工管理、产品录入——一站式运维 |
| 🔄 **自进化 Skills** | 保留 Hermes Skills 机制 + Learning Loop，Agent 从实践中持续成长 |

---

## 🏗️ 架构概览

```
员工微信消息 (from_user_id + context_token)
  │
  ▼
┌─ weixin.py (iLink 协议) ───────────────────────────┐
│  基于 Hermes WeixinAdapter，官方 Clawbot 通道        │
└────────────────────┬───────────────────────────────┘
                     ▼
┌─ Gateway Session Store ─────────────────────────────┐
│  Session Key: "babyagent:weixin:dm:{employee_wxid}" │
│  天然隔离，每个员工独立上下文                         │
└────────────────────┬───────────────────────────────┘
                     ▼
┌─ 意图分类 (LLM structured output) ──────────────────┐
│  product_recommend | diet_advice | health_consult    │
│  out_of_scope → 拒答模板 (I2: 不调用 LLM)            │
└────────────────────┬───────────────────────────────┘
         ┌───────────┴───────────┐
         ▼                       ▼
  产品推荐硬管线            RAG 健康/膳食咨询
  (四步零幻觉)              (端侧统一库检索)
         │                       │
         ▼                       ▼
  LLM 解释推荐理由          LLM 生成自然语言回复
  (产品名从 DB 取)          
         └───────────┬───────────┘
                     ▼
           上下文压缩 (双输出)
           ├─ Session 私有摘要
           └─ 宝宝档案变更建议 → 跨 Session 同步
```

---

## 🚀 快速开始

### 前置要求

- Python 3.11+
- Windows / WSL2 / Linux
- 微信 Clawbot 插件（微信 → 设置 → 插件 → ClawBot）

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/babyagent.git
cd babyagent

# 创建虚拟环境并安装
pip install -e .
```

### 2. 配置

```bash
# 从模板创建配置文件
copy src\babyagent\config\config.yaml.template config.yaml

# 编辑配置：填入模型 API Key、企业信息
notepad config.yaml
```

### 3. 初始化部署

```bash
# 启动 Web 管理控制台 + Gateway
babyagent serve

# 或分步启动
babyagent gateway --verbose   # 仅 Gateway
babyagent web                  # 仅 Web 控制台
```

### 4. 绑定微信

打开 `http://127.0.0.1:8800` → 登录 → 微信绑定 → 扫码完成。

### 5. 开始使用

门店员工直接通过微信向绑定的 Bot 发送消息即可。

**指令示例**：

```
@Agent 建档 小明 男 2025-03-15
@Agent 小明 过敏史添加 花生
@宝宝 小红
小明最近拉肚子，有没有推荐的益生菌？
```

---

## 📁 项目结构

```
babyagent/
├── src/babyagent/
│   ├── main.py                # CLI 入口 (gateway/web/serve/setup)
│   ├── config/                # 配置管理
│   │   ├── loader.py          # 配置加载器
│   │   └── config.yaml.template
│   ├── core/
│   │   ├── agent_original/    # Hermes 原内核（保留）
│   │   ├── plugins/           # 33 model-providers + Skills
│   │   ├── providers/         # ProviderProfile 抽象层
│   │   ├── tools/             # Hermes 工具集
│   │   ├── db/
│   │   │   ├── schema.py      # SQLite + ChromaDB 三层向量库
│   │   │   └── unified_store.py  # 统一检索 + CRUD
│   │   ├── baby/
│   │   │   ├── profile.py     # 宝宝档案模型 + 管理器
│   │   │   └── compression.py # 上下文压缩双输出
│   │   └── pipeline/
│   │       ├── intent.py      # 意图分类 + 路由分发
│   │       ├── rejection.py   # 拒答处理器
│   │       ├── product_recommend.py  # 产品推荐硬管线
│   │       └── health_diet.py # RAG 健康/膳食咨询
│   ├── gateway/
│   │   ├── session_store.py   # 员工隔离 Session
│   │   ├── baby_switch.py     # 宝宝快速切换
│   │   ├── command_parser.py  # @Agent 指令解析
│   │   └── orchestrator.py    # 核心消息流水线
│   └── web/
│       ├── app.py             # FastAPI 管理控制台
│       ├── api.py             # REST API
│       └── templates/         # 11 个管理页面
├── docs/
│   └── adr/                   # 架构决策记录
├── SPEC.md                    # 需求规范 (43 条 + 9 不变式 + 18 验收)
├── CONTEXT.md                 # 术语表
├── pyproject.toml
└── README.md
```

---

## 🛡️ 核心不变式

| 编号 | 不变式 |
|---|---|
| I1 | 产品推荐中产品名必须来自企业产品库（零幻觉） |
| I2 | `out_of_scope` 不调用 LLM 生成回复 |
| I3 | 备注不参与产品推荐决策 |
| I4 | 每个会话最多一个活跃宝宝指针 |
| I5 | 总部种子库对端侧只读 |
| I6 | 种子库同步失败不中断服务 |
| I7 | Session 私有摘要不跨 Session 传播 |
| I8 | 过敏信息 append-only，仅人工可删除 |
| I9 | 产品检索空结果不降级为 LLM 推荐 |

---

## 🔧 技术栈

| 层 | 技术 |
|---|---|
| **基座** | Hermes Agent (Python) 内核魔改 |
| **模型支持** | 33 个 Provider，通过 OpenAI SDK 兼容协议统一调用 |
| **微信通道** | 腾讯 Clawbot / iLink 协议 (HTTP/JSON 长轮询) |
| **数据库** | SQLite (WAL 模式) + ChromaDB (向量检索) |
| **向量嵌入** | sentence-transformers (BAAI/bge-small-zh-v1.5) |
| **Web 控制台** | FastAPI + Jinja2 + uvicorn |

---

## 📋 路线图

- [x] 规范文档 (SPEC + CONTEXT + ADR)
- [x] Hermes 内核搭架 + 裁剪
- [x] 端侧统一数据库 (SQLite + ChromaDB 三层)
- [x] 宝宝档案管理 + 上下文压缩双输出
- [x] 智能管线 (意图分类 + 产品推荐 + RAG)
- [x] 微信 Clawbot 接入 + 员工隔离 + 宝宝切换
- [x] Web 管理控制台
- [ ] 总部知识库同步服务端
- [ ] 产品库批量导入 (Excel/CSV)
- [ ] 对话审计与统计报表
- [ ] Docker 一键部署

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可

MIT License

---

**Built with ❤️ for 母婴行业**
