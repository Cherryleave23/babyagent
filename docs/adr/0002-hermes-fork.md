# ADR-0002：基于 Hermes 内核做减法进行魔改

**日期**：2026-07-19  
**状态**：已接受

## 背景

母婴垂类 B2B Agent 需要一个成熟的开源 Agent 框架作为基础。候选包括 OpenClaw（TypeScript）和 Hermes（Python）。核心需求：微信官方 Clawbot 接入、多模型支持（DeepSeek/GLM/ChatGPT）、多员工会话隔离、上下文压缩、可魔改。

## 决策

**选择 Hermes（Python）**，做减法魔改。基于源码阅读（6600+ 文件 / 145MB）后确定保留/砍除/改造清单：

### 保留
| 模块 | 理由 |
|---|---|
| Gateway + `weixin.py` | 微信 iLink 协议已完整实现，`from_user_id` 天然隔离 |
| SessionStore（SQLite） | 会话管理 + `build_session_key()` 天然支持 `{platform}:{dm}:{user_id}` 隔离 |
| 全部 33 个 model-provider 插件 | 用户要求不做精简 |
| Context Compressor + Compression | 长对话摘要/压缩 |
| Skills 体系 + Learning Loop | 用户要求保留自进化能力 |
| Agent Conversation Loop | 主循环，需改造注入母婴流水线 |

### 砍除
Desktop App (Electron)、TUI、WebChat、Voice Wake、Talk Mode、Browser Control、Canvas/A2UI、微信以外全部消息通道、图片/视频/语音/TTS 生成、Kanban、Spotify、Google Meet 等

### 新增
Web 管理控制台（运维配置：微信绑定、API Key、员工管理、宝宝档案）

## 后果

- **优点**：Hermes 的 Gateway + Session + Context Compression 三条命脉对齐需求；Python 生态适配母婴 RAG/向量数据库/营养计算
- **代价**：Hermes 是通用个人助手，砍除后剩余 ~200 文件/10MB，需重写 agent loop 注入母婴业务逻辑
- **风险**：Hermes 版本更新快，需维护 fork 的同步策略
