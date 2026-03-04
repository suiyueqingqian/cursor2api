# Cursor2API v2

将 Cursor 文档页免费 AI 对话接口代理转换为 **Anthropic Messages API** 和 **OpenAI Chat Completions API**，可直接对接 **Claude Code**、**ChatBox**、**LobeChat** 等各类客户端。

## 原理

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Claude Code  │────▶│              │────▶│              │
│ (Anthropic)  │     │              │     │              │
│              │◀────│              │◀────│              │
├─────────────┤     │  cursor2api  │     │  Cursor API  │
│ ChatBox 等   │────▶│  (代理+转换)  │     │  /api/chat   │
│ (OpenAI)     │     │              │     │              │
│              │◀────│              │◀────│              │
└─────────────┘     └──────────────┘     └──────────────┘
```

1. Claude Code 发送标准 Anthropic Messages API 请求（带工具定义）
2. cursor2api 将工具定义**注入为系统提示词**（XML 格式，Claude 模型原生理解）
3. 将消息转换为 Cursor `/api/chat` 格式，带 Chrome TLS 指纹模拟
4. Cursor 背后的 Claude Sonnet 4.6 按照提示词输出工具调用
5. cursor2api 解析 XML 工具调用 → 转换为 Anthropic `tool_use` 格式返回
6. Claude Code 执行工具 → 发送 `tool_result` → 循环

## 核心特性

- **Anthropic Messages API 完整兼容** - `/v1/messages` 流式/非流式
- **OpenAI Chat Completions API 兼容** - `/v1/chat/completions` 流式/非流式 + 工具调用
- **提示词注入工具能力** - 让 Claude Code 的 Bash、Read、Write 等工具全部可用
- **Node.js/TypeScript** - 无需外部进程生成 x-is-human token
- **Chrome TLS 指纹** - 模拟真实浏览器请求头
- **SSE 流式传输** - 实时响应

## 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 获取必要文件

```bash
# 下载浏览器环境模拟脚本
curl -o jscode/env.js https://raw.githubusercontent.com/jhhgiyv/cursorweb2api/master/jscode/env.js
curl -o jscode/main.js https://raw.githubusercontent.com/jhhgiyv/cursorweb2api/master/jscode/main.js
```

### 3. 配置

编辑 `config.yaml`：
- `script_url` - 从 Cursor 文档页 DevTools 网络面板获取 `c.js` 请求 URL
- `fingerprint` - 浏览器指纹信息

### 4. 启动

```bash
npm run dev
```

### 5. 配合 Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:3010
claude
```

### 6. 配合 OpenAI 兼容客户端（ChatBox、LobeChat 等）

在客户端设置中填入：
- **API Base URL**: `http://localhost:3010/v1`
- **API Key**: 任意值（如 `sk-xxx`，不做校验）
- **Model**: 任意值（实际使用 config.yaml 中配置的模型）

## 项目结构

```
cursor2api/
├── src/
│   ├── index.ts            # 入口 + Express 服务
│   ├── config.ts           # 配置管理
│   ├── types.ts            # Anthropic/Cursor 类型定义
│   ├── openai-types.ts     # OpenAI 类型定义
│   ├── cursor-client.ts    # Cursor API 客户端 + Token 生成
│   ├── converter.ts        # 协议转换 + 工具提示词注入
│   ├── handler.ts          # Anthropic API 处理器
│   └── openai-handler.ts   # OpenAI API 处理器
├── jscode/                 # x-is-human token 生成脚本
├── config.yaml             # 配置文件
├── package.json
└── tsconfig.json
```

## 技术架构

### 提示词注入工具能力

Claude Code 发送工具定义 → 我们将其转换为 XML 格式系统提示：

```xml
<antml_tool_call>
<tool_name>Bash</tool_name>
<tool_input>
{"command": "ls -la"}
</tool_input>
</antml_tool_call>
```

AI 按此格式输出 → 我们解析并转换为标准的 Anthropic `tool_use` content block。

### x-is-human Token

Cursor 使用 `x-is-human` 请求头进行人机验证。Token 由前端 JS 生成，有效期 25 分钟。
在 Node.js 中直接执行验证脚本，无需外部进程。

## 免责声明 / Disclaimer

**本项目仅供学习、研究和接口调试目的使用。**

1. 本项目并非 Cursor 官方项目，与 Cursor 及其母公司 Anysphere 没有任何关联。
2. 本项目包含针对特定 API 协议的转换代码。在使用本项目前，请确保您已经仔细阅读并同意 Cursor 的服务条款（Terms of Service）。使用本项目可能引发账号封禁或其他限制。
3. 请合理使用，勿将本项目用于任何商业牟利行为、DDoS 攻击或大规模高频并发滥用等非法违规活动。
4. **作者及贡献者对任何人因使用本代码导致的任何损失、账号封禁或法律纠纷不承担任何直接或间接的责任。一切后果由使用者自行承担。**

## License

[MIT](LICENSE)
