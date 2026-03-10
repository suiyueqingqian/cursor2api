/**
 * Cursor2API v2 - 入口
 *
 * 将 Cursor 文档页免费 AI 接口代理为 Anthropic Messages API
 * 通过提示词注入让 Claude Code 拥有完整工具调用能力
 */

import 'dotenv/config';
import { createRequire } from 'module';
import express from 'express';
import { getConfig } from './config.js';
import { handleMessages, listModels, countTokens } from './handler.js';
import { handleOpenAIChatCompletions } from './openai-handler.js';

// 从 package.json 读取版本号，统一来源，避免多处硬编码
const require = createRequire(import.meta.url);
const { version: VERSION } = require('../package.json') as { version: string };


const app = express();
const config = getConfig();

// 解析 JSON body（增大限制以支持 base64 图片，单张图片可达 10MB+）
app.use(express.json({ limit: '50mb' }));

// CORS
app.use((_req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', '*');
    if (_req.method === 'OPTIONS') {
        res.sendStatus(200);
        return;
    }
    next();
});

// ==================== 路由 ====================

// Anthropic Messages API
app.post('/v1/messages', handleMessages);
app.post('/messages', handleMessages);

// OpenAI Chat Completions API（兼容）
app.post('/v1/chat/completions', handleOpenAIChatCompletions);
app.post('/chat/completions', handleOpenAIChatCompletions);

// Token 计数
app.post('/v1/messages/count_tokens', countTokens);
app.post('/messages/count_tokens', countTokens);

// OpenAI 兼容模型列表
app.get('/v1/models', listModels);

// 健康检查
app.get('/health', (_req, res) => {
    res.json({ status: 'ok', version: VERSION });
});

// 根路径
app.get('/', (_req, res) => {
    res.json({
        name: 'cursor2api',
        version: VERSION,
        description: 'Cursor Docs AI → Anthropic & OpenAI API Proxy',
        endpoints: {
            anthropic_messages: 'POST /v1/messages',
            openai_chat: 'POST /v1/chat/completions',
            models: 'GET /v1/models',
            health: 'GET /health',
        },
        usage: {
            claude_code: 'export ANTHROPIC_BASE_URL=http://localhost:' + config.port,
            openai_compatible: 'OPENAI_BASE_URL=http://localhost:' + config.port + '/v1',
        },
    });
});

// ==================== 启动 ====================

app.listen(config.port, () => {
    console.log('');
    console.log('  ╔══════════════════════════════════════╗');
    console.log(`  ║        Cursor2API v${VERSION.padEnd(21)}║`);
    console.log('  ╠══════════════════════════════════════╣');
    console.log(`  ║  Server:  http://localhost:${config.port}      ║`);
    console.log('  ║  Model:   ' + config.cursorModel.padEnd(26) + '║');
    console.log('  ╠══════════════════════════════════════╣');
    console.log('  ║  API Endpoints:                      ║');
    console.log('  ║  • Anthropic: /v1/messages            ║');
    console.log('  ║  • OpenAI:   /v1/chat/completions     ║');
    console.log('  ╠══════════════════════════════════════╣');
    console.log('  ║  Claude Code:                        ║');
    console.log(`  ║  export ANTHROPIC_BASE_URL=           ║`);
    console.log(`  ║    http://localhost:${config.port}              ║`);
    console.log('  ║  OpenAI 兼容:                        ║');
    console.log(`  ║  OPENAI_BASE_URL=                     ║`);
    console.log(`  ║    http://localhost:${config.port}/v1            ║`);
    console.log('  ╚══════════════════════════════════════╝');
    console.log('');
});
