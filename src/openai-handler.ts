/**
 * openai-handler.ts - OpenAI Chat Completions API 兼容处理器
 *
 * 将 OpenAI 格式请求转换为内部 Anthropic 格式，复用现有 Cursor 交互管道
 * 支持流式和非流式响应、工具调用
 */

import type { Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import type {
    OpenAIChatRequest,
    OpenAIMessage,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIToolCall,
} from './openai-types.js';
import type {
    AnthropicRequest,
    AnthropicMessage,
    AnthropicContentBlock,
    AnthropicTool,
    CursorSSEEvent,
} from './types.js';
import { convertToCursorRequest, parseToolCalls, hasToolCalls } from './converter.js';
import { sendCursorRequest, sendCursorRequestFull } from './cursor-client.js';
import { getConfig } from './config.js';

function chatId(): string {
    return 'chatcmpl-' + uuidv4().replace(/-/g, '').substring(0, 24);
}

function toolCallId(): string {
    return 'call_' + uuidv4().replace(/-/g, '').substring(0, 24);
}

// ==================== 请求转换：OpenAI → Anthropic ====================

/**
 * 将 OpenAI Chat Completions 请求转换为内部 Anthropic 格式
 * 这样可以完全复用现有的 convertToCursorRequest 管道
 */
function convertToAnthropicRequest(body: OpenAIChatRequest): AnthropicRequest {
    const messages: AnthropicMessage[] = [];
    let systemPrompt: string | undefined;

    for (const msg of body.messages) {
        switch (msg.role) {
            case 'system':
                // OpenAI system → Anthropic system
                systemPrompt = (systemPrompt ? systemPrompt + '\n\n' : '') + extractOpenAIContent(msg);
                break;

            case 'user':
                messages.push({
                    role: 'user',
                    content: extractOpenAIContent(msg),
                });
                break;

            case 'assistant': {
                // 助手消息可能包含 tool_calls
                const blocks: AnthropicContentBlock[] = [];
                const textContent = extractOpenAIContent(msg);
                if (textContent) {
                    blocks.push({ type: 'text', text: textContent });
                }

                if (msg.tool_calls && msg.tool_calls.length > 0) {
                    for (const tc of msg.tool_calls) {
                        let args: Record<string, unknown> = {};
                        try {
                            args = JSON.parse(tc.function.arguments);
                        } catch {
                            args = { input: tc.function.arguments };
                        }
                        blocks.push({
                            type: 'tool_use',
                            id: tc.id,
                            name: tc.function.name,
                            input: args,
                        });
                    }
                }

                messages.push({
                    role: 'assistant',
                    content: blocks.length > 0 ? blocks : (textContent || ''),
                });
                break;
            }

            case 'tool': {
                // OpenAI tool result → Anthropic tool_result
                messages.push({
                    role: 'user',
                    content: [{
                        type: 'tool_result',
                        tool_use_id: msg.tool_call_id,
                        content: extractOpenAIContent(msg),
                    }] as AnthropicContentBlock[],
                });
                break;
            }
        }
    }

    // 转换工具定义：OpenAI function → Anthropic tool
    const tools: AnthropicTool[] | undefined = body.tools?.map(t => ({
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters || { type: 'object', properties: {} },
    }));

    return {
        model: body.model,
        messages,
        max_tokens: body.max_tokens || body.max_completion_tokens || 8192,
        stream: body.stream,
        system: systemPrompt,
        tools,
        temperature: body.temperature,
        top_p: body.top_p,
        stop_sequences: body.stop
            ? (Array.isArray(body.stop) ? body.stop : [body.stop])
            : undefined,
    };
}

/**
 * 从 OpenAI 消息中提取文本内容
 */
function extractOpenAIContent(msg: OpenAIMessage): string {
    if (msg.content === null || msg.content === undefined) return '';
    if (typeof msg.content === 'string') return msg.content;
    if (Array.isArray(msg.content)) {
        return msg.content
            .filter(p => p.type === 'text' && p.text)
            .map(p => p.text!)
            .join('\n');
    }
    return String(msg.content);
}

// ==================== 主处理入口 ====================

export async function handleOpenAIChatCompletions(req: Request, res: Response): Promise<void> {
    const body = req.body as OpenAIChatRequest;

    console.log(`[OpenAI] 收到请求: model=${body.model}, messages=${body.messages?.length}, stream=${body.stream}, tools=${body.tools?.length ?? 0}`);

    try {
        // Step 1: OpenAI → Anthropic 格式
        const anthropicReq = convertToAnthropicRequest(body);

        // Step 2: Anthropic → Cursor 格式（复用现有管道）
        const cursorReq = convertToCursorRequest(anthropicReq);

        if (body.stream) {
            await handleOpenAIStream(res, cursorReq, body);
        } else {
            await handleOpenAINonStream(res, cursorReq, body);
        }
    } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        console.error(`[OpenAI] 请求处理失败:`, message);
        res.status(500).json({
            error: {
                message,
                type: 'server_error',
                code: 'internal_error',
            },
        });
    }
}

// ==================== 流式处理（OpenAI SSE 格式） ====================

async function handleOpenAIStream(
    res: Response,
    cursorReq: ReturnType<typeof convertToCursorRequest>,
    body: OpenAIChatRequest,
): Promise<void> {
    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    });

    const id = chatId();
    const created = Math.floor(Date.now() / 1000);
    const model = body.model;
    const hasTools = (body.tools?.length ?? 0) > 0;

    // 发送 role delta（OpenAI 流式第一个 chunk 通常包含 role）
    writeOpenAISSE(res, {
        id, object: 'chat.completion.chunk', created, model,
        choices: [{
            index: 0,
            delta: { role: 'assistant', content: '' },
            finish_reason: null,
        }],
    });

    let fullResponse = '';
    let sentText = '';

    try {
        await sendCursorRequest(cursorReq, (event: CursorSSEEvent) => {
            if (event.type !== 'text-delta' || !event.delta) return;

            fullResponse += event.delta;

            // 工具模式：缓冲直到完成
            if (hasTools && hasToolCalls(fullResponse)) {
                return;
            }

            // 实时流式推送文本
            writeOpenAISSE(res, {
                id, object: 'chat.completion.chunk', created, model,
                choices: [{
                    index: 0,
                    delta: { content: event.delta },
                    finish_reason: null,
                }],
            });
            sentText += event.delta;
        });

        // 流完成后处理
        let finishReason: 'stop' | 'tool_calls' = 'stop';

        if (hasTools && hasToolCalls(fullResponse)) {
            const { toolCalls, cleanText } = parseToolCalls(fullResponse);

            if (toolCalls.length > 0) {
                finishReason = 'tool_calls';

                // 发送工具调用前的剩余文本
                const matchLen = findMatchLength(cleanText, sentText);
                const unsentCleanText = cleanText.substring(matchLen).trim();

                if (unsentCleanText) {
                    writeOpenAISSE(res, {
                        id, object: 'chat.completion.chunk', created, model,
                        choices: [{
                            index: 0,
                            delta: { content: unsentCleanText },
                            finish_reason: null,
                        }],
                    });
                }

                // 发送每个工具调用
                for (let i = 0; i < toolCalls.length; i++) {
                    const tc = toolCalls[i];
                    // 工具调用开始（包含 id、name）
                    writeOpenAISSE(res, {
                        id, object: 'chat.completion.chunk', created, model,
                        choices: [{
                            index: 0,
                            delta: {
                                tool_calls: [{
                                    index: i,
                                    id: toolCallId(),
                                    type: 'function',
                                    function: {
                                        name: tc.name,
                                        arguments: JSON.stringify(tc.arguments),
                                    },
                                }],
                            },
                            finish_reason: null,
                        }],
                    });
                }
            } else {
                // 误报：发送剩余文本
                const unsentText = fullResponse.substring(sentText.length);
                if (unsentText) {
                    writeOpenAISSE(res, {
                        id, object: 'chat.completion.chunk', created, model,
                        choices: [{
                            index: 0,
                            delta: { content: unsentText },
                            finish_reason: null,
                        }],
                    });
                }
            }
        }

        // 发送完成 chunk
        writeOpenAISSE(res, {
            id, object: 'chat.completion.chunk', created, model,
            choices: [{
                index: 0,
                delta: {},
                finish_reason: finishReason,
            }],
        });

        // OpenAI 流式结束标志
        res.write('data: [DONE]\n\n');

    } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        // 在流中发送错误（非标准，但部分客户端可以处理）
        writeOpenAISSE(res, {
            id, object: 'chat.completion.chunk', created, model,
            choices: [{
                index: 0,
                delta: { content: `\n\n[Error: ${message}]` },
                finish_reason: 'stop',
            }],
        });
        res.write('data: [DONE]\n\n');
    }

    res.end();
}

// ==================== 非流式处理 ====================

async function handleOpenAINonStream(
    res: Response,
    cursorReq: ReturnType<typeof convertToCursorRequest>,
    body: OpenAIChatRequest,
): Promise<void> {
    const fullText = await sendCursorRequestFull(cursorReq);
    const hasTools = (body.tools?.length ?? 0) > 0;

    console.log(`[OpenAI] 原始响应 (${fullText.length} chars): ${fullText.substring(0, 300)}...`);

    let content: string | null = fullText;
    let toolCalls: OpenAIToolCall[] | undefined;
    let finishReason: 'stop' | 'tool_calls' = 'stop';

    if (hasTools) {
        const parsed = parseToolCalls(fullText);

        if (parsed.toolCalls.length > 0) {
            finishReason = 'tool_calls';
            content = parsed.cleanText || null;

            toolCalls = parsed.toolCalls.map(tc => ({
                id: toolCallId(),
                type: 'function' as const,
                function: {
                    name: tc.name,
                    arguments: JSON.stringify(tc.arguments),
                },
            }));
        }
    }

    const response: OpenAIChatCompletion = {
        id: chatId(),
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: body.model,
        choices: [{
            index: 0,
            message: {
                role: 'assistant',
                content,
                ...(toolCalls ? { tool_calls: toolCalls } : {}),
            },
            finish_reason: finishReason,
        }],
        usage: {
            prompt_tokens: 100,
            completion_tokens: Math.ceil(fullText.length / 4),
            total_tokens: 100 + Math.ceil(fullText.length / 4),
        },
    };

    res.json(response);
}

// ==================== 工具函数 ====================

function writeOpenAISSE(res: Response, data: OpenAIChatCompletionChunk): void {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
    // @ts-expect-error flush exists on ServerResponse when compression is used
    if (typeof res.flush === 'function') res.flush();
}

/**
 * 找到 cleanText 中已经发送过的文本长度
 */
function findMatchLength(cleanText: string, sentText: string): number {
    for (let i = Math.min(cleanText.length, sentText.length); i >= 0; i--) {
        if (cleanText.startsWith(sentText.substring(0, i))) {
            return i;
        }
    }
    return 0;
}
