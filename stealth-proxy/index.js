/**
 * Stealth Proxy - 通过无头浏览器绕过 Vercel Bot Protection
 *
 * 架构：
 *   客户端 → cursor2api → stealth-proxy → (Chrome浏览器上下文) → cursor.com/api/chat
 *
 * 原理：
 *   1. 启动时用 stealth 浏览器访问 cursor.com，通过 JS Challenge 获取 _vcrcs cookie
 *   2. 在同一浏览器上下文内通过 page.evaluate(fetch) 代理 API 请求
 *   3. 定时刷新 challenge（_vcrcs 有效期 3600s，每 50 分钟刷新）
 *   4. 支持 SSE 流式响应透传
 */

const express = require('express');
const crypto = require('crypto');

const PORT = parseInt(process.env.PORT || '3011');
const CHALLENGE_URL = process.env.CHALLENGE_URL || 'https://cursor.com/cn/docs';
const REFRESH_INTERVAL = parseInt(process.env.REFRESH_INTERVAL || '3000000'); // 50 分钟
const CHALLENGE_WAIT = parseInt(process.env.CHALLENGE_WAIT || '30000'); // challenge 最长等待时间

let browser, context, challengePage, workerPage;
let ready = false;
let startTime = Date.now();
let challengeCount = 0;
let requestCount = 0;

const pendingRequests = new Map();

// ==================== 浏览器管理 ====================

async function loadStealth() {
    const { chromium } = require('playwright-extra');
    const stealth = require('puppeteer-extra-plugin-stealth');
    chromium.use(stealth());
    return chromium;
}

async function initBrowser() {
    const chromium = await loadStealth();

    console.log('[Stealth] Launching browser...');
    browser = await chromium.launch({
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
        ],
    });

    context = await browser.newContext({
        userAgent:
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
        locale: 'zh-CN',
        viewport: { width: 1920, height: 1080 },
    });

    // ---- Challenge 页面：获取 _vcrcs ----
    challengePage = await context.newPage();
    console.log(`[Stealth] Passing Vercel challenge: ${CHALLENGE_URL}`);
    await challengePage.goto(CHALLENGE_URL, {
        waitUntil: 'domcontentloaded',
        timeout: 60000,
    });

    // 等待 cookie（challenge JS 会异步设置 _vcrcs）
    const ok = await waitForCookie();
    if (!ok) {
        // 重试一次：刷新页面再等
        console.log('[Stealth] First attempt failed, retrying challenge...');
        await challengePage.reload({ waitUntil: 'domcontentloaded', timeout: 60000 });
        const retryOk = await waitForCookie();
        if (!retryOk) {
            throw new Error('Failed to obtain _vcrcs cookie after retry');
        }
    }
    challengeCount++;

    // ---- Worker 页面：代理 API 请求 ----
    // cookie 已在 challenge 页面获取，worker 页面只需加载到 domcontentloaded 即可
    workerPage = await context.newPage();
    await workerPage.goto(CHALLENGE_URL, {
        waitUntil: 'domcontentloaded',
        timeout: 60000,
    });

    // 注册流式回调（Node.js 侧接收浏览器内 fetch 的数据块）
    await workerPage.exposeFunction(
        '__proxyCallback',
        (requestId, type, data) => {
            const pending = pendingRequests.get(requestId);
            if (!pending) return;

            switch (type) {
                case 'headers': {
                    const { status, contentType } = JSON.parse(data);
                    const headers = {
                        'Cache-Control': 'no-cache',
                        Connection: 'keep-alive',
                    };
                    if (contentType) headers['Content-Type'] = contentType;
                    pending.res.writeHead(status, headers);
                    break;
                }
                case 'chunk':
                    pending.res.write(data);
                    break;
                case 'end':
                    pending.res.end();
                    pending.resolve();
                    break;
                case 'error':
                    if (!pending.res.headersSent) {
                        pending.res.writeHead(502, {
                            'Content-Type': 'application/json',
                        });
                    }
                    pending.res.end(
                        JSON.stringify({ error: { message: data } }),
                    );
                    pending.resolve();
                    break;
            }
        },
    );

    ready = true;
    console.log('[Stealth] Ready! Accepting proxy requests.');
}

async function waitForCookie(maxWait) {
    maxWait = maxWait || CHALLENGE_WAIT;
    const start = Date.now();
    while (Date.now() - start < maxWait) {
        const cookies = await context.cookies();
        const vcrcs = cookies.find((c) => c.name === '_vcrcs');
        if (vcrcs) {
            console.log(
                '[Stealth] _vcrcs obtained:',
                vcrcs.value.substring(0, 40) + '...',
            );
            return true;
        }
        await new Promise((r) => setTimeout(r, 2000));
    }
    console.error('[Stealth] Failed to obtain _vcrcs within timeout');
    return false;
}

async function refreshChallenge() {
    console.log('[Stealth] Refreshing challenge...');
    try {
        await challengePage.goto(CHALLENGE_URL, {
            waitUntil: 'networkidle',
            timeout: 30000,
        });
        const ok = await waitForCookie();
        if (ok) {
            challengeCount++;
            console.log(
                `[Stealth] Challenge refreshed (total: ${challengeCount})`,
            );
        } else {
            console.error('[Stealth] Challenge refresh failed - cookie not obtained');
        }
    } catch (e) {
        console.error('[Stealth] Challenge refresh error:', e.message);
    }
}

async function restartBrowser() {
    console.log('[Stealth] Restarting browser...');
    ready = false;
    try {
        if (browser) await browser.close().catch(() => {});
    } catch (_) {}
    browser = null;
    context = null;
    challengePage = null;
    workerPage = null;
    await initBrowser();
}

// ==================== HTTP 服务 ====================

const app = express();
app.use(express.json({ limit: '10mb' }));

// 健康检查
app.get('/health', async (_req, res) => {
    let cookie = null;
    if (context) {
        const cookies = await context.cookies().catch(() => []);
        const vcrcs = cookies.find((c) => c.name === '_vcrcs');
        if (vcrcs) cookie = vcrcs.value.substring(0, 40) + '...';
    }
    res.json({
        status: ready ? 'ok' : 'initializing',
        uptime: Math.floor((Date.now() - startTime) / 1000),
        challengeCount,
        requestCount,
        cookie,
    });
});

// 代理请求
app.post('/proxy/chat', async (req, res) => {
    if (!ready) {
        res.status(503).json({
            error: { message: 'Stealth proxy not ready, please wait' },
        });
        return;
    }

    const requestId = crypto.randomUUID();
    requestCount++;

    // 客户端断开时清理
    let aborted = false;
    req.on('close', () => {
        aborted = true;
    });

    const promise = new Promise((resolve) => {
        pendingRequests.set(requestId, { res, resolve });
    });

    // 在浏览器上下文内发起 fetch 并流式回传
    workerPage
        .evaluate(
            async ({ body, requestId }) => {
                try {
                    const r = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body),
                    });

                    await window.__proxyCallback(
                        requestId,
                        'headers',
                        JSON.stringify({
                            status: r.status,
                            contentType: r.headers.get('content-type'),
                        }),
                    );

                    if (!r.body) {
                        const text = await r.text();
                        if (text)
                            await window.__proxyCallback(
                                requestId,
                                'chunk',
                                text,
                            );
                        await window.__proxyCallback(requestId, 'end', '');
                        return;
                    }

                    const reader = r.body.getReader();
                    const decoder = new TextDecoder();
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value, { stream: true });
                        if (chunk)
                            await window.__proxyCallback(
                                requestId,
                                'chunk',
                                chunk,
                            );
                    }
                    await window.__proxyCallback(requestId, 'end', '');
                } catch (e) {
                    await window.__proxyCallback(
                        requestId,
                        'error',
                        e.message || 'Browser fetch failed',
                    );
                }
            },
            { body: req.body, requestId },
        )
        .catch((err) => {
            const pending = pendingRequests.get(requestId);
            if (pending && !pending.res.headersSent) {
                pending.res.writeHead(502, {
                    'Content-Type': 'application/json',
                });
                pending.res.end(
                    JSON.stringify({
                        error: {
                            message: 'Browser evaluate failed: ' + err.message,
                        },
                    }),
                );
                pending.resolve();
            }
        });

    await promise;
    pendingRequests.delete(requestId);
});

// ==================== 启动 ====================

(async () => {
    try {
        await initBrowser();

        app.listen(PORT, '0.0.0.0', () => {
            console.log(`[Stealth] Proxy listening on port ${PORT}`);
        });

        // 定时刷新 challenge
        setInterval(refreshChallenge, REFRESH_INTERVAL);

        // 浏览器崩溃恢复
        browser.on('disconnected', () => {
            console.error('[Stealth] Browser disconnected! Restarting...');
            ready = false;
            setTimeout(restartBrowser, 3000);
        });
    } catch (e) {
        console.error('[Stealth] Fatal error:', e);
        process.exit(1);
    }
})();

// 优雅退出
const shutdown = async () => {
    console.log('[Stealth] Shutting down...');
    ready = false;
    if (browser) await browser.close().catch(() => {});
    process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
