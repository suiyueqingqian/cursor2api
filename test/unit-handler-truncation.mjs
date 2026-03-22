import { shouldAutoContinueTruncatedToolResponse } from '../dist/handler.js';

let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`  ✅ ${name}`);
        passed++;
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`  ❌ ${name}`);
        console.error(`      ${message}`);
        failed++;
    }
}

function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
}

console.log('\n📦 handler 截断续写判定\n');

test('短参数工具调用可恢复时不再继续续写', () => {
    const text = [
        '我先读取配置文件。',
        '',
        '```json action',
        '{',
        '  "tool": "Read",',
        '  "parameters": {',
        '    "file_path": "/app/config.yaml"',
        '  }',
    ].join('\n');

    assertEqual(
        shouldAutoContinueTruncatedToolResponse(text, true),
        false,
        'Read 这类短参数工具不应继续续写',
    );
});

test('大参数写入工具仍然继续续写', () => {
    const longContent = 'A'.repeat(4000);
    const text = [
        '```json action',
        '{',
        '  "tool": "Write",',
        '  "parameters": {',
        '    "file_path": "/tmp/large.txt",',
        `    "content": "${longContent}`,
    ].join('\n');

    assertEqual(
        shouldAutoContinueTruncatedToolResponse(text, true),
        true,
        'Write 大内容仍应继续续写以补全参数',
    );
});

test('普通代码块截断但文本过短（<200字）不续写', () => {
    // 200-char 保护：非 json action 块截断时，过短的响应缺乏上下文，不触发续写
    const text = '```ts\nexport const answer = {';

    assertEqual(
        shouldAutoContinueTruncatedToolResponse(text, true),
        false,
        '非 json action 块且文本 <200 chars 时不应续写',
    );
});

test('json action 块未闭合且文本过短时仍触发续写（thinking 剥离后场景）', () => {
    // 场景：thinking 剥离后 fullResponse 只剩 json action 块开头（很短）
    // 200-char 保护不应阻止这种明确的工具调用截断
    const text = '```json action\n{\n  "tool": "Write",';

    assertEqual(
        shouldAutoContinueTruncatedToolResponse(text, true),
        true,
        'json action 块未闭合时即使文本 <200 chars 也应续写',
    );
});

console.log(`\n结果: ${passed} 通过 / ${failed} 失败 / ${passed + failed} 总计\n`);

if (failed > 0) process.exit(1);
