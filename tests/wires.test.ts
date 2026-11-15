// Provider answer parsing: the three wire shapes the env answers with,
// reduced to one canonical answer (text, usage, stop reason).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseAnthropicAnswer, parseGeminiAnswer, parseOpenAIAnswer } from '../lib/wires.ts';

test('openai answer: text, usage and finish reason', () => {
  const answer = parseOpenAIAnswer({
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: 1767225600,
    model: 'm',
    choices: [{ index: 0, message: { role: 'assistant', content: 'ANSWER ECHO{}' }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 },
  });
  assert.deepEqual(answer, { text: 'ANSWER ECHO{}', inputTokens: 11, outputTokens: 7, stopReason: 'stop' });
});

test('anthropic answer: text blocks join with no separator', () => {
  const answer = parseAnthropicAnswer({
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model: 'm',
    content: [{ type: 'text', text: 'ANSWER ECHO{}' }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: 3, output_tokens: 9 },
  });
  assert.deepEqual(answer, { text: 'ANSWER ECHO{}', inputTokens: 3, outputTokens: 9, stopReason: 'end_turn' });
});

test('gemini answer: single text part, usageMetadata, STOP', () => {
  const answer = parseGeminiAnswer({
    candidates: [{ content: { parts: [{ text: 'ANSWER ECHO{}' }], role: 'model' }, finishReason: 'STOP', index: 0 }],
    usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 6, totalTokenCount: 11 },
    modelVersion: 'm',
  });
  assert.deepEqual(answer, { text: 'ANSWER ECHO{}', inputTokens: 5, outputTokens: 6, stopReason: 'STOP' });
});

test('malformed answers throw instead of fabricating usage', () => {
  assert.throws(() => parseOpenAIAnswer({ choices: [] }));
  assert.throws(() => parseGeminiAnswer({ candidates: [] }));
  assert.throws(() => parseOpenAIAnswer('nope'));
});
