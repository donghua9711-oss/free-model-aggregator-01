// Client-surface rendering: the answer the client reads back must carry
// the provider's text, usage and stop-reason family exactly, in a shape
// each wire's parser accepts, with stream frames in the legal order.
// These tests re-check the invariants the harness's response parsers
// enforce (program/env/server.mjs: openAIJson/openAIStream/
// anthropicJson/anthropicStream).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { ProviderAnswer } from '../lib/wires.ts';
import {
  renderAnthropicJson,
  renderAnthropicSse,
  renderModelsList,
  renderOpenAIJson,
  renderOpenAISse,
} from '../lib/render.ts';
import { buildCatalog, type ProbeResult } from '../lib/catalog.ts';
import { parseBrief } from '../lib/brief.ts';
import { readFileSync } from 'node:fs';

const ANSWER: ProviderAnswer = {
  text: 'ANSWER ECHO{"user_sha8":"abc"}',
  inputTokens: 12,
  outputTokens: 34,
  stopReason: 'STOP',
};

interface Frame {
  event: string | null;
  data: string;
}

function frames(sse: string): Frame[] {
  return sse
    .split('\n\n')
    .filter((f) => f.trim() !== '')
    .map((f) => {
      let event: string | null = null;
      const data: string[] = [];
      for (const line of f.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      return { event, data: data.join('\n') };
    });
}

test('openai JSON: text, usage and a family-0 finish reason', () => {
  const body = JSON.parse(renderOpenAIJson(ANSWER, 'some-model', 1)) as Record<string, any>;
  assert.equal(body.choices[0].message.role, 'assistant');
  assert.equal(body.choices[0].message.content, ANSWER.text);
  assert.equal(body.choices[0].finish_reason, 'stop'); // STOP maps to the openai family-0 word
  assert.equal(body.usage.prompt_tokens, 12);
  assert.equal(body.usage.completion_tokens, 34);
  assert.equal(body.model, 'some-model');
});

test('anthropic JSON: a message with the text, usage and end_turn', () => {
  const body = JSON.parse(renderAnthropicJson(ANSWER, 'some-model', 2)) as Record<string, any>;
  assert.equal(body.type, 'message');
  assert.equal(body.content[0].type, 'text');
  assert.equal(body.content[0].text, ANSWER.text);
  assert.equal(body.stop_reason, 'end_turn');
  assert.equal(body.usage.input_tokens, 12);
  assert.equal(body.usage.output_tokens, 34);
});

test('openai SSE: content before one finish chunk, usage last, [DONE] at the end', () => {
  const sse = renderOpenAISse(ANSWER, 'some-model', 3);
  const fs = frames(sse);
  assert.equal(fs[fs.length - 1]?.data, '[DONE]');
  const chunks = fs.slice(0, -1).map((f) => JSON.parse(f.data) as Record<string, any>);
  const finishAt = chunks.findIndex((c) => typeof c.choices?.[0]?.finish_reason === 'string');
  assert.ok(finishAt > 0, 'a finish chunk exists');
  assert.equal(chunks.filter((c) => typeof c.choices?.[0]?.finish_reason === 'string').length, 1);
  const text = chunks
    .slice(0, finishAt)
    .map((c) => c.choices?.[0]?.delta?.content ?? '')
    .join('');
  assert.equal(text, ANSWER.text);
  const after = chunks.slice(finishAt + 1);
  assert.ok(after.every((c) => !(c.choices?.[0]?.delta?.content ?? '')), 'no content after the finish chunk');
  const usageChunk = chunks.find((c) => Array.isArray(c.choices) && c.choices.length === 0 && c.usage);
  assert.ok(usageChunk, 'a usage chunk exists');
  assert.equal(usageChunk.usage.prompt_tokens, 12);
  assert.equal(usageChunk.usage.completion_tokens, 34);
});

test('anthropic SSE: message_start first, one balanced block, message_delta then message_stop', () => {
  const sse = renderAnthropicSse(ANSWER, 'some-model', 4);
  const fs = frames(sse);
  const events = fs.map((f) => f.event);
  assert.deepEqual(events, [
    'message_start',
    'content_block_start',
    ...Array(events.length - 5).fill('content_block_delta'),
    'content_block_stop',
    'message_delta',
    'message_stop',
  ]);
  const start = JSON.parse(fs[0]?.data ?? '{}') as Record<string, any>;
  assert.equal(start.message.usage.input_tokens, 12);
  const text = fs
    .filter((f) => f.event === 'content_block_delta')
    .map((f) => (JSON.parse(f.data) as Record<string, any>).delta.text as string)
    .join('');
  assert.equal(text, ANSWER.text);
  const delta = JSON.parse(fs[fs.length - 2]?.data ?? '{}') as Record<string, any>;
  assert.equal(delta.delta.stop_reason, 'end_turn');
  assert.equal(delta.usage.output_tokens, 34);
});

test('rendering is deterministic: same input, same bytes, counter-based ids', () => {
  assert.equal(renderOpenAIJson(ANSWER, 'm', 7), renderOpenAIJson(ANSWER, 'm', 7));
  assert.notEqual(renderOpenAIJson(ANSWER, 'm', 7), renderOpenAIJson(ANSWER, 'm', 8));
  assert.equal(renderAnthropicSse(ANSWER, 'm', 7), renderAnthropicSse(ANSWER, 'm', 7));
});

test('models list: both shapes carry exactly the registered free ids', () => {
  const kit = `${process.env.HOME}/workspace/makefun/free-model-aggregator-01-kit/practice/trial-0`;
  const brief = parseBrief(readFileSync(`${kit}/input/prompt.txt`, 'utf8'));
  const probes: ProbeResult[] = brief.providers.map((provider) => ({
    provider,
    ok: true,
    models: brief.rows
      .filter((row) => row.provider === provider.id)
      .map((row) => ({ id: row.model, free: row.free, contextLength: row.contextLength })),
  }));
  const catalog = buildCatalog(brief, probes);
  const openai = JSON.parse(renderModelsList(catalog, 'openai')) as Record<string, any>;
  const anthropic = JSON.parse(renderModelsList(catalog, 'anthropic')) as Record<string, any>;
  const expected = new Set(brief.rows.filter((r) => r.free).map((r) => r.model));
  assert.deepEqual(new Set(openai.data.map((m: any) => m.id as string)), expected);
  assert.deepEqual(new Set(anthropic.data.map((m: any) => m.id as string)), expected);
});
