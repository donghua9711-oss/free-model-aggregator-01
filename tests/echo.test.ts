// Echo round-trip tests.
//
// The harness grades a request by comparing the echo a provider computed
// from the body WE sent it against the echo it computed from the client's
// original body. These tests re-implement the harness's three request
// parsers (program/env/server.mjs: openAIParsed / anthropicParsed /
// geminiParsed) as the oracle, then prove for every practice workload body
// and every provider wire: parse(render(extract(body))) === parse(body).
//
// The one legal exception: an OpenAI client body in JSON mode can never be
// echoed by an Anthropic-wire provider (its parser hard-codes
// json_mode=false), so routing must exclude that pairing — asserted here.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { extractRequest } from '../lib/echo.ts';
import { renderAnthropic, renderGemini, renderOpenAI } from '../lib/wires.ts';

// ---------- oracle: the harness's parsers, copied semantics ----------

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

function sha8Text(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 8);
}
function sha8Bytes(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex').slice(0, 8);
}
function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}
function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}
function imageShaOfUrl(url: unknown): string | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return sha8Bytes(Buffer.from(data[1] ?? '', 'base64'));
  return sha8Text(url);
}

interface Echo {
  system_present: boolean;
  system_sha8: string | null;
  user_sha8: string;
  tools: string[];
  max_tokens: number | null;
  stops: string[];
  image_sha8: string | null;
  json_mode: boolean;
}

function echoOf(o: {
  systemText: string;
  userText: string;
  tools: string[];
  maxTokens: number | null;
  stops: string[];
  imageSha8: string | null;
  jsonMode: boolean;
}): Echo {
  return {
    system_present: o.systemText !== '',
    system_sha8: o.systemText === '' ? null : sha8Text(o.systemText),
    user_sha8: sha8Text(o.userText),
    tools: o.tools,
    max_tokens: o.maxTokens,
    stops: o.stops,
    image_sha8: o.imageSha8,
    json_mode: o.jsonMode,
  };
}

function openAIEcho(body: Record<string, unknown>): Echo {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'image_url' && isRecord(block.image_url)) {
            imageSha8 = imageShaOfUrl(block.image_url.url) ?? imageSha8;
          }
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') {
        tools.push(tool.function.name);
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return echoOf({
    systemText: systemParts.join('\n'),
    userText,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    imageSha8,
    jsonMode: format === 'json_object' || format === 'json_schema',
  });
}

function anthropicEcho(body: Record<string, unknown>): Echo {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          imageSha8 = sha8Bytes(Buffer.from(block.source.data, 'base64'));
        } else if (block.source.type === 'url') {
          imageSha8 = imageShaOfUrl(block.source.url) ?? imageSha8;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
    }
  }
  return echoOf({
    systemText,
    userText,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    imageSha8,
    jsonMode: false,
  });
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('\n');
}

function geminiEcho(body: Record<string, unknown>): Echo {
  const systemText = isRecord(body.systemInstruction) ? geminiPartsText(body.systemInstruction.parts) : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter(isRecord) : [];
  let userText = '';
  let imageSha8: string | null = null;
  for (const content of contents) {
    if (content.role === 'model') continue;
    userText = geminiPartsText(content.parts);
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (!isRecord(part)) continue;
        if (isRecord(part.inlineData) && typeof part.inlineData.data === 'string') {
          imageSha8 = sha8Bytes(Buffer.from(part.inlineData.data, 'base64'));
        } else if (isRecord(part.fileData) && typeof part.fileData.fileUri === 'string') {
          imageSha8 = imageShaOfUrl(part.fileData.fileUri) ?? imageSha8;
        }
      }
    }
  }
  const tools: string[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !Array.isArray(tool.functionDeclarations)) continue;
      for (const fn of tool.functionDeclarations) {
        if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
      }
    }
  }
  const config = isRecord(body.generationConfig) ? body.generationConfig : {};
  return echoOf({
    systemText,
    userText,
    tools,
    maxTokens: numberOrNull(config.maxOutputTokens),
    stops: stringList(config.stopSequences),
    imageSha8,
    jsonMode: config.responseMimeType === 'application/json',
  });
}

// ---------- the round-trip over every practice body ----------

interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
}

function loadTrial(n: number): { manifest: ManifestEntry[]; bodies: Record<string, Record<string, unknown>> } {
  const path = `${process.env.HOME}/workspace/makefun/free-model-aggregator-01-kit/practice/trial-${n}/input/env/workload.json`;
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as {
    manifest: ManifestEntry[];
    bodies: Record<string, Record<string, unknown>>;
  };
  return parsed;
}

for (const trial of [0, 1, 2]) {
  const { manifest, bodies } = loadTrial(trial);
  for (const entry of manifest) {
    test(`trial ${trial} ${entry.id} (${entry.api}) echo survives every wire`, () => {
      const body = bodies[entry.id] as Record<string, unknown>;
      const expected = entry.api === 'openai' ? openAIEcho(body) : anthropicEcho(body);
      const canonical = extractRequest(entry.api, body);

      assert.deepEqual(geminiEcho(renderGemini(canonical)), expected, 'gemini wire');
      assert.deepEqual(openAIEcho(renderOpenAI(canonical, 'some-model')), expected, 'openai wire');
      if (expected.json_mode) {
        // The one impossible pairing: anthropic's parser hard-codes false.
        assert.equal(canonical.jsonMode, true);
        assert.equal(anthropicEcho(renderAnthropic(canonical, 'some-model')).json_mode, false);
      } else {
        assert.deepEqual(anthropicEcho(renderAnthropic(canonical, 'some-model')), expected, 'anthropic wire');
      }
    });
  }
}

test('extraction keeps max_tokens absent as null, never a default', () => {
  const canonical = extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(canonical.maxTokens, null);
  assert.equal(openAIEcho(renderOpenAI(canonical, 'm')).max_tokens, null);
  assert.equal(geminiEcho(renderGemini(canonical)).max_tokens, null);
});

test('extraction reads max_completion_tokens when max_tokens is absent', () => {
  const canonical = extractRequest('openai', {
    model: 'auto',
    max_completion_tokens: 77,
    messages: [{ role: 'user', content: 'hi' }],
  });
  assert.equal(canonical.maxTokens, 77);
});

test('a bare string stop becomes a one-element stop list on every wire', () => {
  const canonical = extractRequest('openai', {
    model: 'auto',
    stop: 'END',
    messages: [{ role: 'user', content: 'hi' }],
  });
  assert.deepEqual(canonical.stops, ['END']);
  assert.deepEqual(anthropicEcho(renderAnthropic(canonical, 'm')).stops, ['END']);
});
