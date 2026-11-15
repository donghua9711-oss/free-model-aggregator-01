// Client-surface rendering: the provider's buffered answer, written back
// on the client's wire — JSON or server-sent events, OpenAI or Anthropic
// shape. Everything observable is deterministic: fixed timestamps, ids
// from the request counter, the provider's own text, usage and stop
// family relayed exactly (the stop word is translated within its family:
// STOP / stop / end_turn are one family).

import { freeModelIds, type Catalog } from './catalog.ts';
import type { ClientApi } from './echo.ts';
import type { ProviderAnswer } from './wires.ts';

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const DELTA_CHARS = 13;

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);
const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

/** The stop word for a client surface, in the provider stop's family. */
export function stopForSurface(stopReason: string, api: ClientApi): string {
  const family = STOP_FAMILIES.get(stopReason) ?? 0;
  return (api === 'openai' ? OPENAI_STOPS[family] : ANTHROPIC_STOPS[family]) as string;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) pieces.push(text.slice(at, at + DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------- OpenAI surface ----------

export function renderOpenAIJson(answer: ProviderAnswer, model: string, seq: number): string {
  return JSON.stringify({
    id: `chatcmpl-agg-${seq}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: stopForSurface(answer.stopReason, 'openai'),
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
}

export function renderOpenAISse(answer: ProviderAnswer, model: string, seq: number): string {
  const base = { id: `chatcmpl-agg-${seq}`, object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const frames: string[] = [
    sseFrame(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseFrame(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  frames.push(
    sseFrame(null, {
      ...base,
      choices: [{ index: 0, delta: {}, finish_reason: stopForSurface(answer.stopReason, 'openai') }],
    }),
  );
  frames.push(
    sseFrame(null, {
      ...base,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

// ---------- Anthropic surface ----------

export function renderAnthropicJson(answer: ProviderAnswer, model: string, seq: number): string {
  return JSON.stringify({
    id: `msg_agg_${seq}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopForSurface(answer.stopReason, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  });
}

export function renderAnthropicSse(answer: ProviderAnswer, model: string, seq: number): string {
  const frames: string[] = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: `msg_agg_${seq}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseFrame('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopForSurface(answer.stopReason, 'anthropic'), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  frames.push(sseFrame('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------- model list ----------

/** GET /v1/models in the shape of the caller's wire: the registered free ids. */
export function renderModelsList(catalog: Catalog, api: ClientApi): string {
  const ids = [...new Set(freeModelIds(catalog))];
  if (api === 'anthropic') {
    return JSON.stringify({
      data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT })),
      has_more: false,
      first_id: ids[0] ?? null,
      last_id: ids[ids.length - 1] ?? null,
    });
  }
  return JSON.stringify({
    object: 'list',
    data: ids.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
  });
}
