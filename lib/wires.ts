// Provider wires: render a canonical request into each provider's wire
// format, make the (always non-streaming) call, and parse the answer.
//
// Rendering is the inverse of lib/echo.ts extraction: every field the
// provider's parser reads is reproduced exactly — system text as one unit,
// messages block for block, tool names in order, max_tokens only when the
// client sent one, stops only when non-empty, images with their bytes
// intact, and JSON mode only where the wire can express it.

import {
  isRecord,
  textOfBlocks,
  type Block,
  type CanonMessage,
  type CanonicalRequest,
} from './echo.ts';

export type Wire = 'openai' | 'anthropic' | 'gemini';

export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopReason: string;
}

// ---------- OpenAI wire ----------

function openAIUserContent(blocks: Block[]): unknown {
  const rendered = blocks.map((block) => {
    switch (block.kind) {
      case 'text':
        return { type: 'text', text: block.text };
      case 'imageData':
        return { type: 'image_url', image_url: { url: `data:${block.mediaType};base64,${block.data}` } };
      case 'imageUrl':
        return { type: 'image_url', image_url: { url: block.url } };
      default:
        return null; // tool blocks are split out by openAIMessage
    }
  }).filter((b) => b !== null);
  if (rendered.length === 1 && (rendered[0] as { type: string }).type === 'text') {
    return (rendered[0] as { text: string }).text;
  }
  return rendered;
}

function openAIMessage(message: CanonMessage): Record<string, unknown>[] {
  if (message.role === 'assistant') {
    const toolCalls = message.blocks
      .filter((b): b is Extract<Block, { kind: 'toolUse' }> => b.kind === 'toolUse')
      .map((b) => ({
        id: b.id,
        type: 'function',
        function: { name: b.name, arguments: typeof b.input === 'string' ? b.input : JSON.stringify(b.input ?? {}) },
      }));
    const text = textOfBlocks(message.blocks);
    const out: Record<string, unknown> = { role: 'assistant', content: text === '' ? null : text };
    if (toolCalls.length > 0) out.tool_calls = toolCalls;
    return [out];
  }
  const results = message.blocks.filter(
    (b): b is Extract<Block, { kind: 'toolResult' }> => b.kind === 'toolResult',
  );
  const rest = message.blocks.filter((b) => b.kind !== 'toolResult');
  const out: Record<string, unknown>[] = results.map((b) => ({
    role: 'tool',
    tool_call_id: b.toolUseId,
    content: textOfBlocks(b.blocks),
  }));
  // A user turn always stays a user turn, even when it held only tool
  // results: the provider reads the user text from the last user message.
  out.push({ role: 'user', content: openAIUserContent(rest) });
  return out;
}

export function renderOpenAI(req: CanonicalRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
  for (const message of req.messages) messages.push(...openAIMessage(message));
  const body: Record<string, unknown> = { model, messages };
  if (req.tools.length > 0) {
    body.tools = req.tools.map((tool) => {
      const fn: Record<string, unknown> = { name: tool.name };
      if (tool.description !== null) fn.description = tool.description;
      if (tool.schema !== null) fn.parameters = tool.schema;
      return { type: 'function', function: fn };
    });
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.stops.length > 0) body.stop = req.stops;
  if (req.jsonMode) body.response_format = { type: 'json_object' };
  if (req.temperature !== null) body.temperature = req.temperature;
  return body;
}

// ---------- Anthropic wire ----------

function anthropicContent(blocks: Block[]): unknown {
  if (blocks.length === 1 && blocks[0]?.kind === 'text') return (blocks[0] as { text: string }).text;
  return blocks.map((block) => {
    switch (block.kind) {
      case 'text':
        return { type: 'text', text: block.text };
      case 'imageData':
        return { type: 'image', source: { type: 'base64', media_type: block.mediaType, data: block.data } };
      case 'imageUrl':
        return { type: 'image', source: { type: 'url', url: block.url } };
      case 'toolUse':
        return { type: 'tool_use', id: block.id, name: block.name, input: block.input ?? {} };
      case 'toolResult': {
        const out: Record<string, unknown> = {
          type: 'tool_result',
          tool_use_id: block.toolUseId,
          content: anthropicContent(block.blocks),
        };
        if (block.isError) out.is_error = true;
        return out;
      };
    }
  });
}

export function renderAnthropic(req: CanonicalRequest, model: string): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model,
    messages: req.messages.map((message) => ({ role: message.role, content: anthropicContent(message.blocks) })),
  };
  if (req.systemText !== '') body.system = req.systemText;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((tool) => {
      const out: Record<string, unknown> = { name: tool.name };
      if (tool.description !== null) out.description = tool.description;
      out.input_schema = tool.schema ?? { type: 'object' };
      return out;
    });
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.stops.length > 0) body.stop_sequences = req.stops;
  if (req.temperature !== null) body.temperature = req.temperature;
  return body;
}

// ---------- Gemini wire ----------

function geminiParts(blocks: Block[]): Record<string, unknown>[] {
  return blocks.map((block) => {
    switch (block.kind) {
      case 'text':
        return { text: block.text };
      case 'imageData':
        return { inlineData: { mimeType: block.mediaType, data: block.data } };
      case 'imageUrl':
        return { fileData: { fileUri: block.url } };
      case 'toolUse':
        return { functionCall: { name: block.name, args: block.input ?? {} } };
      case 'toolResult':
        return {
          functionResponse: {
            name: block.name ?? block.toolUseId,
            response: { result: textOfBlocks(block.blocks) },
          },
        };
    }
  });
}

export function renderGemini(req: CanonicalRequest): Record<string, unknown> {
  const body: Record<string, unknown> = {
    contents: req.messages.map((message) => ({
      role: message.role === 'assistant' ? 'model' : 'user',
      parts: geminiParts(message.blocks),
    })),
  };
  if (req.systemText !== '') body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((tool) => {
          const out: Record<string, unknown> = { name: tool.name };
          if (tool.description !== null) out.description = tool.description;
          if (tool.schema !== null) out.parameters = tool.schema;
          return out;
        }),
      },
    ];
  }
  const config: Record<string, unknown> = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (req.temperature !== null) config.temperature = req.temperature;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------- answer parsing ----------

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

export function parseOpenAIAnswer(body: unknown): ProviderAnswer {
  if (!isRecord(body)) throw new Error('openai answer is not an object');
  const choice = Array.isArray(body.choices) ? body.choices[0] : undefined;
  if (!isRecord(choice) || !isRecord(choice.message)) throw new Error('openai answer holds no choice');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(choice.message.content),
    inputTokens: num(usage.prompt_tokens),
    outputTokens: num(usage.completion_tokens),
    stopReason: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

export function parseAnthropicAnswer(body: unknown): ProviderAnswer {
  if (!isRecord(body)) throw new Error('anthropic answer is not an object');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: answerText(body.content),
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
    stopReason: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

export function parseGeminiAnswer(body: unknown): ProviderAnswer {
  if (!isRecord(body)) throw new Error('gemini answer is not an object');
  const candidate = Array.isArray(body.candidates) ? body.candidates[0] : undefined;
  if (!isRecord(candidate) || !isRecord(candidate.content)) throw new Error('gemini answer holds no candidate');
  const parts = Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: parts
      .filter((p): p is Record<string, unknown> => isRecord(p) && typeof p.text === 'string')
      .map((p) => p.text as string)
      .join(''),
    inputTokens: num(usage.promptTokenCount),
    outputTokens: num(usage.candidatesTokenCount),
    stopReason: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

// ---------- the call ----------

export interface ProviderTarget {
  wire: Wire;
  auth: 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
  key: string;
  /** Absolute base URL of the provider inside the env, e.g. http://env/p/groq. */
  baseUrl: string;
}

export type CallOutcome =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; detail: string };

const CALL_TIMEOUT_MS = 10_000;

function callTarget(target: ProviderTarget, model: string): { url: string; headers: Record<string, string> } {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (target.auth === 'bearer') headers.authorization = `Bearer ${target.key}`;
  else if (target.auth === 'x-api-key') headers['x-api-key'] = target.key;
  else if (target.auth === 'x-goog-api-key') headers['x-goog-api-key'] = target.key;
  if (target.wire === 'openai') return { url: `${target.baseUrl}/v1/chat/completions`, headers };
  if (target.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return { url: `${target.baseUrl}/v1/messages`, headers };
  }
  return { url: `${target.baseUrl}/v1beta/models/${encodeURIComponent(model)}:generateContent`, headers };
}

/**
 * One non-streaming provider call. A refusal (402/429/5xx, or any non-2xx)
 * and a network failure both come back as { ok: false } so the router can
 * fail over; a 2xx is parsed into the canonical answer.
 */
export async function callProvider(
  target: ProviderTarget,
  req: CanonicalRequest,
  model: string,
): Promise<CallOutcome> {
  const { url, headers } = callTarget(target, model);
  const payload =
    target.wire === 'openai'
      ? renderOpenAI(req, model)
      : target.wire === 'anthropic'
        ? renderAnthropic(req, model)
        : renderGemini(req);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, detail: `network error: ${(err as Error).message}` };
  }
  if (!res.ok) {
    const detail = (await res.text().catch(() => '')).slice(0, 200);
    return { ok: false, status: res.status, detail };
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return { ok: false, status: res.status, detail: 'provider answered a non-JSON body' };
  }
  try {
    const answer =
      target.wire === 'openai'
        ? parseOpenAIAnswer(body)
        : target.wire === 'anthropic'
          ? parseAnthropicAnswer(body)
          : parseGeminiAnswer(body);
    return { ok: true, answer };
  } catch (err) {
    return { ok: false, status: res.status, detail: `unparseable answer: ${(err as Error).message}` };
  }
}
