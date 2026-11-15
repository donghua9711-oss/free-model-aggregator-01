// Canonical request extraction.
//
// A client body (OpenAI or Anthropic shape) is reduced to one canonical
// request: the system text, the conversation as block-for-block messages,
// the tool definitions, and the scalar knobs (max tokens, stops, JSON mode,
// temperature). The reduction mirrors the harness's own request parsers
// field for field, so a provider body rendered from the canonical request
// (lib/wires.ts) parses back to exactly the echo the client body produced:
// same system text, same last-user text, same tool names in order, same
// max_tokens, same stops, same image hash, same JSON-mode flag.

export type ClientApi = 'openai' | 'anthropic';

export type Block =
  | { kind: 'text'; text: string }
  | { kind: 'imageData'; mediaType: string; data: string }
  | { kind: 'imageUrl'; url: string }
  | { kind: 'toolUse'; id: string; name: string; input: unknown }
  | { kind: 'toolResult'; toolUseId: string; name: string | null; blocks: Block[]; isError: boolean };

export interface CanonMessage {
  role: 'user' | 'assistant';
  blocks: Block[];
}

export interface CanonTool {
  name: string;
  description: string | null;
  schema: unknown;
}

export interface CanonicalRequest {
  systemText: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
}

type Rec = Record<string, unknown>;

export function isRecord(v: unknown): v is Rec {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The harness's blocksText: a string is itself; blocks join text with "\n". */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** Text of a block list the way blocksText would read it back. */
export function textOfBlocks(blocks: Block[]): string {
  return blocks
    .filter((b): b is Extract<Block, { kind: 'text' }> => b.kind === 'text')
    .map((b) => b.text)
    .join('\n');
}

const DATA_URL = /^data:([^;,]*)(;base64)?,(.*)$/s;

function imageBlockFromUrl(url: string): Block {
  const m = DATA_URL.exec(url);
  if (m !== null && m[2] === ';base64') {
    return { kind: 'imageData', mediaType: m[1] === '' ? 'image/png' : (m[1] as string), data: m[3] as string };
  }
  return { kind: 'imageUrl', url };
}

function parseJsonOrText(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

// ---------- OpenAI client bodies ----------

function openAIBlocks(content: unknown): Block[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Block[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') {
      blocks.push({ kind: 'text', text: part.text });
    } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      blocks.push(imageBlockFromUrl(part.image_url.url));
    }
    // Other part types carry no echo field; dropping them is echo-neutral.
  }
  return blocks;
}

function extractOpenAI(body: Rec): CanonicalRequest {
  const rawMessages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const messages: CanonMessage[] = [];
  const toolNames = new Map<string, string>();
  for (const message of rawMessages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      messages.push({ role: 'user', blocks: openAIBlocks(message.content) });
    } else if (message.role === 'assistant') {
      const blocks = openAIBlocks(message.content);
      if (Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
          const id = typeof call.id === 'string' ? call.id : '';
          toolNames.set(id, call.function.name);
          blocks.push({
            kind: 'toolUse',
            id,
            name: call.function.name,
            input: typeof call.function.arguments === 'string' ? parseJsonOrText(call.function.arguments) : null,
          });
        }
      }
      messages.push({ role: 'assistant', blocks });
    } else if (message.role === 'tool') {
      const id = typeof message.tool_call_id === 'string' ? message.tool_call_id : '';
      messages.push({
        role: 'user',
        blocks: [
          {
            kind: 'toolResult',
            toolUseId: id,
            name: toolNames.get(id) ?? null,
            blocks: openAIBlocks(message.content),
            isError: false,
          },
        ],
      });
    }
    // Any other role contributes nothing to the echo; it is dropped.
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : null,
        schema: tool.function.parameters ?? null,
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numberOrNull(body.temperature),
  };
}

// ---------- Anthropic client bodies ----------

function anthropicBlocks(content: unknown, toolNames: Map<string, string>): Block[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: Block[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      blocks.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        blocks.push({
          kind: 'imageData',
          mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
          data: block.source.data,
        });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        blocks.push({ kind: 'imageUrl', url: block.source.url });
      }
    } else if (block.type === 'tool_use' && typeof block.name === 'string') {
      const id = typeof block.id === 'string' ? block.id : '';
      toolNames.set(id, block.name);
      blocks.push({ kind: 'toolUse', id, name: block.name, input: block.input ?? null });
    } else if (block.type === 'tool_result') {
      const id = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
      blocks.push({
        kind: 'toolResult',
        toolUseId: id,
        name: toolNames.get(id) ?? null,
        blocks: anthropicBlocks(block.content, toolNames),
        isError: block.is_error === true,
      });
    }
    // Other block types carry no echo field; dropping them is echo-neutral.
  }
  return blocks;
}

function extractAnthropic(body: Rec): CanonicalRequest {
  const toolNames = new Map<string, string>();
  const rawMessages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const messages: CanonMessage[] = [];
  for (const message of rawMessages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue;
    messages.push({ role: message.role, blocks: anthropicBlocks(message.content, toolNames) });
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : null,
        schema: tool.input_schema ?? null,
      });
    }
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false, // the Anthropic wire has no JSON mode to ask for
    temperature: numberOrNull(body.temperature),
  };
}

export function extractRequest(api: ClientApi, body: Rec): CanonicalRequest {
  return api === 'openai' ? extractOpenAI(body) : extractAnthropic(body);
}
