// The trial brief: three JSON documents inside the prompt text at
// $MAKEFUN_PROMPT_FILE — the provider directory, the ratings table and the
// workload manifest. Parsed and validated here, at the boundary; the rest
// of the program trusts these types.

import { isRecord } from './echo.ts';
import type { ClientApi } from './echo.ts';
import type { Wire } from './wires.ts';

export type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  keyEnv: string;
  key: string;
  path: string;
}

export interface ModelRow {
  provider: string;
  model: string;
  ratingMilli: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestRequest {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requirements;
}

export interface Brief {
  providers: ProviderEntry[];
  rows: ModelRow[];
  /** Keyed `${provider}/${model}`. */
  rowByKey: Map<string, ModelRow>;
  manifest: ManifestRequest[];
}

/** The harness's own rule: only a `true` flag binds, and min_context is a floor. */
export function meetsRequirements(row: ModelRow, requires: Requirements): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && row.contextLength < requires.min_context) return false;
  return true;
}

/** The balanced `{...}` slice starting at `start`, string-aware; null if it never closes. */
function balancedSlice(text: string, start: number): string | null {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i] as string;
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/** Every top-level JSON document in the prompt that carries a schema name. */
function documentsIn(text: string): Map<string, Record<string, unknown>> {
  const docs = new Map<string, Record<string, unknown>>();
  for (let i = 0; i < text.length; i++) {
    if (text[i] !== '{') continue;
    const slice = balancedSlice(text, i);
    if (slice === null) continue;
    try {
      const value: unknown = JSON.parse(slice);
      if (isRecord(value) && typeof value.schema === 'string' && !docs.has(value.schema)) {
        docs.set(value.schema, value);
      }
    } catch {
      // Not a document start; keep scanning.
    }
  }
  return docs;
}

function docNamed(docs: Map<string, Record<string, unknown>>, suffix: string): Record<string, unknown> {
  for (const [schema, doc] of docs) if (schema.endsWith(suffix)) return doc;
  throw new Error(`the brief holds no ${suffix} document`);
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function parseProviders(doc: Record<string, unknown>): ProviderEntry[] {
  const list = Array.isArray(doc.providers) ? doc.providers : [];
  return list.filter(isRecord).map((p) => ({
    id: str(p.id),
    name: str(p.name),
    wire: (str(p.wire) || 'openai') as Wire,
    auth: (str(p.auth) || 'none') as AuthKind,
    keyEnv: str(p.key_env),
    key: str(p.key),
    path: str(p.path),
  }));
}

function parseRows(doc: Record<string, unknown>): ModelRow[] {
  const list = Array.isArray(doc.ratings) ? doc.ratings : [];
  return list.filter(isRecord).map((r) => ({
    provider: str(r.provider),
    model: str(r.model),
    ratingMilli: typeof r.rating_milli === 'number' ? r.rating_milli : 0,
    contextLength: typeof r.context_length === 'number' ? r.context_length : 0,
    tools: r.tools === true,
    vision: r.vision === true,
    json: r.json === true,
    free: r.free === true,
  }));
}

function parseManifest(doc: Record<string, unknown>): ManifestRequest[] {
  const list = Array.isArray(doc.manifest) ? doc.manifest : [];
  return list.filter(isRecord).map((m) => ({
    id: str(m.id),
    api: (m.api === 'anthropic' ? 'anthropic' : 'openai') as ClientApi,
    stream: m.stream === true,
    weight: typeof m.weight === 'number' ? m.weight : 1,
    requires: isRecord(m.requires) ? (m.requires as Requirements) : {},
  }));
}

export function parseBrief(text: string): Brief {
  const docs = documentsIn(text);
  const providers = parseProviders(docNamed(docs, '/providers/v1'));
  const rows = parseRows(docNamed(docs, '/ratings/v1'));
  const manifest = parseManifest(docNamed(docs, '/workload/v1'));
  const rowByKey = new Map<string, ModelRow>();
  for (const row of rows) rowByKey.set(`${row.provider}/${row.model}`, row);
  return { providers, rows, rowByKey, manifest };
}
