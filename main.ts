// Free Model Aggregator — the entrypoint.
//
// One server in front of ten free-tier providers. It reads the trial
// brief, probes every provider's catalog, registers with the env, then
// serves the client's twenty requests one at a time: each is routed to
// the best free model that can serve it (lib/route.ts), the provider's
// answer is buffered whole and rendered back on the client's own wire
// (lib/render.ts). When the env says the workload is done, it exits 0.
//
//   node main.ts          run a trial (brief at $MAKEFUN_PROMPT_FILE,
//                         env at $MAKEFUN_ENV_URL)
//   node main.ts --list   print the catalog document and exit
//
// Node 22 standard library only; TypeScript runs directly.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import { parseBrief, type Brief, type Requirements } from './lib/brief.ts';
import { buildCatalog, probeAll, type Catalog, type ProbeResult } from './lib/catalog.ts';
import { extractRequest, isRecord, type CanonicalRequest, type ClientApi } from './lib/echo.ts';
import {
  renderAnthropicJson,
  renderAnthropicSse,
  renderModelsList,
  renderOpenAIJson,
  renderOpenAISse,
} from './lib/render.ts';
import { Router } from './lib/route.ts';

const PROTOCOL_VERSION = 1;
const WORK_POLL_MS = 100;
const MAX_BODY_BYTES = 4 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  console.error(`aggregator: ${message}`);
}

interface Loaded {
  brief: Brief | null;
  probes: ProbeResult[];
  catalog: Catalog;
}

async function load(): Promise<Loaded> {
  if (PROMPT_FILE === '') return { brief: null, probes: [], catalog: { providers: [] } };
  const brief = parseBrief(readFileSync(PROMPT_FILE, 'utf8'));
  const probes = ENV_URL === '' ? [] : await probeAll(ENV_URL, brief);
  return { brief, probes, catalog: buildCatalog(brief, probes) };
}

// ---------- request handling ----------

interface Reply {
  status: number;
  contentType: string;
  body: string;
}

function errorReply(api: ClientApi, status: number, message: string): Reply {
  const body =
    api === 'openai'
      ? JSON.stringify({ error: { message, type: 'api_error', code: 'aggregator_error' } })
      : JSON.stringify({ type: 'error', error: { type: 'api_error', message } });
  return { status, contentType: 'application/json', body };
}

/** Requirements from the request itself, when the manifest has run out. */
function derivedRequirements(req: CanonicalRequest): Requirements {
  const requires: Requirements = {};
  if (req.tools.length > 0) requires.tools = true;
  if (req.jsonMode) requires.json = true;
  const hasImage = req.messages.some((m) => m.blocks.some((b) => b.kind === 'imageData' || b.kind === 'imageUrl'));
  if (hasImage) requires.vision = true;
  return requires;
}

class Handler {
  #brief: Brief | null;
  #catalog: Catalog;
  #router: Router | null;
  #next = 0;
  #seq = 0;

  constructor(loaded: Loaded) {
    this.#brief = loaded.brief;
    this.#catalog = loaded.catalog;
    this.#router =
      loaded.brief !== null && ENV_URL !== '' ? new Router(ENV_URL, loaded.brief, loaded.probes) : null;
  }

  models(api: ClientApi): Reply {
    return { status: 200, contentType: 'application/json', body: renderModelsList(this.#catalog, api) };
  }

  async complete(api: ClientApi, raw: string): Promise<Reply> {
    // Requests arrive in manifest order; take this request's entry by
    // arrival, before anything can fail, so alignment can never shift.
    const entry = this.#brief?.manifest[this.#next];
    this.#next += 1;
    let body: unknown;
    try {
      body = JSON.parse(raw);
    } catch {
      return errorReply(api, 400, 'the request body is not JSON');
    }
    if (!isRecord(body)) return errorReply(api, 400, 'the request body is not one JSON object');
    if (this.#router === null || this.#brief === null) {
      return errorReply(api, 502, 'the aggregator has no provider directory');
    }
    const req = extractRequest(api, body);
    const requires = entry !== undefined ? entry.requires : derivedRequirements(req);
    const served = await this.#router.serve(req, requires);
    if (served === null) return errorReply(api, 502, 'every free provider refused the request');
    this.#seq += 1;
    const { answer, candidate } = served;
    const stream = body.stream === true;
    const text = stream
      ? api === 'openai'
        ? renderOpenAISse(answer, candidate.row.model, this.#seq)
        : renderAnthropicSse(answer, candidate.row.model, this.#seq)
      : api === 'openai'
        ? renderOpenAIJson(answer, candidate.row.model, this.#seq)
        : renderAnthropicJson(answer, candidate.row.model, this.#seq);
    return { status: 200, contentType: stream ? 'text/event-stream; charset=utf-8' : 'application/json', body: text };
  }
}

// ---------- HTTP server ----------

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function surfaceOf(req: IncomingMessage): ClientApi {
  return req.headers['x-api-key'] !== undefined ? 'anthropic' : 'openai';
}

async function dispatch(handler: Handler, req: IncomingMessage): Promise<Reply> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (req.method === 'GET' && url.pathname === '/v1/models') return handler.models(surfaceOf(req));
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    return handler.complete('openai', await readBody(req));
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    return handler.complete('anthropic', await readBody(req));
  }
  return errorReply(surfaceOf(req), 404, 'no such route');
}

function send(res: ServerResponse, reply: Reply): void {
  res.writeHead(reply.status, { 'content-type': reply.contentType });
  res.end(reply.body);
}

// ---------- trial lifecycle ----------

async function register(port: number, catalog: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  const reply: unknown = await res.json().catch(() => null);
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the workload holds ${reply.requests} requests`);
  }
}

/** Poll GET /work until the env says done; then exit 0. */
function pollWork(): void {
  const tick = async (): Promise<void> => {
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
      const reply: unknown = await res.json().catch(() => null);
      if (isRecord(reply) && reply.done === true) {
        log('the workload is done; exiting');
        process.exit(0);
      }
    } catch {
      // The env is momentarily unreachable; keep polling.
    }
    setTimeout(() => void tick(), WORK_POLL_MS);
  };
  void tick();
}

async function main(): Promise<void> {
  const loaded = await load();
  if (process.argv.includes('--list')) {
    // The same object the register body carries, byte for byte.
    process.stdout.write(`${JSON.stringify(loaded.catalog)}\n`);
    return;
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');

  const handler = new Handler(loaded);
  // Requests are processed one at a time: the client is sequential, and a
  // single in-flight provider call per provider can never trip a
  // provider's concurrency limit.
  let chain: Promise<unknown> = Promise.resolve();
  const server = createServer((req, res) => {
    const run = chain.then(() => dispatch(handler, req));
    chain = run.catch(() => undefined);
    run.then(
      (reply) => send(res, reply),
      (err: Error) => send(res, errorReply(surfaceOf(req), 500, `internal error: ${err.message}`)),
    );
  });
  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no TCP address');
  log(
    `listening on 0.0.0.0:${address.port} with ${loaded.catalog.providers.length} providers ` +
      `and ${loaded.catalog.providers.reduce((n, p) => n + p.models.filter((m) => m.free).length, 0)} free models`,
  );
  await register(address.port, loaded.catalog);
  pollWork();
}

main().catch((err: Error) => {
  console.error(`aggregator: ${err.message}`);
  process.exit(1);
});
