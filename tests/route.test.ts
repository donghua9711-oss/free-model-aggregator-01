// Routing: candidate selection and the failover attempt loop.
//
// Candidates are models that are rated in the brief, marked free there,
// actually listed by a reachable provider, able to meet the request's
// requirements, and able to echo the request faithfully (an OpenAI JSON
// mode request can never go to an Anthropic-wire provider). They are
// tried best-rating-first; a refusal moves to the next candidate.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseBrief, type Brief } from '../lib/brief.ts';
import type { ProbeResult } from '../lib/catalog.ts';
import { extractRequest } from '../lib/echo.ts';
import { Router, type CallFn } from '../lib/route.ts';
import type { ProviderAnswer } from '../lib/wires.ts';

function loadBrief(): Brief {
  const path = `${process.env.HOME}/workspace/makefun/free-model-aggregator-01-kit/practice/trial-0/input/prompt.txt`;
  return parseBrief(readFileSync(path, 'utf8'));
}

/** Every provider reachable, listing exactly its rated models. */
function fullProbes(brief: Brief): ProbeResult[] {
  return brief.providers.map((provider) => ({
    provider,
    ok: true,
    models: brief.rows
      .filter((row) => row.provider === provider.id)
      .map((row) => ({ id: row.model, free: row.free, contextLength: row.contextLength })),
  }));
}

const ANSWER: ProviderAnswer = { text: 'ANSWER ECHO{}', inputTokens: 1, outputTokens: 2, stopReason: 'stop' };

function routerWith(brief: Brief, call: CallFn): Router {
  return new Router('http://env', brief, fullProbes(brief), call);
}

test('vision+tools: the gemini flash model leads, all candidates can serve', () => {
  const brief = loadBrief();
  const router = routerWith(brief, async () => ({ ok: true, answer: ANSWER }));
  const req = extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'x' }] });
  const candidates = router.candidatesFor(req, { tools: true, vision: true });
  assert.equal(candidates[0]?.provider.id, 'gemini');
  assert.equal(candidates[0]?.row.model, 'gemini-3.8-flash');
  assert.equal(candidates[0]?.row.ratingMilli, 1458);
  assert.ok(candidates.every((c) => c.row.free && c.row.tools && c.row.vision));
});

test('min_context 200000: only long-context models, best first', () => {
  const brief = loadBrief();
  const router = routerWith(brief, async () => ({ ok: true, answer: ANSWER }));
  const req = extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'x' }] });
  const candidates = router.candidatesFor(req, { min_context: 200000 });
  assert.ok(candidates.length >= 3);
  assert.ok(candidates.every((c) => c.row.contextLength >= 200000));
  assert.equal(candidates[0]?.row.ratingMilli, 1475);
});

test('a JSON-mode request never routes to an anthropic-wire provider', () => {
  const brief = loadBrief();
  const router = routerWith(brief, async () => ({ ok: true, answer: ANSWER }));
  const req = extractRequest('openai', {
    model: 'auto',
    response_format: { type: 'json_object' },
    messages: [{ role: 'user', content: 'x' }],
  });
  assert.equal(req.jsonMode, true);
  const candidates = router.candidatesFor(req, { json: true });
  assert.ok(candidates.length > 0);
  assert.ok(candidates.every((c) => c.provider.wire !== 'anthropic'));
  // zai's glm-4.7-flash (1470, json-capable) is excluded by wire, not by rating.
  assert.ok(!candidates.some((c) => c.provider.id === 'zai'));
});

test('the paid model is never a candidate', () => {
  const brief = loadBrief();
  const router = routerWith(brief, async () => ({ ok: true, answer: ANSWER }));
  const req = extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'x' }] });
  const candidates = router.candidatesFor(req, {});
  assert.ok(!candidates.some((c) => c.row.model === 'anthropic/claude-opus-5.5'));
  assert.ok(candidates.every((c) => c.row.free));
});

test('failover: a refusal moves to the next candidate, which serves', async () => {
  const brief = loadBrief();
  const seen: string[] = [];
  const router = routerWith(brief, async (target, _req, model) => {
    seen.push(`${target.baseUrl}/${model}`);
    if (seen.length === 1) return { ok: false, status: 429, detail: 'rate limited' };
    return { ok: true, answer: ANSWER };
  });
  const req = extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'x' }] });
  const served = await router.serve(req, {});
  assert.ok(served);
  assert.equal(seen.length, 2);
  assert.notEqual(seen[0], seen[1]);
  assert.equal(served.candidate.row.ratingMilli, 1470); // second best after the 1475 refusal
});

test('when every candidate refuses, serve returns null', async () => {
  const brief = loadBrief();
  const router = routerWith(brief, async () => ({ ok: false, status: 402, detail: 'payment required' }));
  const req = extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'x' }] });
  assert.equal(await router.serve(req, {}), null);
});

test('a provider whose probe failed contributes no candidates', () => {
  const brief = loadBrief();
  const probes = fullProbes(brief).map((p) => (p.provider.id === 'openrouter' ? { ...p, ok: false, models: [] } : p));
  const router = new Router('http://env', brief, probes, async () => ({ ok: true, answer: ANSWER }));
  const req = extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'x' }] });
  const candidates = router.candidatesFor(req, {});
  assert.ok(!candidates.some((c) => c.provider.id === 'openrouter'));
  assert.equal(candidates[0]?.row.ratingMilli, 1470);
});
