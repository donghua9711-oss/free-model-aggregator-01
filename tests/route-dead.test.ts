// Dead-provider memory: three consecutive refusals (402/429/5xx) prove a
// provider is spent or out for the rest of the trial — the generator
// draws at most two single-call failures per provider, and budget and
// outage refusals never recover. The router then stops spending calls
// on it (score ties break on upstream calls), unless it is the only
// candidate left.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseBrief, type Brief } from '../lib/brief.ts';
import type { ProbeResult } from '../lib/catalog.ts';
import { extractRequest, type CanonicalRequest } from '../lib/echo.ts';
import { Router, type CallFn } from '../lib/route.ts';
import type { ProviderAnswer } from '../lib/wires.ts';

const ANSWER: ProviderAnswer = { text: 'ANSWER ECHO{}', inputTokens: 1, outputTokens: 2, stopReason: 'stop' };

/** A two-provider brief: alpha (rating 1500) and beta (rating 1400). */
function syntheticBrief(): Brief {
  const providers = {
    schema: 'free-model-aggregator-01/providers/v1',
    providers: [
      { id: 'alpha', name: 'Alpha', wire: 'openai', auth: 'none', key_env: '', key: '', path: '/p/alpha' },
      { id: 'beta', name: 'Beta', wire: 'openai', auth: 'none', key_env: '', key: '', path: '/p/beta' },
    ],
  };
  const ratings = {
    schema: 'free-model-aggregator-01/ratings/v1',
    ratings: [
      { provider: 'alpha', model: 'a-max', rating_milli: 1500, context_length: 100000, tools: true, vision: true, json: true, free: true },
      { provider: 'beta', model: 'b-max', rating_milli: 1400, context_length: 100000, tools: true, vision: true, json: true, free: true },
    ],
    best: [],
  };
  const workload = { schema: 'free-model-aggregator-01/workload/v1', manifest: [], bodies: {} };
  return parseBrief([providers, ratings, workload].map((d) => JSON.stringify(d)).join('\n'));
}

function probes(brief: Brief): ProbeResult[] {
  return brief.providers.map((provider) => ({
    provider,
    ok: true,
    models: brief.rows
      .filter((row) => row.provider === provider.id)
      .map((row) => ({ id: row.model, free: row.free, contextLength: row.contextLength })),
  }));
}

function request(): CanonicalRequest {
  return extractRequest('openai', { model: 'auto', messages: [{ role: 'user', content: 'x' }] });
}

test('after three consecutive refusals a provider is skipped', async () => {
  const brief = syntheticBrief();
  const called: string[] = [];
  const call: CallFn = async (target, _req, model) => {
    called.push(`${target.baseUrl}/${model}`);
    if (target.baseUrl.endsWith('/p/alpha')) return { ok: false, status: 429, detail: 'limited' };
    return { ok: true, answer: ANSWER };
  };
  const router = new Router('http://env', brief, probes(brief), call);
  for (let i = 0; i < 3; i++) {
    const served = await router.serve(request(), {});
    assert.equal(served?.candidate.provider.id, 'beta');
  }
  assert.equal(called.filter((c) => c.includes('/p/alpha')).length, 3);
  called.length = 0;
  const fourth = await router.serve(request(), {});
  assert.equal(fourth?.candidate.provider.id, 'beta');
  assert.deepEqual(called, ['http://env/p/beta/b-max'], 'alpha is not called again');
});

test('a success resets the streak: two refusals, a success, two more refusals — still tried', async () => {
  const brief = syntheticBrief();
  let alphaAttempts = 0;
  const call: CallFn = async (target) => {
    if (target.baseUrl.endsWith('/p/alpha')) {
      alphaAttempts += 1;
      // Refuse attempts 1, 2, then succeed on 3, refuse 4 and 5.
      if (alphaAttempts === 3) return { ok: true, answer: ANSWER };
      return { ok: false, status: 402, detail: 'payment required' };
    }
    return { ok: true, answer: ANSWER };
  };
  const router = new Router('http://env', brief, probes(brief), call);
  await router.serve(request(), {}); // alpha refuses (1), beta serves
  await router.serve(request(), {}); // alpha refuses (2), beta serves
  const third = await router.serve(request(), {}); // alpha serves (3)
  assert.equal(third?.candidate.provider.id, 'alpha');
  await router.serve(request(), {}); // alpha refuses (1 of new streak), beta serves
  const fifth = await router.serve(request(), {}); // alpha refuses (2), beta serves
  assert.equal(fifth?.candidate.provider.id, 'beta');
  const sixth = await router.serve(request(), {}); // alpha still tried: refuses (3), beta serves
  assert.equal(sixth?.candidate.provider.id, 'beta');
  assert.equal(alphaAttempts, 6, 'alpha was attempted on every request');
});

test('network errors (status 0) do not build a dead streak', async () => {
  const brief = syntheticBrief();
  let alphaAttempts = 0;
  const call: CallFn = async (target) => {
    if (target.baseUrl.endsWith('/p/alpha')) {
      alphaAttempts += 1;
      return { ok: false, status: 0, detail: 'network error' };
    }
    return { ok: true, answer: ANSWER };
  };
  const router = new Router('http://env', brief, probes(brief), call);
  for (let i = 0; i < 4; i++) await router.serve(request(), {});
  assert.equal(alphaAttempts, 4, 'alpha is still attempted after network errors');
});

test('when the dead provider is the only candidate, it is still tried', async () => {
  const brief = syntheticBrief();
  const onlyAlpha = probes(brief).map((p) => (p.provider.id === 'beta' ? { ...p, ok: false, models: [] } : p));
  let alphaAttempts = 0;
  const call: CallFn = async () => {
    alphaAttempts += 1;
    return { ok: false, status: 429, detail: 'limited' };
  };
  const router = new Router('http://env', brief, onlyAlpha, call);
  for (let i = 0; i < 4; i++) assert.equal(await router.serve(request(), {}), null);
  assert.equal(alphaAttempts, 4, 'the fallback keeps trying the only candidate');
});
