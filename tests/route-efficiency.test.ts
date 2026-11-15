// Call efficiency: refusals are read, risk is priced, quality is fenced.
//
// The score rounds and the ranking falls to upstream calls, so routing
// spends as few calls as the failure draw allows: a refusal that proves
// a provider spent or out kills it on the spot (no three-strike wait),
// a refusal that spends a drawn failure ordinal makes the provider
// safer and retryable at once, the next call's failure probability —
// from the competition's published budget ranges — is priced against
// the rating, and a quality ledger fences the pricing so it can never
// trade the rounded score away.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseBrief, type Brief, type ManifestRequest } from '../lib/brief.ts';
import type { ProbeResult } from '../lib/catalog.ts';
import { extractRequest, type CanonicalRequest } from '../lib/echo.ts';
import { Router, type CallFn } from '../lib/route.ts';
import type { ProviderAnswer } from '../lib/wires.ts';

const ANSWER: ProviderAnswer = { text: 'ANSWER ECHO{}', inputTokens: 1, outputTokens: 2, stopReason: 'stop' };

interface ModelSpec {
  provider: string;
  model: string;
  rating: number;
}

/** A brief over the given providers/models, with an optional manifest+best. */
function syntheticBrief(models: ModelSpec[], manifest: ManifestRequest[] = [], best: Record<string, number> = {}): Brief {
  const providerIds = [...new Set(models.map((m) => m.provider))];
  const providers = {
    schema: 'free-model-aggregator-01/providers/v1',
    providers: providerIds.map((id) => ({
      id,
      name: id,
      wire: 'openai',
      auth: 'none',
      key_env: '',
      key: '',
      path: `/p/${id}`,
    })),
  };
  const ratings = {
    schema: 'free-model-aggregator-01/ratings/v1',
    ratings: models.map((m) => ({
      provider: m.provider,
      model: m.model,
      rating_milli: m.rating,
      context_length: 100000,
      tools: true,
      vision: true,
      json: true,
      free: true,
    })),
    best: Object.entries(best).map(([id, rating_milli]) => ({ id, rating_milli })),
  };
  const workload = { schema: 'free-model-aggregator-01/workload/v1', manifest, bodies: {} };
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

const TWO_MODELS: ModelSpec[] = [
  { provider: 'alpha', model: 'a-max', rating: 1500 },
  { provider: 'beta', model: 'b-max', rating: 1400 },
];

test('a spent-budget refusal kills the provider on the spot', async () => {
  const brief = syntheticBrief(TWO_MODELS);
  const called: string[] = [];
  const call: CallFn = async (target) => {
    called.push(target.baseUrl);
    if (target.baseUrl.endsWith('/p/alpha')) {
      return { ok: false, status: 429, detail: `{"error":{"message":"alpha's free-tier budget is spent"}}` };
    }
    return { ok: true, answer: ANSWER };
  };
  const router = new Router('http://env', brief, probes(brief), call);
  const first = await router.serve(request(), {});
  assert.equal(first?.candidate.provider.id, 'beta');
  assert.equal(called.filter((c) => c.endsWith('/p/alpha')).length, 1, 'one proving call, not three');
  called.length = 0;
  const second = await router.serve(request(), {});
  assert.equal(second?.candidate.provider.id, 'beta');
  assert.deepEqual(called, ['http://env/p/beta'], 'alpha is never called again');
});

test('an outage refusal kills the provider on the spot', async () => {
  const brief = syntheticBrief(TWO_MODELS);
  let alphaCalls = 0;
  const call: CallFn = async (target) => {
    if (target.baseUrl.endsWith('/p/alpha')) {
      alphaCalls += 1;
      return { ok: false, status: 503, detail: `{"error":{"message":"alpha is out"}}` };
    }
    return { ok: true, answer: ANSWER };
  };
  const router = new Router('http://env', brief, probes(brief), call);
  await router.serve(request(), {});
  await router.serve(request(), {});
  assert.equal(alphaCalls, 1, 'the outage message alone proved alpha dead');
});

test('a spent failure ordinal makes the provider retryable at once', async () => {
  const brief = syntheticBrief(TWO_MODELS);
  let alphaCalls = 0;
  let betaCalls = 0;
  const call: CallFn = async (target) => {
    if (target.baseUrl.endsWith('/p/alpha')) {
      alphaCalls += 1;
      if (alphaCalls === 1) {
        return { ok: false, status: 429, detail: `{"error":{"message":"alpha refused this call"}}` };
      }
      return { ok: true, answer: ANSWER };
    }
    betaCalls += 1;
    return { ok: true, answer: ANSWER };
  };
  const router = new Router('http://env', brief, probes(brief), call);
  const served = await router.serve(request(), {});
  assert.equal(served?.candidate.provider.id, 'alpha', 'the same provider is retried and serves');
  assert.equal(alphaCalls, 2);
  assert.equal(betaCalls, 0, 'no failover call was spent');
});

test('risk pricing steers around a small-budget provider’s second call', async () => {
  // zai's drawn budget is 4-5, so its second call is the likeliest in
  // the trial to hit a drawn failure ordinal; kilol's is 12-16. Once
  // both have served once, the safer kilol call wins despite the
  // 50-milli rating gap.
  const brief = syntheticBrief([
    { provider: 'zai', model: 'z-max', rating: 1470 },
    { provider: 'kilol', model: 'k-max', rating: 1420 },
  ]);
  const called: string[] = [];
  const call: CallFn = async (target) => {
    called.push(target.baseUrl);
    return { ok: true, answer: ANSWER };
  };
  const router = new Router('http://env', brief, probes(brief), call);
  const first = await router.serve(request(), {});
  assert.equal(first?.candidate.provider.id, 'zai', 'first calls are risk-free: best rating leads');
  called.length = 0;
  const second = await router.serve(request(), {});
  assert.equal(second?.candidate.provider.id, 'kilol');
  assert.deepEqual(called, ['http://env/p/kilol'], 'zai’s dangerous second call was not spent');
});

test('the quality guard refuses a pick that would spend the rounded score', async () => {
  // Two requests, best 1500 each. After alpha serves the first at 1500,
  // serving the second at beta's 1400 projects quality 96.7 — under the
  // 98 floor — so the guard keeps the request on alpha even though
  // alpha's second call carries priced risk and beta's first is free.
  const manifest: ManifestRequest[] = [
    { id: 'r01', api: 'openai', stream: false, weight: 1, requires: {} },
    { id: 'r02', api: 'openai', stream: false, weight: 1, requires: {} },
  ];
  const brief = syntheticBrief(TWO_MODELS, manifest, { r01: 1500, r02: 1500 });
  const call: CallFn = async () => ({ ok: true, answer: ANSWER });
  const router = new Router('http://env', brief, probes(brief), call);
  const first = await router.serve(request(), {}, brief.manifest[0]);
  assert.equal(first?.candidate.provider.id, 'alpha');
  const second = await router.serve(request(), {}, brief.manifest[1]);
  assert.equal(second?.candidate.provider.id, 'alpha', 'the guard overruled the risk price');
});
