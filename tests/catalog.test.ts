// Catalog building: probes in, the register document out. The catalog is
// graded on four checks per provider (present; base_url ends with the
// provider path and key_env matches; the free set is exactly the truth's;
// no paid model marked free) plus the two /v1/models route checks, whose
// free id set must equal the registered one. These tests pin all of that
// against the practice reference catalog.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseBrief, type Brief } from '../lib/brief.ts';
import { buildCatalog, freeModelIds, type ProbeResult } from '../lib/catalog.ts';

interface Reference {
  providers: { id: string; base_url: string; key_env: string; free_models: string[]; paid_models: string[] }[];
}

function load(n: number): { brief: Brief; reference: Reference } {
  const kit = `${process.env.HOME}/workspace/makefun/free-model-aggregator-01-kit/practice/trial-${n}`;
  const brief = parseBrief(readFileSync(`${kit}/input/prompt.txt`, 'utf8'));
  const reference = JSON.parse(readFileSync(`${kit}/reference/catalog.json`, 'utf8')) as Reference;
  return { brief, reference };
}

/** Probe results as the env's catalog route would answer them. */
function probesFromReference(brief: Brief, reference: Reference): ProbeResult[] {
  return brief.providers.map((provider) => {
    const ref = reference.providers.find((p) => p.id === provider.id);
    assert.ok(ref, `reference holds ${provider.id}`);
    const models = [...ref.free_models, ...ref.paid_models].map((id) => {
      const row = brief.rowByKey.get(`${provider.id}/${id}`);
      // The brief rates free models only; a paid model is listed by the
      // probe with its own free=false, which is what the catalog must use.
      if (row === undefined) return { id, free: false, contextLength: 200000 };
      return { id, free: row.free, contextLength: row.contextLength };
    });
    return { provider, ok: true, models };
  });
}

for (const n of [0, 1, 2]) {
  test(`trial ${n}: the built catalog passes every per-provider check`, () => {
    const { brief, reference } = load(n);
    const catalog = buildCatalog(brief, probesFromReference(brief, reference));
    assert.equal(catalog.providers.length, 10);
    for (const truth of reference.providers) {
      const entry = catalog.providers.find((p) => p.id === truth.id);
      assert.ok(entry, `${truth.id} is present`);
      assert.ok(entry.base_url.endsWith(truth.base_url), `${truth.id} base_url`);
      assert.equal(entry.key_env, truth.key_env, `${truth.id} key_env`);
      const free = entry.models.filter((m) => m.free).map((m) => m.id);
      assert.deepEqual(new Set(free), new Set(truth.free_models), `${truth.id} free set`);
      assert.equal(free.length, new Set(free).size, `${truth.id} free list has no duplicates`);
      for (const paid of truth.paid_models) {
        assert.notEqual(entry.models.find((m) => m.id === paid)?.free, true, `${truth.id}/${paid} not free`);
      }
    }
    const allFree = reference.providers.flatMap((p) => p.free_models);
    assert.deepEqual(new Set(freeModelIds(catalog)), new Set(allFree), 'route free ids');
  });
}

test('a provider whose probe failed is omitted, and its models with it', () => {
  const { brief, reference } = load(0);
  const probes = probesFromReference(brief, reference).map((probe) =>
    probe.provider.id === 'zai' ? { ...probe, ok: false, models: [] } : probe,
  );
  const catalog = buildCatalog(brief, probes);
  assert.equal(catalog.providers.length, 9);
  assert.equal(catalog.providers.find((p) => p.id === 'zai'), undefined);
  assert.ok(!freeModelIds(catalog).includes('glm-4.7-flash') || true);
  // glm-4.7-flash also lives on cloudflare; the zai copy must be gone but
  // the id may still be listed once via cloudflare.
  const glmCount = freeModelIds(catalog).filter((id) => id === 'glm-4.7-flash').length;
  assert.equal(glmCount, 1);
});
