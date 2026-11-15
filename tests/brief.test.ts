// Brief parsing: the three documents inside the trial prompt, validated
// against the practice briefs the kit ships.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { meetsRequirements, parseBrief } from '../lib/brief.ts';

function trialBrief(n: number) {
  const path = `${process.env.HOME}/workspace/makefun/free-model-aggregator-01-kit/practice/trial-${n}/input/prompt.txt`;
  return parseBrief(readFileSync(path, 'utf8'));
}

test('trial-0 brief: ten providers, 26 rating rows, 20 manifest requests', () => {
  const brief = trialBrief(0);
  assert.equal(brief.providers.length, 10);
  assert.equal(brief.rows.length, 26);
  assert.equal(brief.manifest.length, 20);
});

test('provider entries carry wire, auth, key env, key and path', () => {
  const groq = trialBrief(0).providers.find((p) => p.id === 'groq');
  assert.ok(groq);
  assert.equal(groq.wire, 'openai');
  assert.equal(groq.auth, 'bearer');
  assert.equal(groq.keyEnv, 'GROQ_API_KEY');
  assert.equal(groq.path, '/p/groq');
  assert.equal(typeof groq.key, 'string');
  assert.notEqual(groq.key, '');
  const gemini = trialBrief(0).providers.find((p) => p.id === 'gemini');
  assert.equal(gemini?.wire, 'gemini');
  assert.equal(gemini?.auth, 'x-goog-api-key');
  const pollinations = trialBrief(0).providers.find((p) => p.id === 'pollinations');
  assert.equal(pollinations?.auth, 'none');
  assert.equal(pollinations?.keyEnv, '');
});

test('rating rows: the best free model; the paid model is not in the brief', () => {
  const brief = trialBrief(0);
  const glm = brief.rowByKey.get('openrouter/z-ai/glm-5.3-flash:free');
  assert.ok(glm);
  assert.equal(glm.ratingMilli, 1475);
  assert.equal(glm.free, true);
  assert.equal(glm.contextLength, 204800);
  // The brief's table lists free models only: the one paid model of the
  // sealed table (openrouter's claude-opus) has no row, so routing, which
  // works from rows, can never pick it.
  assert.equal(brief.rowByKey.get('openrouter/anthropic/claude-opus-5.5'), undefined);
  assert.ok(brief.rows.every((row) => row.free));
});

test('manifest r01 is a weight-2 openai request needing tools and vision', () => {
  const r01 = trialBrief(0).manifest[0];
  assert.ok(r01);
  assert.equal(r01.id, 'r01');
  assert.equal(r01.api, 'openai');
  assert.equal(r01.stream, false);
  assert.equal(r01.weight, 2);
  assert.deepEqual(r01.requires, { tools: true, vision: true });
});

test('meetsRequirements mirrors the harness: only true flags and min_context bind', () => {
  const brief = trialBrief(0);
  const scout = brief.rowByKey.get('cerebras/llama-4-scout-17b');
  assert.ok(scout);
  assert.equal(meetsRequirements(scout, { tools: true }), true);
  assert.equal(meetsRequirements(scout, { vision: true }), false);
  assert.equal(meetsRequirements(scout, { min_context: 200000 }), false);
  assert.equal(meetsRequirements(scout, {}), true);
  assert.equal(meetsRequirements(scout, { tools: false }), true);
});

test('a prompt without the three documents is rejected', () => {
  assert.throws(() => parseBrief('no documents here'), /providers\/v1/);
});
