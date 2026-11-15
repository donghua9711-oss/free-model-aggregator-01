// Routing: who can serve a request, in what order, and the failover loop.
//
// A candidate is a model that is rated in the brief and free there, is
// actually listed by a provider whose probe answered, meets the request's
// requirements, and can echo the request faithfully — an OpenAI JSON-mode
// request is never sent to an Anthropic-wire provider, whose parser
// cannot echo json_mode.
//
// Which candidate is tried is a price, not a sort. The score rounds, so
// at the top the ranking falls to upstream calls: every wasted call is
// pure loss. A provider's next call fails with a probability the trial's
// own rules let us estimate (see #failureRisk), and each attempt picks
// the candidate with the best rating minus that risk priced in milli —
// the first call to any provider is free of risk, a provider whose two
// drawn failure ordinals are both spent is safe again, and so on.
//
// Refusals are read, not just counted. The harness says why a call was
// refused, and the three answers mean different things: a spent budget
// or an outage never recovers (the provider is dead on the spot, no
// three-strike wait), while a single-call refusal spends one of the at
// most two failure ordinals the generator drew for that provider — the
// provider is safer afterwards, and may be retried at once. Refusals
// without a readable reason keep the older memory: three in a row
// (402/429/5xx) mark the provider dead; network errors prove nothing.
//
// A quality guard fences the risk pricing. The brief carries the
// scorer's best rating for every manifest request, so the router keeps
// the weighted ledger and only lets a pick spend quality while the
// projected final quality (this pick, plus every later request at its
// best) still floors to 99 — or, when that is already out of reach, to
// 98, the lowest quality that still rounds the total score to 99.

import {
  meetsRequirements,
  type Brief,
  type ManifestRequest,
  type ModelRow,
  type ProviderEntry,
  type Requirements,
} from './brief.ts';
import type { ProbeResult } from './catalog.ts';
import type { CanonicalRequest } from './echo.ts';
import { callProvider, type CallOutcome, type ProviderAnswer, type ProviderTarget } from './wires.ts';

export type CallFn = (target: ProviderTarget, req: CanonicalRequest, model: string) => Promise<CallOutcome>;

export interface Candidate {
  provider: ProviderEntry;
  row: ModelRow;
  target: ProviderTarget;
}

export interface Served {
  answer: ProviderAnswer;
  candidate: Candidate;
}

/**
 * The free-tier budget ranges the competition's generator draws from
 * (kit RULES: calls_min..calls_max per provider). Failure ordinals are
 * drawn inside the same range, so the range is also the risk prior for
 * providers whose draw we cannot see. Unknown providers get a wide,
 * flat prior — production briefs name only the ten known ids.
 */
const BUDGET_RANGES = new Map<string, [number, number]>([
  ['groq', [9, 12]],
  ['cerebras', [5, 6]],
  ['gemini', [6, 8]],
  ['zai', [4, 5]],
  ['openrouter', [10, 12]],
  ['sambanova', [4, 5]],
  ['pollinations', [10, 14]],
  ['cloudflare', [7, 8]],
  ['ovh', [5, 6]],
  ['kilol', [12, 16]],
]);
const DEFAULT_RANGE: [number, number] = [10, 14];

/** Milli of rating one unit of failure probability costs a candidate. */
const RISK_WEIGHT = 1200;
/** Projected-quality floors the guard defends, best first. */
const QUALITY_FLOORS = [0.9905, 0.9805];
/** Attempts per provider within one request; bounds the failover loop. */
const MAX_PROVIDER_ATTEMPTS_PER_REQUEST = 4;
/** Attempts per request overall; a safety net, never expected to bind. */
const MAX_ATTEMPTS_PER_REQUEST = 48;

/** What a refusal's detail says happened, when it says anything readable. */
type RefusalKind = 'dead' | 'mine' | 'unknown';

function classifyRefusal(detail: string): RefusalKind {
  if (detail.includes('free-tier budget is spent') || /\bis out\b/.test(detail)) return 'dead';
  if (detail.includes('refused this call')) return 'mine';
  return 'unknown';
}

export class Router {
  #envUrl: string;
  #brief: Brief;
  #probes: ProbeResult[];
  #call: CallFn;
  /** Calls made per provider id — the harness's own per-provider ordinal. */
  #attempts = new Map<string, number>();
  /** Consecutive unreadable refusals per provider id (the older memory). */
  #refusalStreaks = new Map<string, number>();
  /** Providers a refusal proved spent or out for the rest of the trial. */
  #dead = new Set<string>();
  /** Ordinals per provider id already known to be drawn failure ordinals. */
  #mines = new Map<string, Set<number>>();
  /** Quality ledger: weighted milli earned, and the manifest cursor. */
  #earned = 0;
  #entryIndex = 0;
  #tops: number[];
  #bound: number;
  #futureBest: number[];

  constructor(envUrl: string, brief: Brief, probes: ProbeResult[], call: CallFn = callProvider) {
    this.#envUrl = envUrl;
    this.#brief = brief;
    this.#probes = probes;
    this.#call = call;
    // The quality ledger: each manifest request's best rating (the
    // scorer's own table when the brief carries it), its weight, and
    // suffix sums so a pick can be priced against what is still to come.
    this.#tops = brief.manifest.map((entry) => this.#topFor(entry));
    this.#bound = brief.manifest.reduce((sum, entry, i) => sum + entry.weight * (this.#tops[i] ?? 0), 0);
    this.#futureBest = new Array<number>(brief.manifest.length + 1).fill(0);
    for (let i = brief.manifest.length - 1; i >= 0; i--) {
      this.#futureBest[i] = (this.#futureBest[i + 1] ?? 0) + brief.manifest[i]!.weight * (this.#tops[i] ?? 0);
    }
  }

  #topFor(entry: ManifestRequest): number {
    const best = this.#brief.bestById.get(entry.id);
    if (best !== undefined) return best;
    let top = 0;
    for (const row of this.#brief.rows) {
      if (row.free && meetsRequirements(row, entry.requires) && row.ratingMilli > top) top = row.ratingMilli;
    }
    return top;
  }

  #isDead(id: string): boolean {
    return this.#dead.has(id) || (this.#refusalStreaks.get(id) ?? 0) >= 3;
  }

  /**
   * The probability that provider id's next call is refused, from the
   * trial's published draw rules: the first call never fails; failure
   * ordinals (one or two, expected 1.5) hide in [2, budget-1], with one
   * provider in the trial forced to fail its second call; past the low
   * end of the budget range the budget itself may run out. A provider
   * whose drawn ordinals are both spent is safe until its budget tail.
   */
  #failureRisk(id: string): number {
    if (this.#isDead(id)) return 1;
    const next = (this.#attempts.get(id) ?? 0) + 1;
    if (next === 1) return 0;
    const [callsMin, callsMax] = BUDGET_RANGES.get(id) ?? DEFAULT_RANGE;
    const seen = this.#mines.get(id)?.size ?? 0;
    let risk: number;
    if (seen >= 2) {
      risk = 0;
    } else {
      const remainingFails = seen === 1 ? 0.5 : 1.5;
      const budgetMean = (callsMin + callsMax) / 2;
      const remainingOrdinals = Math.max(1, budgetMean - next + 1);
      risk = remainingFails / remainingOrdinals;
      // One provider in ten is designated to fail its second call.
      if (next === 2 && seen === 0) risk = 0.1 + 0.9 * risk;
    }
    if (next > callsMin) risk = Math.max(risk, 0.3 + 0.08 * (next - callsMin));
    return Math.min(risk, 0.97);
  }

  candidatesFor(req: CanonicalRequest, requires: Requirements): Candidate[] {
    const candidates: Candidate[] = [];
    for (const probe of this.#probes) {
      if (!probe.ok) continue;
      const provider = probe.provider;
      // The Anthropic wire cannot echo JSON mode; never route such a
      // request there, whatever the model's rating.
      if (req.jsonMode && provider.wire === 'anthropic') continue;
      for (const listed of probe.models) {
        if (!listed.free) continue;
        const row = this.#brief.rowByKey.get(`${provider.id}/${listed.id}`);
        if (row === undefined || !row.free) continue;
        if (!meetsRequirements(row, requires)) continue;
        candidates.push({
          provider,
          row,
          target: {
            wire: provider.wire,
            auth: provider.auth,
            key: provider.key,
            baseUrl: `${this.#envUrl}${provider.path}`,
          },
        });
      }
    }
    const attempts = (id: string) => this.#attempts.get(id) ?? 0;
    // Plain code-unit order for the tie-breaks: localeCompare can vary
    // with the runtime's ICU data, and routing must be replay-identical.
    const byText = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
    candidates.sort(
      (a, b) =>
        b.row.ratingMilli - a.row.ratingMilli ||
        attempts(a.provider.id) - attempts(b.provider.id) ||
        byText(a.provider.id, b.provider.id) ||
        byText(a.row.model, b.row.model),
    );
    // Dead-provider memory: a provider proved spent or out — by its own
    // refusal message, or by three unreadable refusals in a row — stops
    // costing calls, unless it is all that is left.
    const live = candidates.filter((c) => !this.#isDead(c.provider.id));
    return live.length > 0 ? live : candidates;
  }

  /**
   * Align the quality ledger with the manifest entry being served and
   * return the guard context, or null when no entry is supplied (unit
   * tests) or the entry is not in the manifest. Requests arrive in
   * manifest order; an entry found ahead of the cursor resyncs it (the
   * skipped entries simply never earn).
   */
  #guardFor(entry: ManifestRequest | undefined): { weight: number; top: number; future: number } | null {
    if (entry === undefined || this.#bound === 0) return null;
    let i = this.#entryIndex;
    while (i < this.#brief.manifest.length && this.#brief.manifest[i]!.id !== entry.id) i++;
    if (i >= this.#brief.manifest.length) return null;
    this.#entryIndex = i;
    return { weight: entry.weight, top: this.#tops[i] ?? 0, future: this.#futureBest[i + 1] ?? 0 };
  }

  /** Try candidates, best risk-priced first, until one answers. */
  async serve(
    req: CanonicalRequest,
    requires: Requirements,
    entry?: ManifestRequest,
  ): Promise<Served | null> {
    const guard = this.#guardFor(entry);
    const attemptsHere = new Map<string, number>();
    // Providers that refused this request for any reason other than a
    // spent failure ordinal are not retried within the request: a dead
    // provider stays dead, and an unreadable refusal has told us nothing
    // that makes an immediate retry wiser than the next candidate.
    const blockedHere = new Set<string>();
    let attempts = 0;
    while (attempts < MAX_ATTEMPTS_PER_REQUEST) {
      let pool = this.candidatesFor(req, requires).filter(
        (c) =>
          (attemptsHere.get(c.provider.id) ?? 0) < MAX_PROVIDER_ATTEMPTS_PER_REQUEST &&
          !blockedHere.has(c.provider.id),
      );
      if (pool.length === 0) return null;
      // The quality guard: keep only the picks that hold the projected
      // final quality at the best floor still within reach.
      if (guard !== null) {
        for (const floor of QUALITY_FLOORS) {
          const kept = pool.filter(
            (c) =>
              (this.#earned + guard.weight * Math.min(c.row.ratingMilli, guard.top) + guard.future) /
                this.#bound >=
              floor,
          );
          if (kept.length > 0) {
            pool = kept;
            break;
          }
        }
      }
      // Best risk-priced candidate: rating minus priced failure risk.
      // The pool keeps candidatesFor's order (rating, attempts, ids), so
      // the first candidate to reach the best score wins the tie.
      let pick = pool[0]!;
      let pickScore = -Infinity;
      for (const candidate of pool) {
        const score = candidate.row.ratingMilli - RISK_WEIGHT * this.#failureRisk(candidate.provider.id);
        if (score > pickScore) {
          pick = candidate;
          pickScore = score;
        }
      }
      const id = pick.provider.id;
      const ordinal = (this.#attempts.get(id) ?? 0) + 1;
      this.#attempts.set(id, ordinal);
      attemptsHere.set(id, (attemptsHere.get(id) ?? 0) + 1);
      attempts += 1;
      const outcome = await this.#call(pick.target, req, pick.row.model);
      if (outcome.ok) {
        this.#refusalStreaks.set(id, 0);
        if (guard !== null) {
          this.#earned += guard.weight * Math.min(pick.row.ratingMilli, guard.top);
          this.#entryIndex += 1;
        }
        return { answer: outcome.answer, candidate: pick };
      }
      const kind = classifyRefusal(outcome.detail);
      if (kind === 'dead') {
        // Spent or out: proved by the refusal itself, no streak needed.
        this.#dead.add(id);
        blockedHere.add(id);
      } else if (kind === 'mine') {
        // A drawn failure ordinal, now spent: the provider is safer
        // than before the call and stays in this request's pool.
        let spent = this.#mines.get(id);
        if (spent === undefined) {
          spent = new Set<number>();
          this.#mines.set(id, spent);
        }
        spent.add(ordinal);
      } else {
        blockedHere.add(id);
        // Only a real refusal counts toward the streak (the harness's
        // own failed-call statuses); a network error proves nothing.
        if (outcome.status === 402 || outcome.status === 429 || outcome.status >= 500) {
          this.#refusalStreaks.set(id, (this.#refusalStreaks.get(id) ?? 0) + 1);
        }
      }
    }
    return null;
  }
}
