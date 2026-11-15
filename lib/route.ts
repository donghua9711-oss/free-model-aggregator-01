// Routing: who can serve a request, in what order, and the failover loop.
//
// A candidate is a model that is rated in the brief and free there, is
// actually listed by a provider whose probe answered, meets the request's
// requirements, and can echo the request faithfully — an OpenAI JSON-mode
// request is never sent to an Anthropic-wire provider, whose parser
// cannot echo json_mode. Candidates are tried best-rating first; ties
// spread across providers by attempts made. A refusal (402/429/5xx or a
// network failure) moves to the next candidate; the first success is the
// request's one successful provider call.

import {
  meetsRequirements,
  type Brief,
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

export class Router {
  #envUrl: string;
  #brief: Brief;
  #probes: ProbeResult[];
  #call: CallFn;
  #attempts = new Map<string, number>();
  #refusalStreaks = new Map<string, number>();

  constructor(envUrl: string, brief: Brief, probes: ProbeResult[], call: CallFn = callProvider) {
    this.#envUrl = envUrl;
    this.#brief = brief;
    this.#probes = probes;
    this.#call = call;
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
    // Dead-provider memory: a provider that has refused three calls in a
    // row is spent or out for the rest of the trial — at most two of the
    // refusals can be single-call failures, and budget and outage
    // refusals never recover — so stop spending calls on it, unless it
    // is all that is left.
    const live = candidates.filter((c) => (this.#refusalStreaks.get(c.provider.id) ?? 0) < 3);
    return live.length > 0 ? live : candidates;
  }

  /** Try candidates in order until one answers; null when all refuse. */
  async serve(req: CanonicalRequest, requires: Requirements): Promise<Served | null> {
    for (const candidate of this.candidatesFor(req, requires)) {
      const id = candidate.provider.id;
      this.#attempts.set(id, (this.#attempts.get(id) ?? 0) + 1);
      const outcome = await this.#call(candidate.target, req, candidate.row.model);
      if (outcome.ok) {
        this.#refusalStreaks.set(id, 0);
        return { answer: outcome.answer, candidate };
      }
      // Only a real refusal counts toward the streak (the harness's own
      // failed-call statuses); a network error proves nothing.
      if (outcome.status === 402 || outcome.status === 429 || outcome.status >= 500) {
        this.#refusalStreaks.set(id, (this.#refusalStreaks.get(id) ?? 0) + 1);
      }
    }
    return null;
  }
}
