# Spec: Free Model Aggregator (make.fun free-model-aggregator-01)

## Objective

Build the entry for the make.fun competition **free-model-aggregator-01**: one
Node server (`main.ts`, Node 22 standard library only, TypeScript run directly)
that sits in front of ten simulated free-tier providers and answers twenty
client requests per trial, on two client surfaces (OpenAI
`POST /v1/chat/completions`, Anthropic `POST /v1/messages`, plus
`GET /v1/models`). The trial scores five columns — quality (60), served (15),
wire (10), failover (10), catalog (5) — combined as
`round((60q + 15s + 10w + 10f + 5c) / 100)`, ties broken by fewer upstream
calls. The kit's starter scores **38** on practice trial-0 (quality 38,
served 40, wire 39, failover 0, catalog 100).

Success means: every request answered by exactly one successful call to a
free model that meets the request's manifest requirements, with the
provider's answer relayed byte-faithfully (text, stop reason, usage), clean
failover when a provider refuses, and a catalog document that matches the
providers' real free sets.

## Assumptions (surfaced, per spec-driven development)

1. The parent agent's written brief is the approval for this spec; the
   deadline (~19 h) does not allow a separate review round-trip.
2. The kit's `RULES.txt`, harness source (`program/`) and practice trials are
   the ground truth for scoring; where prose and code disagree, code wins.
3. The brief's three documents (providers, ratings, workload manifest) are
   authoritative at runtime; nothing about budgets or failures is known in
   advance, so routing must be adaptive, never memorized per trial.
4. The client sends the twenty requests one at a time, in manifest order.
5. No LLM calls: the entry is a scripted server (allowed; model calls are
   optional for this competition). No secrets are read or stored.

## Capability Map

| Module id | Responsibility | Depends on |
|---|---|---|
| brief | Parse and validate the three brief documents | — |
| catalog | Probe provider model lists; build the register catalog and the free-model list | brief |
| echo | Extract a canonical request from a client body (system text, messages block-for-block, tool defs, max_tokens, stops, json mode) | brief |
| wires | Render a canonical request into each provider wire (openai / anthropic / gemini), call the provider non-streaming, parse its answer | echo |
| route | Pick candidates (free, listed, requirement-meeting, echo-feasible), order by rating, fail over on refusal | catalog, wires |
| render | Render an answer onto the client surface (JSON or SSE, both wires) with deterministic ids/timestamps | wires |
| main | Trial lifecycle: listen, register, serve, poll `/work`, exit 0; `--list` prints the catalog | all |

Build order: brief → catalog → echo → wires → route → render → main.

## Commands

```
Test:        node --test "tests/*.test.ts"
Trial (one): /tmp/agg1/run-trial.sh <trial-n> <bot-dir>     (dev harness around the kit env + sealed scorer)
Grade:       node cli/makefun.mjs grade-local --kit ~/workspace/makefun/free-model-aggregator-01-kit --bot ~/workspace/makefun/agg1-repo --driver process
List:        node main.ts --list                            (needs MAKEFUN_PROMPT_FILE / MAKEFUN_ENV_URL)
```

## Project Structure

```
main.ts          entrypoint: lifecycle + HTTP server (thin; delegates to lib/)
lib/brief.ts     brief parsing and shared types
lib/catalog.ts   provider probes + catalog document
lib/echo.ts      canonical request extraction from client bodies
lib/wires.ts     provider-wire rendering, provider calls, answer parsing
lib/route.ts     candidate selection + failover attempt loop
lib/render.ts    client-surface rendering (JSON / SSE, both wires)
tests/           node:test unit tests (not loaded by the entrypoint)
.makefun/owner.json   repository ownership claim
SPEC.md          this file
README.md        what it is, how it routes, how to run
LICENSE          MIT
```

## Code Style

TypeScript, ESM, relative imports always carry the `.ts` extension, no
dependencies. Small pure functions; validation only at boundaries (brief,
client bodies, provider replies). Example:

```ts
/** Candidates for one request, best first. Pure: same inputs, same order. */
export function rankCandidates(directory: Directory, requires: Requirements): Candidate[] {
  return directory.models
    .filter((m) => m.free && m.listed && meets(m.row, requires))
    .sort((a, b) => b.row.rating_milli - a.row.rating_milli || a.callsMade - b.callsMade);
}
```

## Testing Strategy

- **Unit (node:test, `tests/`):** echo round-trip — for every practice
  workload body and every provider wire, rendering the canonical request and
  re-parsing it with test-side copies of the harness's own parsers must
  reproduce the client body's echo exactly; candidate filtering (free,
  requirements, JSON-mode feasibility); SSE frame ordering for both client
  wires; catalog free-set equality against the practice reference.
- **Integration:** the real kit env + sealed scorer over practice trials
  0, 1 and 2 (dev harness), then the kit's `grade-local` (process driver).
- A slice is done only when its tests pass and the previous slices still do.

## Boundaries

- **Always:** develop against the practice trials; keep exactly one
  successful provider call per request; keep every output deterministic
  (no wall clock, no randomness); commit in small verified slices.
- **Ask first:** nothing — the brief fixes scope; surprises go in the report.
- **Never:** network beyond the trial env; push or submit; read keypair or
  credentials files; call a paid or unlisted model; add a dependency.

## Success Criteria

- [ ] Practice trials 0, 1, 2: served 100, wire 100, catalog 100,
      failover 100, quality ≥ 95 (score ≥ 97) — or the gap explained.
- [ ] `grade-local` on trial-0 clearly beats the starter baseline of 38.
- [ ] `node main.ts --list` prints the same catalog the register body carries.
- [ ] Unit tests pass; final review pass over the diff recorded in the report.

## Open Questions

None blocking. Budgets/failures are drawn per trial and unknowable; the
router therefore plans nothing around them beyond rating order + failover.
