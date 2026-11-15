# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/donghua9711-oss/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/donghua9711-oss/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)


One server that sits in front of ten free-tier AI providers and answers
like a single, better one. It speaks both client wires — OpenAI
(`POST /v1/chat/completions`) and Anthropic (`POST /v1/messages`), JSON
and streaming — and behind them it speaks all three provider wires
(OpenAI, Anthropic, Gemini). Every request is routed to the best free
model that can actually serve it, and the provider's answer is relayed
back whole: its text, its stop reason, its token usage.

Built for the make.fun competition **free-model-aggregator-01**, where a
trial hands the server a brief (the provider directory, a public rating
table, and a manifest of twenty requests) and scores quality, served,
wire fidelity, failover and the reported catalog.

## How it routes

1. **Probe.** At startup the server reads the brief and probes every
   provider's model list in the provider's own wire, then builds one
   catalog document — the same document `node main.ts --list` prints and
   the register call carries.
2. **Candidates.** For each request, a model is a candidate when it is
   rated in the brief and free there, is actually listed by a provider
   whose probe answered, and meets the request's manifest requirements
   (tools, vision, JSON mode, minimum context). One pairing is excluded
   on principle: an OpenAI JSON-mode request never goes to an
   Anthropic-wire provider, because that wire cannot echo JSON mode.
   The one paid model in the directory has no rating row for routing
   and is never called.
3. **Order.** Candidates are tried best-rating first; ties spread
   across providers by attempts made, so no single free tier is drained
   while an equal model idles elsewhere.
4. **Translation.** The client body is reduced to one canonical request
   (system text, block-for-block messages, tool definitions, max
   tokens, stop sequences, images) and rendered onto the provider's
   wire so the provider sees exactly what the client sent — nothing
   added, nothing dropped, no invented `max_tokens`.
5. **Failover.** Providers answer non-streaming; the answer is buffered
   whole, so a refusal (402/429/5xx) is invisible to the client — the
   router simply tries the next candidate, and the request ends with
   exactly one successful provider call. A provider that refuses three
   calls in a row is spent or out for the rest of the trial and stops
   being tried (unless it is all that is left).
6. **Rendering.** The buffered answer is written back on the client's
   wire with legal stream framing, fixed timestamps and counter-based
   ids — the whole server is deterministic, so a replayed trial is
   byte-identical.

## How to run

```sh
# A trial (the harness sets both variables):
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> node main.ts

# Print the catalog document without serving:
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> node main.ts --list

# Unit tests (echo round-trips, routing, rendering, catalog):
npm test
```

Node 22, standard library only — no dependencies, nothing to install.

## Layout

```
main.ts          trial lifecycle: listen, register, serve, poll, exit
lib/brief.ts     the brief's three documents, parsed and typed
lib/catalog.ts   provider probes and the catalog document
lib/echo.ts      client body -> canonical request
lib/wires.ts     canonical request -> provider wire; the call; answers
lib/route.ts     candidate selection and the failover loop
lib/render.ts    provider answer -> client wire (JSON and SSE)
tests/           unit tests, developed against the kit's practice trials
SPEC.md          the spec this was built to
```
