**Provisional verdict: no blocking findings in reviewed scope; one medium-risk unverified dependency and a few low-severity items.**

## Findings

**1. Medium — Adapter correctness depends on an unreviewed TacticalView/createActor contract** (`showdown-confirmation/client.mjs:24-26`). The sealed `ShowdownBot.receive` expects `room.state` to expose `ingest`, `ended`, `winner`, `turn`, `gameType`, `tier`, and calls `room.choose(structuredClone(request), structuredClone(state))` (`showdown-bot/client.mjs:90,115-125,132`). `structuredClone` strips prototypes, so if `actor.choose` relies on `TacticalView` methods on its second argument (rather than closing over `actor.state`), tactical decisions would silently differ from the simulator panel. `view.mjs`/`policy.mjs` are not in the packet. Test 1 (`client.test.mjs:43-52`) exercises revealed-ability handling and would catch a crash, but not a degraded-but-legal fallback. Verification: confirm in `createActor` that `choose` ignores or only reads plain data from the state argument, and that simulator `playGame` invokes `choose` identically.

**2. Low — No positive guard against baseline fallback.** The heuristic fallback at `showdown-bot/client.mjs:87-88` is unreachable in practice because `SessionConnection.receive` throws on any battle line before `|init|battle` (`showdown-session/client.mjs:34`) and the adapter seeds the room on every new `|init|battle`. However the adapter never asserts this invariant. Suggest: after `await receive(frame)`, throw if any battle room's `state` is not a `TacticalView` (or lacks a marker). Cheap, closes the residual risk if sealed ordering is ever misunderstood.

**3. Low — Async ordering is sound but relies on an undocumented sealed-parent property.** `super.attempt()` runs synchronously through `child.connect()` (socket created, listeners installed) before its first `await` at `showdown-session/client.mjs:124`; the message listener defers to `this.receive` via a `.then` microtask (`showdown-bot/client.mjs:56`), so the instance-property override at `client.mjs:20` always precedes the first `receive` call. Correct, but add a test that a message dispatched synchronously inside `socketFactory` still hits the wrapper.

**4. Low — Freshness check omits the confirmation smoke namespace** (`teams.mjs:6`). Only panel/panel-smoke pools are excluded; `showdown-tactical-confirmation-smoke-v1` teams are not. Since `developmentTeams` is namespace-keyed this is almost certainly disjoint, but plan.json "freshness" doesn't state it; audit reproduces `freshTeams` so bindings are consistent.

**5. Info — Seal script coverage is tight and self-consistent.** `preregistration.provenance == admission()` binds source hashes; `verification.json.source_files == now.source_files` binds tests to the same tree; per-game sha256 and audit hashes verified; `final_teams_used == 0` enforced. Hardcoded Python suite counts (`seal-showdown-confirmation.py:65`) will need updating if suites change — intentional brittleness.

Committed-action suppression (`sentchoice` → `room.sent`, `restored.committed`), stale-socket generation checks, and transport-only reconnect classification are inherited unchanged and covered by tests 3,4,10,17,18.

## Scope limitations
- Cannot run tests; test pass claimed only via tool logs.
- `statistics.mjs` (`confirm`), `TacticalView`, `createActor`, `playGame`, `developmentTeams`, `report.mjs` (panel) not supplied; statistical decision rule and quantile implementation unreviewed (separate QA pending).
- No Git diff; cannot confirm which lines are new.

## Verdict
**No findings requiring change before proceeding, conditional on verifying item 1 (choose/state contract under `structuredClone`).** Items 2–4 are recommended hardening, not blockers.
