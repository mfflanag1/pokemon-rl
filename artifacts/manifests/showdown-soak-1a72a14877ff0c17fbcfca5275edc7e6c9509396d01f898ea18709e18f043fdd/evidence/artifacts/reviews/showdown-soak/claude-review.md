**Provisional verdict: findings (no blocker in reviewed code; several audit gaps and one timing-fragile assertion).**

Most important: the `accepted-choice` audit requires `restores[0].committed === true` (audit.mjs:143), which depends on uncontrolled timing and server behaviour, not on the harness.

## Findings

**1. Medium — `accepted-choice` pass/fail hinges on a race and an unverified server behaviour.** campaign.mjs:88 slows the baseline to 800 ms; the tactical reconnect (80 ms backoff + login + `/join`) must complete before baseline moves, otherwise the turn resolves, the rejoin request carries a new rqid, `sent` is empty, `committed` is false and the whole 1000-game audit fails with no distinguishing message. It also assumes the server replays `|sentchoice|` on rejoin (client.mjs:102 is the only place `sent` is populated). Verification: smoke run on a loaded machine; log baseline decision time vs tactical `restored` time; consider gating baseline’s decision on tactical’s `restored` event rather than a fixed delay.

**2. Medium — "both-seat team upload" is checked only on the sent `/utm` string, not on the received wire.** audit.mjs:81 compares the packed string; audit.mjs:69 only checks `pokemon.length === 6`. Nothing binds `request.side.pokemon` (species/moves/items) to `teams[spec[role_team]]`, so the fixed-team-per-seat design invariant is not evidenced from the battle itself. Suggest asserting species/move sets per seat on the first non-wait request.

**3. Medium — sealer trusts `audit.json` content.** seal-showdown-soak.py:46–65 verifies that audit.json hashes bind to report/preregistration/teams, but never re-executes the audit. A stale or hand-edited audit.json with correct hashes seals. Suggest the sealer run `audit.mjs` (write=false) and compare to the on-disk result. Also noted: `NODE_TESTS=191` placeholder and hard-coded Python suite list (line 70) — per task, must be replaced before production.

**4. Low — reconnect budget has zero headroom.** `maxReconnects` defaults to 3 (session/client.mjs:73) and never resets within a session; the plan schedules exactly 3 faults per session (`fault_offsets` length). Any single unplanned transport hiccup fails the campaign. Acceptable under the stopping rule, but should be explicit in plan.json `scope`.

**5. Low — stray late frames on a closed session socket can land in the next game.** After `bots = null` (campaign.mjs:113) the old sockets’ `message` listeners still call `event()` with the *new* `current`; audit would then hit `Unknown connection` (audit.mjs:38). The 30 ms `intergame_delay_ms` mitigates but does not prevent. Suggest ignoring events once the socket’s generation is retired.

**6. Low — missing assertions:** (a) `lost-write`/`before-choice` don’t assert `!restores[0].committed`; (b) no check that a `delivered:false` send is never followed by `|sentchoice|` on that generation; (c) no ordering check that `team` precedes `challenge/accept` on the same generation; (d) a game timeout surfaces only as an `AbortError` in failure.json — add an explicit timeout message.

**7. Low — actor-input isolation check may be vacuous.** `actor_input` is compared after JSON round-trip (campaign.mjs:77 `structuredClone`→gzip JSON; audit.mjs:98 `plain`). If `TacticalView` holds `Map`/`Set` fields they serialize to `{}` on both sides and the comparison proves nothing. Verify `TacticalView` is plain-JSON representable.

## Scope limitations
- `schedule.mjs`, `TacticalView`, `chooseOpponent`, `developmentTeams`, `PublicBattle`, and the upstream server rejoin semantics are not in the packet; items 1 and 7 rely on them.
- No runs or tests were executed; preflight/smoke results absent.

## Verdict
Findings. Socket-generation arithmetic (`session_generation = 2g−1`), fault scheduling, hash/record binding, and failure preservation look internally consistent; the gaps above are in audit strength and timing robustness.
