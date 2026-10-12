# Frozen tactical client: local network soak

This milestone qualifies sustained local network operation of the already sealed
`tactical-v1` client. It adds a harness and auditor; the policy, transport and
recovering client remain unchanged. The predecessor is the
[fresh tactical confirmation](SHOWDOWN_TACTICAL_CONFIRMATION.md).

## Fixed experiment

`packages/showdown-soak/plan.json` fixes 1,000 complete loopback-server games in
20 fresh login sessions of 50 games each. The primary tactical client faces the
original public heuristic through the sealed recovering baseline client. Five
hundred team pairs keep teams in fixed simulator seats while swapping policy
seats. All 1,000 distinct teams come from the pinned official generator and pass
the frozen development partition. Exact overlap against the named prior pools
in `teams.mjs` and the eight own-smoke teams fails admission without replacement.

Each session injects one disconnect before the first scheduled choice, one lost
outgoing choice and one disconnect just after an accepted pending choice. Offset
parity distributes all three faults across both tactical seats: 60 total faults,
20 of each kind. The baseline socket stays connected. The fixed design expects
80 tactical and 20 baseline connections, 980 within-session rematches, and 500
games in each tactical seat. Sessions rotate before the inherited 64-room limit;
this is not a claim of indefinite operation on one connection or process restart.
All three permitted tactical reconnects are consumed deliberately; an extra
unplanned transport loss fails the session. The accepted-choice fixture holds at
most one baseline outgoing action until tactical restoration, recording attempt
and actual delivery separately. It does not rely on winning a fixed-delay race.

Each saved game includes the actual network inputs, attempted sends and separately
captured actor inputs, plus terminal results and client memory/session counters.
The auditor reconstructs both players independently across rematches and fresh
connections. It checks own requests, uploaded official teams, policy seats,
frozen choices, accepted-choice suppression, lost-write reconsideration, login
identities, recovery counters, duplicate rooms and terminal agreement. Both
players' private records are audit evidence; each policy receives only its own
request and reconstructed public view.

The local server copy adds a private, read-only `/soakteam` test command. It uses
the original `RoomBattle.getTeam(user)` path and simulator `requestteam` response
to return only the actual caller's original team; targets and nonparticipants are
rejected. It is enabled only for the loopback test configuration. The auditor
compares the complete packed receipt, with the simulator's canonical default
nickname. No item/move/stat fields are dropped to get a match. This avoids
comparing mutable first-turn requests after Imposter or entry effects. The client
policy ignores this test receipt; no battle mechanics are changed.

The standard `/showteam` command was unsuitable for unregistered local identities
because it requires registered chat access. That failed preflight is preserved.
A second eight-game preflight passed after matching the constructed validator's
normalization, but a full-pool check then found official random sets rejected by
constructed legality (164 of 1,000 teams, including event-specific IV requirements). Normal Random
Battles bypass those checks. The final local format therefore admits only exact
packed teams from the regenerated official development pool and preserves all
Random Battle rules and set values. It neither edits nor excludes rejected sets.

The third preflight completes eight games and six faults, with every one of 159
tactical and 165 baseline decisions passing the wire audit. That trace is the
regression fixture. Negative cases corrupt actor inputs, outcomes, team receipts,
recovery, delivery, login, timing and counters. The full-pool preflight checks
allowlist acceptance without set mutation and stable canonical serialization;
the first pack/unpack changes move ID spelling to display names, not battle values.

The real local server controls battle randomness. We do not claim coupled chance,
fresh deterministic simulator replay or a paired strength result from this soak.
The exact wire history, code and outcome are retained for reconstruction. The
server uses the pinned Random Battle rules with explicit development teams and
an unrated, non-searchable custom format, all on loopback.

Any timeout, unexpected client/server error, missing planned recovery, memory
budget failure or audit mismatch fails the run. No failed game is silently replaced
or scored as a loss. Client RSS must remain at or below 768 MiB at each game
boundary; sampled heap/RSS and request-to-attempt latency are reported. This does
not certify server or device memory, continuous unsampled memory peaks, or public
network latency. Each game has a 120-second timeout and the inherited action cap.

## Reproduction

Use new output directories. Production binds its frozen plan and source/runtime
admission before any game. A source-bound full regression must pass on the same
tree; do not run the regression while a network campaign is active because older
provenance tests temporarily mutate sentinels.
Sealing regenerates the development teams and reruns the complete wire audit,
then compares its result to the saved audit before archiving the receipt.

```sh
npm run bot:soak -- artifacts/runs/NEW-smoke --smoke
npm run bot:soak-audit -- artifacts/runs/NEW-smoke
.venv/bin/python scripts/verify-fixed-loop.py artifacts/reviews/showdown-soak
npm run bot:soak -- artifacts/runs/NEW-production
npm run bot:soak-audit -- artifacts/runs/NEW-production
.venv/bin/python scripts/seal-showdown-soak.py artifacts/runs/NEW-production
```

## Result

The production run `artifacts/runs/showdown-soak-v1` completes all **1,000 games**
in **169.097 seconds** of campaign elapsed time, including admission and server
setup. No game is replaced, no policy is tuned and no unexpected failure occurs.
The separate auditor regenerates the official team pool and reconstructs both
players from their actual received network histories.

| Measurement | Verified result |
|---|---:|
| Distinct official development teams | 1,000 |
| Login sessions / within-session rematches | 20 / 980 |
| Tactical games in each seat | 500 / 500 |
| Before-choice / lost-write / accepted-choice faults | 20 / 20 / 20 |
| Tactical / baseline connections | 80 / 20 |
| Audited tactical / baseline decisions | 22,928 / 23,096 |
| Total audited decisions | 46,024 |
| Unavailable-choice retries | 0 |
| Peak sampled client RSS / heap | 445.063 / 127.693 MiB |
| Tactical request-to-attempt median / p99 / max | 2.473 / 2.941 / 23.791 ms |
| Baseline request-to-attempt median / p99 / max | 2.430 / 2.888 / 23.769 ms |

All planned recoveries pass, including suppression of already accepted actions
and reconsideration of a lost write. Every game has exact own-team receipts,
matching terminal results, request/counter linkage and player-view policy choices.
No known-forbidden action or unexplained protocol disagreement is observed.
Timing includes the fixed 2 ms local decision delay and is not a public-network
latency guarantee. Memory is sampled at game boundaries, not a continuous peak.

The full source-bound regression passes **426 tests (237 Node + 189 Python)** on
**298 unchanged source/configuration files**. All 1,523 unmodified files in the
disposable server match the pinned upstream; only the declared local configuration,
exact-pool format and own-team receipt endpoint differ. Build verification passes.
Eight corrupted-evidence cases are rejected at their intended gates, including
a forged saved audit caught by fresh reconstruction; none can write a receipt.
The production pool, source/runtime admission, complete wire audit, review outcomes,
preflight evidence and regression are bound by the
[completion receipt](../artifacts/reports/showdown-soak.json).

The cross-model record preserves the rejected first scheduler delivery, accepted
minimal repair, opposite-model QA and three snapshot reviews with independently
checked dispositions. The failed preflights and diagnostic serialization checks
remain available; they are not counted as production games.

This satisfies the **1,000-game local adapter soak prerequisite** for this frozen
client, pinned runtime and declared fault/session scope. It does not close full
G2/G3/G4, establish public-server readiness, certify indefinite sessions or process
restart, or measure competitive strength. No public login/game, model training,
final evaluation data or Prime resources were used.

**Next:** implement the already versioned M3 switching/hidden-set scalar and JAX
transitions and player-view reducer, establish conditional and fresh hidden-state
parity, then build the categorical actor and phase-aware collector. Preserve all
sealed predecessors and keep final evaluation reserved. Local CPU remains enough.
