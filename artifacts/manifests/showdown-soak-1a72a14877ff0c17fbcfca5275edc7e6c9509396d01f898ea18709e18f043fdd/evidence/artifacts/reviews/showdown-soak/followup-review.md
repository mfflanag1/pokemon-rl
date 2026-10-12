**Provisional verdict: findings (none blocking; one audit-coverage gap worth closing before production).**

Most important: the auditor accepts an `accepted-choice` game in which the baseline hold never happened, so the "deterministic hold" is not actually verified per game.

## Findings

**1. Medium — audit does not require the hold to be exercised** (`audit.mjs` 165–170, 146; `campaign.mjs` 34, 85–94)
The hold only engages if baseline's first `/choose` precedes tactical's `restored` event. If tactical reconnects faster than baseline's `decisionDelayMs` + policy latency, `released` is set first, no `send` carries `held:true`, no `deliver` occurs, and every check still passes (`every(s => s.delivered || s.held)`, `p.held === null`, `restores[0].committed`). The fixture test (`audit.test.mjs` 41–45) only asserts the smoke fixture happened to contain a `deliver`; the production audit has no such requirement. Verification: in `check()`, for `accepted-choice` assert exactly one baseline send with `held === true` and exactly one `deliver` event, and that the hold's `at_ms` precedes `per.tactical.restores[0].at_ms`.

**2. Low — receipt equality relies on unproven pack/unpack idempotence** (`teams.mjs` 36–43)
Server path is `pack(client) → unpack → validate → pack → unpack → Pokemon ctor (name) → pack(side.team) → unpack (RoomBattle) → pack (plugin)`. Reference path is `unpack(pack(team)) → validate → name → pack`. Equality holds only if `pack∘unpack` is idempotent on validator-canonicalized sets (hpType/teraType/gigantamax/level/happiness defaults). The 148+154 smoke matches are empirical evidence for the smoke pool only; the production pool is different teams. Verification: a unit test asserting `pack(unpack(pack(validated))) === pack(validated)` over the full production `freshTeams(false)` pool before sealing.

**3. Low — `assertJSON` runs after `structuredClone`, which already strips class prototypes** (`wire.mjs` 10–13)
`structuredClone(room.state)` turns `TacticalView`/`PublicBattle` instances into plain objects, so the `Object.prototype` check can never flag an instance-typed field, and functions/symbols throw `DataCloneError` rather than the intended assertion. `Map`/`Set`/`Date`/non-finite are still caught (test line 27 covers these). Sparse-array holes are skipped by `Object.values` but serialize as `null`; harmless because both sides go through JSON. Not a correctness bug, just narrower than described.

**4. Low — held action has no bound other than game timeout** (`campaign.mjs` 36, 110)
If tactical never emits `restored` (e.g. reconnect failure), the baseline socket silently withholds its choice and the game only fails via `game_timeout_ms`; `failure.json` will then show a timeout rather than the real cause. Acceptable, but worth a clearer assertion message that references the held state.

**5. Informational** — `campaign.mjs` 134 finds the first `restored` of any role; correct only because the audit independently forbids baseline restores (line 132/153).

## Scope limitations
- Not in packet: `RecoveringBot`/`TacticalBot` (whether baseline has any silence-triggered resend that a long hold could trip), the sealer's `write=false` re-run comparison, `schedule.mjs` delay/retry values, and pinned upstream `pokemon.ts`/`room-battle.ts`; name-normalization and `getTeam` FIFO ordering are taken from the stated pinned-source claim.
- No tests executed; fixture `network.json.gz` not inspected.

## Verdict
Findings. Changed logic is sound as far as visible; close #1 (and ideally #2) in the audit before the production run so the claimed guarantees are enforced rather than assumed.
