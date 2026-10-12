**Provisional verdict: findings (one medium correctness/validity risk; rest low).** Most important: the client assumes the public log for a turn is visible before it must act on the turn's `|request|`, but the Showdown sim emits `sideupdate` (request) before the `update` chunk, so decisions can be made on stale public state — and the first-request format check can throw.

## Findings

**1. Medium — request/update ordering (client.mjs:123–131, 124).**
In `BattleStream._write`, `makeRequest → side.emitRequest` pushes `sideupdate` immediately; `sendUpdates()` (public `|tier|…|switch|…|turn|N`) is pushed only at the end of the chunk, and room-battle forwards each stream message as its own socket frame. Consequences:
- Triggering sequence (first turn): bot joins room (`|init|battle` frame, log empty) → frame `|request|{rqid:1,…}` → `touched` loop at line 124 sees `room.state.tier === null` → throws `Unexpected battle format or game type` → whole client fails. The reported two successful local games contradict this, so verify by dumping raw frame order from `server/server.log`/a socket trace; if `|tier|` does precede the first request on your pinned build, downgrade to (b) only.
- (b) With `decisionDelayMs: 0` (network-smoke) the timer fires before the turn's `|switch|/|-terastallize|/|turn|` frame arrives; `room.state` and the emitted `turn` lag by one turn. `campaign.playGame` ingests the full chunk before choosing (line 29), so the network bot and the benchmark heuristic see different public inputs for identical requests — undermines the "same policy, same observation" validity claim. Fix: treat null tier/gametype as "not yet known" (defer, don't throw) and defer the decision until a `|turn|`/`|start|` line newer than the request is seen.

**2. Low — repeat challenges silently ignored (client.mjs:164–182).**
`accepting` is only cleared by a `|pm|…|/challenge` (empty) PM or `updatechallenges`. Sequence: Friend challenges → accepted → battle ends → Friend challenges again → `accepting.has('friend')` → `acceptChallenge` no-ops. Whether the server sends the clearing PM on accept is unverified in the packet; check `Challenges.remove`/`send` in pinned `ladders-challenges.ts`. Also a challenge received mid-battle is dropped and never retried.

**3. Low — dev format is validated, stock random format is not (local-server.mjs:23–25).**
Dropping `team: 'random'` means `/utm` teams pass through the team validator (`Obtainable`, Species Clause). `playGame` (direct sim) performs no validation. A generated set the validator rejects produces `|popup|` → `global()` line 161 kills the client, while the campaign plays it fine. "Identical battle rules" is also slightly overstated (`|tier|`, `rated`). Verify with `developmentTeams(200, …)` against the validator.

**4. Low — `|popup|` is fatal for any content (client.mjs:160–161).** Benign popups (e.g. challenge declined, room-expiry notices) terminate the process. Consider scoping to login/challenge phases.

No issues found in: stale/duplicate rqid gating, `[Unavailable choice]` refresh (new rqid + different JSON), `sentchoice` replay suppression, revival/Tera/trapped legality, mirror seat scoring (`candidateSeat = mirror`, pair means equal audit's global mean exactly), loopback-only bind/WSS gating, or disconnect-without-replay.

## Scope limitations
- `engine.mjs`, `observations.mjs` (PlayerProjection ordering/`pending` semantics), `identity.mjs`, pinned PS source not in packet; finding 1 relies on recalled sim/room-battle behaviour and must be confirmed against the pinned build.
- No tests executed; no claims about public-server login behaviour.

## Verdict
Findings. Address #1 (verify frame order; defer rather than throw; align decision timing with the benchmark path) before treating network results as comparable to campaign results.
