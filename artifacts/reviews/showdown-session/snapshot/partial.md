INCOMPLETE / UNVERIFIED — not a final verdict.

**Provisional verdict: findings (no blocking correctness bug in the covered paths; one consequential misclassification and one unverified server assumption).**

## Findings

**1. Medium — Transport loss during CLOSING is treated as a fatal protocol error (no reconnect).**
`packages/showdown-bot/client.mjs:43` throws `'Connection is not active'` whenever `socket.readyState !== 1`. Between the server's close frame (readyState → 2) and the `close` event, any timer-driven send in the sealed baseline — the decision timer (`:136`), keepalive (`:156`), or `/leave` on battleEnd (`:120`) — throws, and the baseline `fail()`s with that message. `packages/showdown-session/client.mjs:136` does not list it in `TRANSPORT`, so `connectionFailed` escalates to a fatal failure instead of scheduling a retry.
Sequence: active request, decision timer pending → server begins close handshake → timer fires → `send` throws → `fail('Connection is not active')` → session `failure`, `expected` never populated, battle abandoned. Same path if the socket drops between `updateuser` and the `ready` handler's `/join` (`:106`).
Verification: mock socket with `readyState = 2` before dispatching `close`, with `decisionDelayMs: 0` and a live request; expect `reconnecting`, not `failure`.

**2. Low/Medium — `|sentchoice|` binding relies on unverified server state hygiene.**
`client.mjs:102` (baseline) marks the *current* request as committed on any `|sentchoice|`; the line carries no rqid. The session reports `committed: true` and never acts. Correct only if the pinned `room-battle` clears `player.request.choice` after `[Unavailable choice]` / `[Invalid choice]` and on every new `receiveRequest`. If a rejected-but-stored choice is echoed on rejoin, the bot sits silently on a live request until the battle timer forfeits. The supplied excerpt doesn't show this; please confirm in the pinned server, and add a loopback test: choose, receive `[Unavailable choice]`, drop, rejoin, assert a decision is emitted.

**3. Low — One-frame request+sentchoice is a hard dependency.**
With `decisionDelayMs: 0`, a 0 ms timer and a subsequent I/O message are both macrotasks with no ordering guarantee. If request and sentchoice ever arrive in separate frames, the duplicate `/choose` yields a server `|error|` → fatal via baseline `:110`. Acceptable given the pinned server, but document it; no test exercises the split case (test 2 deliberately sends them together).

**4. Low — Coverage gaps versus the review objective.**
No test for: (a) multiple pending rooms restored across one reconnect; (b) a second drop mid-recovery (history received, request not), exercising the `Math.max` merge at session `:140-141`; (c) forced-switch re-request with same turn after a committed restore (`rqid+1`, no `|turn|`). The code paths look correct by inspection (`:48` uses `<`, baseline accepts same-turn higher rqid), but these are the stated focus areas.

**5. Informational.** `finished`, `decisions`, `retries` (session) and baseline `rooms` grow for the session lifetime; bounded only by battles played. Not a defect for single-battle-at-a-time use.

## Scope limitations
- No access to the pinned server `room-battle.ts`/`rooms.ts` beyond the described excerpt; findings 2–3 are assumption checks, not confirmed defects.
- Cannot confirm whether the server auto-rejoins a reconnecting user (would double-`init` and trip `:29`); PS `joinRoom` is idempotent per connection, so likely fine.
- Loopback fault tests were not observed.

## Verdict
Findings. Finding 1 is a concrete, reproducible gap in retry classification caused by the sealed baseline's `send` guard; the fix belongs in the session layer (treat `'Connection is not active'` as transport when `child.socket?.readyState !== 1`, or compare against the socket state at failure time). Duplicate-choice, rollback, admission, and budget logic otherwise hold under the stated one-frame server behavior.