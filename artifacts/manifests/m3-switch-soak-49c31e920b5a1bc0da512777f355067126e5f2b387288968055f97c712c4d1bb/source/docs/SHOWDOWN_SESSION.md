# Recoverable Showdown session

This additive client wraps the sealed playable baseline with bounded transport
recovery. It uses the same deterministic public-view heuristic and the real
Showdown battle engine. It does not add learned-policy strength or custom JAX
mechanics. Earlier handoffs and sealed source/evidence are preserved.

## Run locally

Start the server as described in [the baseline guide](PLAYABLE_BASELINE.md), using
a fresh output directory. Run the recoverable client in another terminal:

```sh
npm run bot:recover -- --name BaselineBot --accept YourName
```

The original `npm run bot` remains available with its original fail-on-disconnect
behavior. The new command uses the heuristic only, accepts challenges from named
allowlisted players and optionally supports `--challenge NAME`. It has no ladder
search option. Passwords for remote WSS are read from `SHOWDOWN_PASSWORD` on each
connection; the CLI never prints them or authentication assertions.

Public login and public games remain untested. See the [deployment status and
official policy sources](SHOWDOWN_DEPLOYMENT.md) before attempting public play.

## Recovery contract

On a transport failure the session cancels the old client's timers, creates a new
socket, authenticates again and joins each already-known unfinished battle room.
It rebuilds the reducer from the server's player-visible history and reads the
current own request. It never copies an old request, pending action or sent marker
into the new connection. The server's combined request/sentchoice frame suppresses
an already-accepted choice. If an action was lost before transport, a fresh
uncommitted server request permits the heuristic to choose again. This is not a
general exactly-once transport guarantee.

Default limits are three reconnects across the whole session, 1/2/4-second retry
delays, a 30-second login timeout per attempt and a 30-second active-room recovery
deadline. Success does not reset the retry budget. Authentication/protocol errors,
unavailable rooms, backwards history/request IDs and seat changes fail visibly.
Hidden-legality rejection still waits for Showdown's refreshed request. Ctrl-C
cancels recovery. Decision counters span connections. Stale callbacks cannot
reactivate a replaced connection. Each connection allows at most 64 room histories,
and the session caps attempted submissions per battle at 20,000.

The ordering contract is specific to the pinned server: rooms.ts sends initialization
and complete player-visible scrollback, then room-battle.ts sends the current
request and any sentchoice together. This was checked in source and real traffic.
A server that splits request/sentchoice across frames is not qualified by this work.

## Evidence

Reproduce in new output directories:

```sh
npm run bot:recovery-smoke -- artifacts/runs/NEW-session 18010
npm run bot:recovery-audit -- artifacts/runs/NEW-session
npm test
```

Six complete local-server games exercise disconnect before choosing, a discarded
outgoing action, disconnect after writing an accepted action, a later-turn request,
a forced replacement, and two successive disconnects. The campaign uses two
freshly generated development-partition teams, admitted before play, in the
baseline's unrated local format with stock Random Battle rules. These are transport
fixtures, not independent strength samples or a broad team-distribution evaluation.

The wire audit reconstructs each connection's public reducer and own request,
checks every attempted action and actor input, forbids duplicate/committed choices,
and recomputes fault/reconnect/restore counts and terminal outcomes. One intentionally
lost write is counted as an attempted action, never as a server-accepted move.
This audit checks the transport and actor projection; it does not independently
re-simulate these unseeded local-server games.

Nineteen focused tests additionally cover shutdown, budget exhaustion, missing
history/rooms, multiple pending rooms, repeated loss midway through restoration,
fresh authentication with a mock endpoint, close-handshake races, same-turn forced
switching and a real Arena Trap rejection through pinned server methods. Four
deliberately corrupted report/trace copies fail the audit. Cross-model review
findings, the repaired close-handshake race and review limitations are recorded in
`artifacts/reviews/showdown-session/review-dispositions.md`.

## Measured result — 2026-10-11

The accepted production run is `artifacts/runs/showdown-session-v1`: six complete
games, seven injected disconnects, thirteen recovering-client connections and restores
for all seven failures. Both players agree on each terminal result. All **120
attempted decisions** pass the wire reconstruction/action audit; one write was
deliberately discarded and later reconsidered from the fresh server request.
There are zero client failures or unavailable-choice retries in these six games.
The accepted pending action is confirmed by a real sentchoice frame and is never
resubmitted. The separate server-method test covers unavailable-choice hygiene.

| Fault scenario | Turns | Attempted decisions | Disconnects |
|---|---:|---:|---:|
| Before scheduled choice | 17 | 19 | 1 |
| Outgoing write discarded | 17 | 20 | 1 |
| After accepted action write | 17 | 19 | 1 |
| Later-turn request | 18 | 20 | 1 |
| Forced replacement | 19 | 22 | 1 |
| Two successive disconnects | 18 | 20 | 2 |

The [completion receipt](../artifacts/reports/showdown-session.json) binds the
source, immutable predecessor, real wire trace, audit, regression and review records.
The final source-bound regression passes **303 tests (114 Node + 189 Python)**
with all **254 tested source/configuration files unchanged**. Node syntax checks
and pinned build verification also pass. Four targeted corruptions of the final
production report/trace are rejected, including a replay inserted immediately
after the server confirms an accepted choice.

## Remaining limits

Recovery is in memory, for already-observed room IDs and this stateless policy.
It does not resume after process restart, recover a challenge lost before the room
ID was received, persist recurrent policy memory, retry challenges, or certify
arbitrarily long outages. Server popups remain fatal. Public-server authentication,
live protocol compatibility, browser play and sustained ladder operation remain
unverified. The network tests use a loopback-only unthrottled server; they do not
measure public-server throttling or network latency.

Next establish account/registration details and run a controlled public direct
challenge, or continue locally with a stronger frozen opponent panel. Continue
the switching/hidden-set scalar/JAX port separately before full-format training.
Full G2/G3 remain open; no final reserved data, new training, GPU or Prime resources
were used.
