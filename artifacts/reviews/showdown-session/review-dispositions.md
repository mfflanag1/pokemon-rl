# Recovery review dispositions — 2026-10-11

Fable snapshot review: `20261011-125327-eaacc925`, completed in 212.88 seconds.
Its four-file packet covered the new session client, the first twelve tests and
the sealed baseline client/protocol. The review prompt summarized the pinned
server ordering; it did not actually include a server source excerpt. The reviewer
had no tools and did not execute tests. Review/source deltas are retained separately.

1. **Close-handshake misclassification — confirmed and repaired.** The sealed
   send guard throws a generic inactive-connection error when readyState is 2,
   which the original new session did not classify as transport failure. The new
   layer checks state before that guard and emits its own transport error; its
   ready callback now delegates errors to child.fail so the same retry path applies.
   Counters increment only after a successful send call. Two focused tests verify
   a pending move and an authenticated rejoin during CLOSING. Both recover without
   a permanent failure or falsely counted unsent move. The sealed baseline is intact.
2. **Rejected-choice cache hygiene — checked against pinned executable code.**
   sim/side.ts:527 emits the Unavailable error followed by the updated request.
   server/room-battle.ts:803 replaces the request object with choice='', a new rqid
   and actionable isWait=false. The new server-method fixture runs a real Arena
   Trap rejection through the pinned RoomBattle.choose/receive/onConnect methods.
   It verifies order, changed legality, increasing rqid and no stale sentchoice on
   reconnect. This is a simulator/server-method integration test, not a separate
   loopback hidden-trapping game. Ordinary invalid choices remain fatal client errors.
3. **Atomic request/sentchoice frame — a documented pin dependency.**
   server/room-battle.ts:946-962 sends both together; rooms.ts:2011 sends complete
   player-specific history before calling that method. The real accepted-choice
   fault case observes this and suppresses resubmission. No compatibility with a
   server that splits the two lines across frames is claimed.
4. **Coverage gaps — addressed locally.** Additional tests cover two pending rooms,
   a second disconnect after history but before request, preserved rqid floors,
   and same-turn forced replacement after a committed restore. The real six-case
   campaign includes forced switching and two successive transport losses.
5. **Lifetime bookkeeping — bounded, not a long soak.** Each connection admits at
   most 64 initialized battle rooms and a session permits at most three reconnects
   by default (configurable to ten). No indefinite memory/uptime claim is made.

The final client repair and seven added tests were verified locally; no second
Fable approval is claimed. The separate deployment-research packet received Codex
QA and was rejected for overclaiming current registration status and treating
configurable implementation limits as policy. Independently verified facts and
remaining unknowns appear in docs/SHOWDOWN_DEPLOYMENT.md.

Two exploratory network campaigns are preserved. The second passes a wire audit
of 121 attempted actions. The first lacks the later explicit wire/session generation
distinction and is not the accepted audit source. Four corrupted copies of the
second trace/report were rejected (actor view, duplicate accepted action, terminal
result, fault inventory). The server-method fixture initially failed because its
import order loaded server Config before the oracle; importing server methods after
the frozen oracle fixes the isolated test. No qualification failure was concealed.

Only final source-bound production evidence counts toward the completion receipt.
The final production run passes all six scenarios, seven restorations and 120
wire/actor checks. Four fresh corruptions of that final evidence fail the audit;
the accepted-action replay probe is inserted after the actual sentchoice restoration
and fails specifically on the duplicate/committed-choice guard. All 303 regression
tests pass (114 Node, 189 Python); 254 tested source files remain unchanged.
