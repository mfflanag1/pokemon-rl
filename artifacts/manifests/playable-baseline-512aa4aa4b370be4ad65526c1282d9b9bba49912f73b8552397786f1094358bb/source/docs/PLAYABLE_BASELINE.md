# Playable Showdown baseline

This is a separate, deterministic heuristic bot for complete Gen 9 Random Battles.
Showdown remains the battle engine. The bot does not use the restricted Rattata
checkpoint or extend the custom JAX simulator. Full G2/G3 and competitive-strength
qualification remain open; local CPU is sufficient.

## Play locally

From the repository root, start a new local server directory in one terminal:

```sh
npm run bot:server -- artifacts/runs/my-local-showdown
```

In a second terminal, start the bot with the name you will use as its challenger:

```sh
npm run bot -- --name BaselineBot --accept YourName
```

Open `http://localhost:8000` in your browser. The stock Showdown landing page
redirects to its web client for the local server. Use the name `YourName` and
challenge `BaselineBot` to **Gen 9 Random Battle**. Keep both terminals running;
Ctrl-C stops the bot or server. Use a new server directory on each launch because
the launcher preserves prior evidence rather than overwriting it.

The local launcher copies the pinned server into the requested directory, shares
read-only dependency files, binds only `127.0.0.1`, disables the developer backdoor,
and permits local guest names without the public login service. Never expose this
development configuration to another network. The original pinned oracle is unchanged.
The web UI is provided by Showdown and requires internet access; the actual battle
and bot connection are local. Automated verification exercises real WebSocket
clients, not the browser UI.

Interactive games are not benchmark or training data. The verification campaign
below admits development-partition teams before play. Do not ingest future
interactive logs into the dataset without the existing partition/provenance gates.

## Public-server client

The client implements registered-account authentication over HTTPS and WSS.
Supply the account password as the `SHOWDOWN_PASSWORD` environment variable using
your local secret-management workflow; do not put it in arguments or source files.

```sh
npm run bot -- --url wss://sim3.psim.us/showdown/websocket \
  --name YourBotAccount --accept YourPlayerAccount
```

No public account was logged in and no public game or ladder search was performed
for this milestone. Public credentials, current server bot requirements, upstream
version differences and actual public-server compatibility remain deployment checks.
The CLI has no ladder-search option. `--challenge NAME` initiates a direct challenge;
`--accept NAME` accepts only that named player's exact-format challenge and may repeat
for additional allowed names. It sends no chat messages.

## Policy and protocol behavior

`public-matchup-heuristic-v1` ranks moves using public move/type data, visible
opponent typing and its own exact stats. It considers offensive Tera, healing at
low HP, voluntary switches out of poor offensive matchups, and forced replacements.
Status/setup play is deliberately basic. This is a ranking heuristic, not an exact
damage model: items, abilities, boosts, weather, move callbacks and many interactions
are not modeled in its score. Showdown still resolves all those mechanics correctly.

The actor receives only its own private request and a public-event reducer.
It cannot inspect the battle engine, opposing requests, unrevealed sets or seeds.
Battle rooms have separate request IDs, state and policy instances. Duplicate/stale
requests do not resubmit; `wait`, null requests, `sentchoice` and terminal messages
suppress choices. Hidden trapping/disabling errors wait for a refreshed request.
Ordinary invalid choices fail visibly. Revival Blessing targets fainted slots.
Modern PM challenges and the older documented challenge notification are supported.

Disconnects fail visibly and cancel scheduled choices. There is no automatic
reconnection or mid-battle recovery. A keepalive detects a nonresponsive connection.
The default decision delay is 750 ms; local automated smoke tests use zero delay
on an unthrottled loopback server. Authentication assertions and passwords are not
written to the diagnostic event log.

## Reproduction and evidence

```sh
npm test
npm run bot:benchmark -- artifacts/runs/NEW-baseline 32 NEW-seed-namespace
npm run bot:audit -- artifacts/runs/NEW-baseline
npm run bot:network-smoke -- artifacts/runs/NEW-network-smoke 18000
```

The benchmark admits fresh official teams only when the frozen R7 identity policy
assigns them to development. Each pair keeps the teams and mechanics seed fixed
and swaps the heuristic/random policies across the two seats. The random baseline
selects non-Tera moves 90% of the time when moves are available, otherwise samples
visible non-Tera choices. This is a weak functional baseline, not a ladder opponent.
Wins are computed from the terminal winner; Showdown's internal `score` is retained
separately as remaining-Pokémon counts.

Every saved game is regenerated in a fresh BattleStream and must reproduce the
input tape, public protocol, outcome and every actor input/action. The audit also
recomputes team partitions, file hashes, counters, scores and latency summaries.
The network smoke uses explicit development teams with the stock Random Battle
rules in a separately named, unrated local format; it tests the real server's
authentication, challenges, request IDs and complete battle transport. It does not
claim public-server login or live random-team generation coverage.

## Measured result — 2026-10-11

The accepted campaign is `artifacts/runs/playable-baseline-v2`, namespace
`playable-baseline-qualified-v1`: **64 complete games, 32 paired matchups, 60 wins
and 4 losses against the random baseline (93.75%)**. No draw, truncation or choice
rejection occurred. The saved 2,904 decisions include 575 switches and 44 Tera
choices across both players. All 64 games and every actor input/action reproduce
exactly in fresh simulator instances. Heuristic decision latency was 0.0224 ms
median, 0.0846 ms p95 and 1.2382 ms maximum on this local run; this excludes
network delay, data generation, server simulation and startup.

The accepted network run is `artifacts/runs/playable-baseline-network-v2`:
**three complete local-server games, 134 submitted choices, 158 observed requests,
zero client failures and zero unavailable-choice retries**. Both players agree on
each terminal result. It includes a same-connection rematch and a new-login game,
checks public-turn/type availability before actionable requests, and confirms that
emitted decisions use the latest received public turn. The local static landing
page is available; the browser UI and public login remain untested.

Twenty focused tests cover codecs, transport state, authentication with a mock
endpoint, ranked identities, request ownership, hidden trapping, Revival Blessing,
hidden-reserve invariance, Tera memory, Illusion replacement and pinned request
ordering. Three deliberately corrupted evidence copies (win score, actor input,
counter) were rejected by the replay audit. The full source-bound regression passes **284 tests (95 Node + 189 Python)**
with all 247 tested source files unchanged. The immutable source/evidence archive is bound by the
[completion receipt](../artifacts/reports/playable-baseline.json).

This is a small descriptive development sample against a deliberately weak policy.
It is not an Elo estimate, a trained-policy result, a full G2/G3 pass or evidence of
competitiveness against good human players. No reserved final team or final episode
was used. No GPU or Prime resources were used.

The [claude-review skill](/Users/maxf/.codex/skills/claude-review/SKILL.md) provided
independent protocol exploration and snapshot review. Verified review dispositions
are in `artifacts/reviews/playable-baseline/review-dispositions.md`. The protocol
research packet was rejected against its precise pinned-evidence criteria; useful
findings were independently checked against local source. The main review's
request-ordering concern was disproved for the pin. Its rematch concern led to the
reversed challenge-clear fix and a real network regression. Snapshot-to-final
changes and the absence of a second Fable approval are explicitly recorded.

Draft evidence remains preserved. The first draft used an incorrect stream cleanup;
a later draft incorrectly treated remaining Pokémon counts as win scores. Both were
corrected before the paired benchmark. The first 64-game production run completed
but failed replay because Illusion replacement introduced an undefined HP field;
that run is not qualified. Its replacement passes complete replay. One full-suite
attempt was interrupted after the replay failure was observed; only the final
passing source-bound regression counts.

Known deployment limitations: any server popup currently stops the session, and
challenges received during an active match are not queued. Automatic reconnect,
resume of an in-progress game, live ladder operation and full competitive policy
quality remain future work.

## Next work

Use this client as the deployment and evaluation boundary for stronger policies.
First test a chosen public account in a direct challenge after checking current
server requirements. Add disconnect recovery and a stronger frozen opponent panel
before a sustained ladder experiment. Continue the switching/hidden-set scalar/JAX
port separately; this playable baseline does not close its fidelity or learning gates.
