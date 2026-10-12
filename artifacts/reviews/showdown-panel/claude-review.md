**Provisional verdict: no consequential findings in reviewed scope** (low-severity heuristic/public-state inaccuracies only; no actor information leakage found).

## Findings

**1. Low — Illusion contaminates the disguise ident's memory** (`view.mjs:27-30`, `19-22`)
Moves used while disguised are pushed to `known['p2a: Squirtle']`. After `|replace|p2a: Zoroark|…`, Zoroark correctly inherits them, but the real Squirtle later switching in also carries e.g. `darkpulse`. Public-information ambiguity, not a leak, but it is a wrong "observed" fact. Verification: extend the Illusion test with a subsequent `|switch|p2a: Squirtle|…` and assert `moves` is `[]`.

**2. Low — unhandled stage events leave stale boosts** (`view.mjs:36-45`)
`-clearnegativeboost` (White Herb), `-clearpositiveboost` (Spectral Thief), `-copyboost`, `-swapboost`, `-restoreboost` are ignored; `switch` after Baton Pass drops boosts that actually carry. Stale stages feed `effectiveStat`. Verification: ingest `|-clearnegativeboost|p2a: X|[silent]` after an `-unboost` and assert `boosts` is cleared.

**3. Low — entry cost charged to the mon already on the field** (`policy.mjs:173`, `116-119`)
`switchValue(own, …)` subtracts `entryCost` for the current active, which is not entering. With hazards up both sides of the comparison are penalized, so the net bias is small, but the `current + 0.3` threshold is computed from an inconsistent quantity. Verification: unit test `switchValue` on active vs bench with Stealth Rock up; current should not pay the cost.

**4. Low — Fake Out/First Impression always scored 0 for switch candidates** (`policy.mjs:92`, `96-99`)
Bench mons built by `ownMon(..., active=false)` have no `enteredTurn`, so `view.turn > -1` zeroes these moves even though a fresh switch-in could use them next turn. Deterministic, mild undervaluation only.

**5. Info — weather sentinel** (`view.mjs:46`): `|-weather|none` stores `'none'` rather than `''`. Harmless since only `raindance`/`sunnyday` are compared.

**Leakage check (no finding):** actors ingest only `players[side].protocol` (channel 1/2 via `extractChannelMessages`); `|split|` throws; `foe` has no `stats` so `estimate` always falls to species proxies; `effectiveness`/`effectiveStat` consult only revealed `ability`/`item`; `end` payload never reaches actors; `spectator` is not passed. `structuredClone` isolation plus the mutation-snapshot test hold. The `|request|` line arriving before the turn's `update` is harmless because `game.mjs:22` drains the full protocol before choosing.

**Legal-action handling (no finding):** all returned choices originate from `legalChoices(request)`; `game.mjs:29-34` re-validates and tolerates only own-side `[Unavailable choice]` with a refreshed request (trapped/disabled re-choice works). Note rejected attempts are still pushed to `decisions` and count toward `maxDecisions`.

**Paired protocol (no finding):** same mechanics seed/teams per pair, policy seats swapped, `score[mirror]` consistently records `a`'s result, common-index pair bootstrap, no stopping/tuning hooks. Supports only descriptive development claims, as the plan states.

## Scope limitations
- `report.mjs` (`counters`/`countGame`/`report`), `developmentTeams`, `makePolicy`, `seal-showdown-panel.py`, `extractChannelMessages`, and the sealed `showdown-bot/campaign.mjs` were not supplied; parity and matrix aggregation are taken from the described tests.
- I did not run tests or the engine; PS protocol event shapes are from memory.
- Hidden-reserve test only covers the first decision.

## Verdict
No consequential errors in reviewed scope; four low-severity heuristic/public-state inaccuracies (Illusion memory pollution, unhandled stage events, entry cost on active, Fake Out for bench) that do not affect leakage, legality, or the descriptive pairing design.
