**Provisional verdict: findings (no consequential validity break found; one correctness bug in the Illusion correction and one unverified binding).**

## Findings

**1. Medium — Illusion `replace` attributes the disguise's prior memory to the revealed mon** (`view.mjs:19-27`)
On `switch`, the disguised mon is populated with `structuredClone(memory)` of the disguise ident (line 31). On `replace`, `previous.moves`/`previous.ability`/`previous.item` are merged into the revealed ident's memory (lines 22–24). If the real Squirtle had earlier revealed Water Gun/Torrent, those now become Zoroark's "observed" moves/ability. Only the delta vs `previous.memoryAtEntry` should migrate. The test at `policy.test.mjs:42-52` uses a fresh Squirtle with empty memory, so it does not exercise this. Verification: pre-reveal `|move|p2a: Squirtle|Water Gun`, switch out/in, boost, Dark Pulse, `replace` → assert `known['p2a: Zoroark'].moves` equals `['darkpulse']` only. Affects opponent quality, not replay determinism.

**2. Medium (unverified) — source binding depends on `source_manifest()` covering the new panel files** (`seal-showdown-panel.py:15,35,57`)
`verify-fixed-loop.py` is not in the packet. If its manifest omits `packages/showdown-panel/*`, `tests/showdown-panel/*`, or `scripts/seal-showdown-panel.py`, then `registration.provenance == now`, `v['source_files'] == now['source_files']`, and the audit's fresh replay (which uses live `playGame`) are unbound to frozen source. Verification: print `source_manifest()` keys and confirm all panel sources appear.

**3. Low — audit asserts literals as evidence** (`audit.mjs:35-36`)
`final_teams_used: 0` and `pairing_seeds_policy_seats_and_intervals_verified: true` are hardcoded; the seal (`:52-53`) requires them as if computed. Seats/intervals are in fact covered by the `summarize` deep-equal and `gameSpec` replay, but `final_teams_used` is not checked anywhere in the packet (depends on `developmentTeams` namespacing, not shown).

**4. Low — protocol event shapes tested only with hand-written lines** (`view.mjs:53-61`, `policy.test.mjs:54-63`)
The `-copyboost` recipient-first ordering and `-swapboost` stat-list parsing are asserted against literal strings, not pinned engine output. Verification: run Psych Up / Power Swap through the pinned `BattleStream` and ingest the actual spectator lines.

**5. Low — seal/campaign gaps**
- `report.mjs` `score: game.outcome.score[mirror]` is consistent between campaign and audit (policy `a` sits at seat `mirror`), but tie semantics and `summarizeMatrix`'s use of `a/b/score` are out of packet.
- No check that `failure.json` is absent or `progress.json.completed == 768`; `wx` on `report.json` makes this mostly safe.
- Node threshold `>= 122` is loose given 114 prior + 10 new tests.
- `policy.mjs:173-174`: `best` is undefined if no `move ` choice exists outside `forceSwitch`; would throw and invalidate the run (safe direction).
- `view.mjs:66`: `|-weather|none` leaves `weather='none'` (harmless). Traced abilities (`-ability ... [from] ability: Trace`) persist in memory across switches (approximation, undocumented).

## Policy corrections verified by inspection
- `accuracyStage` 3-based formula correct; applied only when `accuracy !== true` (`policy.mjs:13,84-86`).
- Entry hazards charged only with `entering=true`; active uses `false` (`:119-120,175`).
- Bench `enteredTurn = view.turn` makes Fake Out/First Impression scorable (`:100,93`); turn arithmetic for post-switch first turn checks out.
- Selective clears/invert/`-clearallboost` handled; `replace` preserves boosts/enteredTurn.

## Scope limitations
Not in packet: `game.mjs`, `statistics.mjs`, `protocol.mjs` (PublicBattle ident format, `condition` updates), `development.mjs`, `verify-fixed-loop.py`. Tests claimed passing are not executed here; all above is inspection.

**Verdict: findings** — repair #1 and verify #2 before freezing source.
