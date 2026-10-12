## Result: design input for the first restricted Gen 9 1v1 slice

**Scope statement.** Fixed teams, one Rattata per side: Run Away, no item, Tackle + Quick Attack, level 50, neutral nature, 0 EVs, 31 IVs, no Tera. This is Stage B (restricted) per `docs/ACCEPTANCE_GATES.md:21`; it is **not** Gen 9 Random Battles and must not be labeled as such.

### 1. Profile ID and format realization

- **Profile ID (new, per `docs/R6_MECHANICS.md:111`):** `slice.gen9customgame.1v1.rattata-runaway-noitem-tackle-quickattack.l50-hardy-0ev-31iv-notera.v1`
- **Why not `gen9randombattle`:** the pinned format has `team: 'random'` (config/formats.ts), so supplied teams are ignored; Rattata is `isNonstandard: "Past", tier: "Illegal"` in data/formats-data.ts and is absent from all of `data/random-battles/gen9/sets.json` (checked all three fetched portions, first key `venusaur`, last `pecharunt`). Random Battle also uses `HP Percentage Mod`; this slice shows exact HP. Any artifact must carry the new profile ID and the format string, never the pinned `format_id` from `showdown.pin.json`.
- **Proposed format string:** `gen9customgame@@@Terastal Clause` (+ optionally `,!Team Preview,!Cancel Mod`). Evidence: Custom Game ruleset is `['Team Preview','Cancel Mod','Max Team Size = 24','Max Move Count = 24','Max Level = 9999','Default Level = 100']` with `battle: { trunc: Math.trunc }` and `debug: true`; `@@@` custom-rule parsing and `!` repeals exist in sim/dex-formats.ts; `terastalclause.onBegin` sets `pokemon.canTerastallize = null` for all Pokémon (data/rulesets.ts). Without `Obtainable`, team-validator does no learnset or `nonexistent` check, so a Past species at level 50 passes (`set.level` kept because `adjustLevel` is unset: `clampIntRange(set.adjustLevel || set.level || 100, 1, 9999)`).
- **Team set must be fully explicit:** species, ability, item `""`, moves, `level: 50`, `nature: "Hardy"`, `evs` all 0, `ivs` all 31, `teraType` irrelevant (clause), and **`gender` fixed** (I recall unset gender consumes a PRNG draw in the Pokémon constructor — unverified, see §9).

### 2. Stats (source: `battle.statModify`, fetched)

`tr(tr(2·B + IV + tr(EV/4) [+100 for HP]) · L/100 + {10 | 5})`, neutral nature → no 110/90 step.

| Stat | Base | Value |
|---|---:|---:|
| HP | 30 | tr(191·0.5+10) = **105** |
| Atk | 56 | tr(71.5+5) = **76** |
| Def | 35 | **55** |
| SpA | 25 | 45 |
| SpD | 35 | 55 |
| Spe | 72 | tr(87.5+5) = **92** |

Action speed: `getActionSpeed` → `trunc(getStat('spe'), 13)` = 92 for both (no Trick Room).

### 3. PP obligations

`calculatePP(move, ppUps=3)` = `pp·(5+3)/5` (neither move has `noPPBoosts`): **Tackle 56/56, Quick Attack 48/48**. `deductPP` subtracts 1 per use (no Pressure). Struggle is substituted only when `getMoves()` is empty (sim/pokemon.ts `getMoveRequestData`; sim/side.ts `!moves.length`).

### 4. Order

Priority is read from `dex.moves.get(move.id).priority` (+ `ModifyPriority` events, none here; `fractionalPriority` 0 with no item/ability): Quick Attack **1**, Tackle **0** (excerpt lines 14483, 18819). Quick Attack user always acts first vs. Tackle. Equal priority → speeds 92=92 → `comparePriority` returns 0 → `speedSort` calls `prng.shuffle(list, sorted, sorted+2)` → one `random(0,2)` draw → 50/50 (excerpt battle.ts:441-461; prng.ts `shuffle`). Gen ≥ 8 re-sorts the queue after each move (`updateSpeed`/`queue.sort`), a no-op for the single remaining action.

### 5. Accuracy

`hitStepAccuracy`: `accuracy = 100` is a number, not `true`, so `randomChance(100, 100)` executes → **a PRNG draw is consumed but the move can never miss**. No `ModifyAccuracy`/`Accuracy` handlers and no boosts exist in this slice.

### 6. Critical hits

`critRatio` defaults to `Number(data.critRatio) || 1` (sim/dex-moves.ts); `clampIntRange(1,0,4)=1`; Gen 9 `critMult[1] = 24` → `randomChance(1, 24)` = **1/24** (excerpt battle-actions.ts:1615-1632). `CriticalHit` event has no handlers. Crit multiplier `tr(x·1.5)` applied before the random roll (excerpt 1740-1743). Crit's boost-ignoring is moot (no boosts).

### 7. Damage arithmetic (excerpt 1709; `modify`, `randomizer`, `tr(…,16)` fetched)

`baseDamage = tr(tr(tr(tr(2·50/5+2)·40·76)/55)/50)` = tr(tr(66880/55)/50) = tr(1216/50) = **24**; `+2` → 26. Crit: tr(26·1.5) = 39.
Random: `tr(tr(x·(100−r))/100)`, r = `random(16)` ∈ {0..15} uniform. STAB (Normal on Normal): `modify(x, 1.5)` = `tr((tr(x·6144)+2047)/4096)` (round-half-down). Type mod 0, no burn, no `ModifyDamage` handlers, no Protect. Final `tr(x,16)` is identity here (and Custom Game's `Math.trunc` ignores `bits` — identical for these magnitudes; a divergence from `dex.trunc` only above 2^16).

| r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-crit pre-STAB | 26 | 25 | 25 | 25 | 24 | 24 | 24 | 24 | 23 | 23 | 23 | 23 | 22 | 22 | 22 | 22 |
| **non-crit final** | 39 | 37 | 37 | 37 | 36 | 36 | 36 | 36 | 34 | 34 | 34 | 34 | 33 | 33 | 33 | 33 |
| crit pre-STAB | 39 | 38 | 38 | 37 | 37 | 37 | 36 | 36 | 35 | 35 | 35 | 34 | 34 | 33 | 33 | 33 |
| **crit final** | 58 | 57 | 57 | 55 | 55 | 55 | 54 | 54 | 52 | 52 | 52 | 51 | 51 | 49 | 49 | 49 |

Non-crit support {33,34,36,37,39} with weights 4/4/4/3/1 of 16; crit support {49,51,52,54,55,57,58}. Both moves yield identical damage (same BP/type/category); they differ only in priority and PP.

### 8. Duration bound, Struggle, ties

- HP 105. Two non-crits max 78 < 105; crit+non-crit max 97 < 105 → **≥ 2 hits** (needs two crits with sum ≥ 105, e.g. 54+51). Three hits min 99 < 105 (not guaranteed); **four hits min 132 ≥ 105 → guaranteed KO.**
- Every move always hits and each side gets exactly one action per turn while alive, so a full-HP game ends on **turn ≤ 4**, with **≤ 7 move executions** and **≤ 4 PP deducted** from any one slot. PP floors are 48 → **Struggle is unreachable** in any full-HP game. The bound is strict and does not rely on excluding Struggle; Struggle merely needs to be fail-closed if it ever appears.
- **Tie unreachable:** `runAction` calls `faintMessages()` after each action, and `checkWin` calls `win(side)` when `!side.foePokemonLeft()`; a tie requires `sides.every(!pokemonLeft)` in a single faint pass. Damage here is sequential with no recoil/residual, and a fainted Pokémon's move is skipped (`if (action.pokemon.fainted) return false;`) while `turnLoop` exits on `this.ended`. Winner = side landing the KO.
- Hand-computed check value (my arithmetic, to be confirmed by the enumeration fixture): P(KO within 3 non-crit hits) = 2384/4096 ≈ 0.582 → first-mover win ≈ 1 − 0.418·0.582 ≈ 0.76 ignoring crits. So Quick Attack weakly dominates Tackle; mirror QA/QA is a 50/50 speed tie. The slice is strategically trivial by design (a learner canary: policy should put mass on Quick Attack).

### 9. Semantic chance-event strategy

Raw-seed coupling is unsound: besides the four meaningful draws, the reference consumes PRNG in `eachEvent('Update')` (speed-tied `speedSort` after every action, gen ≥ 5), in `runEvent` handler sorting, in 100%-accuracy rolls, and possibly in gender assignment; the default RNG is `SodiumRNG` (prng.ts `generateSeed`). Define per-event semantic outcomes, extracted from reference protocol rather than from the seed:

| Event ID | When | Outcome | Reference evidence |
|---|---|---|---|
| `order.tie{turn}` | both chose same priority | permutation of seats | order of `|move|` lines |
| `acc{turn,seat,move}` | every move | forced `hit` (p=1) | no `|-miss|` |
| `crit{turn,seat,move}` | every hit | bool, p=1/24 | `|-crit|` |
| `roll{turn,seat,move}` | every hit | r∈0..15 ↔ damage value | `|-damage|` HP delta (exact HP in Custom Game) |

Engine side: inject these outcomes and compare HP/PP/order/winner exactly (G2 "forced random outcomes"); separately enumerate the finite outcome tree (≤7 hits × 32 branches, plus ≤4 tie bits) to produce exact win probabilities. Reference side: predeclared distribution tests (crit rate, 5-point damage distribution, 50/50 order) with a family-wise error policy. Note `battle.forceRandomChance` forces accuracy *and* crit together, so it cannot isolate crits.

### 10. Fail-closed configurations (abort before sample collection)

Request contains `canTerastallize`, `struggle`, a `trapped`/switch option, `maybeTrapped`; protocol contains `|-miss|`, `|tie|`, `|-boost|`, `|-status|`, `|-weather|`, `|-fail|`, `|cant|`, `|-heal|`, `|detailschange|`, `|-terastallize|`; any species/ability/item/move outside the manifest; HP shown as a percentage; damage value outside the two tables; move count ≠ 2; level ≠ 50; game past turn 4; Team Preview or `undo` traffic if the corresponding rules are repealed. `|debug|` lines (Custom Game `debug: true`) must be explicitly stripped by normalization, not ignored silently.

### 11. Planned fixtures

Edge cases (both seats each): stats via `|switch|` 105/105 and request `stats`; PP 56/48 and decrement; QA-before-Tackle regardless of seat; QA/QA tie 50/50; damage ∈ tables with `|-crit|` iff crit; exact-105 KO (54+51); KO turn shows no `|move|` for the fainted second mover, then `|faint|`,`|win|`, `requestState=''`; min game (2 crits, 2 turns) and max game (4 turns, 7 executions); never `|tie|`; Tera choice → choice error and masked (class 1, not hidden legality); bad slot/move id → error, no PP consumed; Team Preview/Cancel fixtures or repeal assertions depending on the chosen rule string. Differential: semantic-replay parity, outcome-tree enumeration vs. reference distribution tests, and a 10,000-game independent sample for G2 (note the hidden-state mutation suite is near-vacuous here: sets are fully known; only opponent PP is unshown and is inferable).

### 12. Known limits, missing context, blockers

- The supplied excerpts omit the stat/PP formulas, `randomizer`/`modify`/`trunc`, the accuracy step, faint/win ordering, format definitions and PRNG. I filled these from the pinned commit on GitHub via a summarizing fetch tool; quotes are secondary extractions and **must be re-verified against a vendored tree whose SHA256s match the excerpt headers before being used as parity evidence**. No vendored Showdown exists in this workspace.
- Unverified (recalled, not sourced): random gender draw when `gender` is unset; neutral-nature data entry; default EV/IV fill in the Pokémon constructor; exact semantics of `!Team Preview` repeal when applied as a custom rule over the base format's own ruleset; count of `Update`-event shuffles per turn. Fallback if the repeal does not take effect: keep Team Preview and add `team 1` phase fixtures.
- `Max Move Count = 24` / level handling were only partially read (team-validator fetch truncated at 100 kB).
- Hand arithmetic in §7–§8 was done by me, not executed; the enumeration fixture is the check. This document is design input, not parity evidence; nothing was implemented or run.

Sources (pinned commit 942cda71…): https://raw.githubusercontent.com/smogon/pokemon-showdown/942cda71d14bf0618bd57991708a2e2734b6daba/sim/battle.ts · …/sim/battle-actions.ts · …/sim/pokemon.ts · …/sim/dex-moves.ts · …/sim/prng.ts · …/sim/dex.ts · …/sim/dex-formats.ts · …/sim/side.ts · …/sim/team-validator.ts · …/config/formats.ts · …/data/rulesets.ts · …/data/formats-data.ts · …/data/random-battles/gen9/sets.json
