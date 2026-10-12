Done. Two files created, nothing else touched.

## Implementation — `packages/showdown-soak/schedule.mjs`

- Loads `plan.json` once via `readFileSync(new URL('./plan.json', import.meta.url))` and **deep-freezes** it, so the source can never be mutated (modules are strict-mode, so any write would throw).
- **`planFor(smoke=false)`** — `TypeError` on non-boolean. Destructures `smoke` out of the top level, spreads the smoke overrides on top when requested, `structuredClone`s the result (fresh arrays/objects every call), and adds `games = sessions * games_per_session`.
- **`gameSpec(index, smoke=false)`** — `RangeError` unless `Number.isSafeInteger(index)` and `0 ≤ index < games`. Returns exactly the 12 specified keys with the specified formulas; scenario is the first fault whose `fault_offsets[k] + (session % 2) === within_session`, else `'normal'`; file is `game-NNNN.json.gz`; names are `SoakTac${session}` / `SoakBase${session}`.
- **`expectedCoverage(smoke=false)`** — walks every index through `gameSpec` and derives: `rematches` = games with `within_session !== 0` (→ games − sessions), `faults` = non-normal scenarios, `tactical_connections = sessions + faults`, `baseline_connections = sessions`, seat counts, and a `scenarios` histogram keyed `normal` + each fault in plan order.

Only Node builtins (`node:fs`); no simulator or network imports.

## Tests — `tests/showdown-soak/schedule.test.mjs` (`node:test` + `node:assert/strict`)

- Production known answer: 1000 / 20 / 980 / 60 / 80 / 20 / 500 / 500, scenarios `{normal:940, before-choice:20, lost-write:20, accepted-choice:20}`; exact key set checked.
- Smoke known answer: 8 / 2 / 6 / 6 / 8 / 2 / 4 / 4, scenarios 2 each.
- Pair mapping: full `deepEqual` on indices 0, 1, 50, 999 (and smoke 7); loop over all 500 pairs asserting fixed teams/seats with swapped policies.
- Fault placement: even session → within 10/20/30 (p1), odd → 11/21/31 (p2); smoke sequence for all 8 games; every session has exactly the three faults in plan order.
- All six fault×seat strata: 10 each in production, 1 each in smoke.
- No mutation: `planFor` copies are independent (push/delete/reassign on one doesn't leak into the next or into `gameSpec`/`expectedCoverage`); `gameSpec` and `expectedCoverage` return fresh objects; on-disk JSON unchanged.
- Rejection: non-boolean `smoke` → `TypeError` for all three exports; `gameSpec` rejects negatives, ≥games, fractions, NaN/±Infinity, `2**53`, strings, null/undefined, booleans, arrays, objects, BigInt → `RangeError`; `-0` accepted as index 0.

## Limitations

- I had no shell tool in this session, so **I did not run the tests**; the controller's acceptance run will be the first execution. I've reasoned through the arithmetic (e.g. smoke session 1 faults at within 1/2/3 → indices 5/6/7 → p2/p1/p2) but a runtime check is still outstanding.
- `structuredClone` requires Node ≥ 17; no `package.json` exists to pin an engine, so I'm inferring that's acceptable.
- "Rematches" isn't defined in the plan; I interpreted it as "any game that isn't the first of its session," which is the only reading that yields 980/6. If the intended definition differs, that one counter is the place to change.
