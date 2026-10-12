**Provisional verdict: findings (minor/medium), none blocking on the evidence shown.** Most important: the receipt nickname rule in `teams.mjs:35` is only self-validated, and the server-side allowlist depends on an unrequired `Teams` global that the test sandbox masks.

## Findings

**1. Medium — Receipt nickname rule is asserted circularly.** `teams.mjs:35` rewrites `name` to `baseSpecies` when `!set.name || set.name === set.species`. The test (`format.test.mjs:47-50`) reapplies the same expression, so it proves nothing about the simulator's actual behavior; the pinned-format evidence file covers `validateTeam`, not `Pokemon`/`Side` name defaulting. The 8-game smoke passing `audit.mjs:65` is the only real evidence, over 16 receipts. In the 1,000-team pool, any set where `name === species` but `baseSpecies` differs (forme species with generator-emitted name equal to the forme) would fail audit as a "Received simulator team" mismatch — a false campaign failure, not a correctness hole, but it stops the soak. *Verification:* quote the pinned `sim/pokemon.ts` name-defaulting lines into `pinned-format-evidence.md`, and grep the regenerated pool for sets with `name === species && species !== baseSpecies`.

**2. Medium — `validateTeam` relies on a bare `Teams` global.** `local-server.mjs:13` emits `allowed.has(Teams.pack(team))` with no `require`. In `format.test.mjs:15` the sandbox injects `Teams`, so the test cannot detect a missing global. If validation runs in a validator subprocess (the settings use `subprocesses: 0`, which is not the standard `validatorprocesses` key, so its effect is unverified from this packet), `Teams` may be undefined there → ReferenceError → validator failure. Smoke passing suggests it currently resolves, but the dependency is implicit. *Verification:* emit `const {Teams} = require('../sim/teams');` in `formatSource`, and confirm which setting disables validator subprocesses in the pinned `config-example.js`.

**3. Low — Audit binds the file, not the loaded format.** `audit.mjs:200` checks `custom-formats.js` bytes equal `formatSource(teams)`, and `audit.mjs:105` checks `tier === DEV_FORMAT`, but nothing confirms the server loaded `dist/config/custom-formats.js` rather than a same-named entry in copied `config/formats.js`, nor that `DEV_FORMAT` (from `protocol.mjs`, not shown) equals `developmentFormat().name`. *Verification:* assert the server log contains a format-load line or that `developmentFormat().name === DEV_FORMAT` in audit.

**4. Low — Rules parity test covers the rule table only.** `format.test.mjs:36-43` compares `getRuleTable`, `mod`, `gameType`. Non-ruleset battle hooks on the official format (`onBegin`, `teraPreviewDefault`, etc.) are not compared; `developmentFormat` copies only `ruleset`. Likely fine for gen9randombattle, but unverified. *Verification:* diff `Object.keys(Dex.formats.get('gen9randombattle'))` against copied fields.

**Allowlist identity — no finding.** Allowlist uses `pack(unpack(pack(team)))`; server-side validation receives `unpack(upload)` and re-packs, matching canonical form. Upload check (`audit.mjs:89`) uses single `pack(team)`, which is the correct client-side wire form; the normalization report's "first roundtrip canonicalizes move names" is consistent. Length-6 check compensates for skipped `baseValidateTeam` size rules. Mutation test (`format.test.mjs:28-35`) covers level, IV, item, move, order, size.

## Scope limitations
- No vendor source (`pokemon-showdown` `room-battle.ts` `getTeam`, `pokemon.ts`, `dex-formats.ts` load path, config keys) in the packet.
- `protocol.mjs`, `development.mjs`, `schedule.mjs` not supplied; `DEV_FORMAT` identity and team-key/namespace derivation unreviewed.
- Smoke/normalization results taken as reported, not executed.

## Verdict
Findings — no identified defect in allowlist identity or set-value preservation; two medium evidence gaps (receipt nickname rule, implicit `Teams` global) should be closed before committing 1,000 games.
