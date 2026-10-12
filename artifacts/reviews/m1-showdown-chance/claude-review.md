**Provisional verdict: findings (none blocking in the mapping itself; two medium evidentiary gaps).** Most important: the report asserts `final_reservation_unchanged=True` and "unchanged binary actor/update loop" without any check or cross-comparison against the predecessor's collector.

## Findings

**1. Medium — Unverified overclaims written into the sealed report.** `campaign.py:178` sets `result['final_reservation_unchanged']=True` unconditionally; nothing reads or hashes the reserved final-evaluation artifact. Similarly `player_parity` (`:138-139`) hard-codes only two prior namespaces (`m1-slice-reference-v1`, `m1-g2-player-reference-v1`) for the "disjoint from prior reference seeds" claim; if the binary-policy predecessor captured its own namespace it is not excluded. *Verify:* add a hash/stat check of the sealed final artifact before setting the flag; derive excluded namespaces from the predecessor receipt rather than a literal list.

**2. Medium — "Unchanged actor/update loop" is only partially true.** `pokemon_showdown/loop.py:24-71` re-implements `collect` and `cycle` (copied, not parameterized) and only imports `update`, `encode`, `gae`, `action_kernel` from `pokemon_binary.loop`. No test compares the new `collect` against `pokemon_binary.loop.collect` with chance injected; `test_complete_collector_uses_new_chance` re-derives the same author's reading (same `event_key` wiring, same stacking) rather than an independent oracle. `checkpoint.py:64` describes the actor stream as `ACT/game/turn/seat/DYB1`, a constant that does not appear in the new collector — possible stale provenance string. *Verify:* diff `collect`/`cycle` against `pokemon_binary/loop.py`; add a test that runs the predecessor collector with a monkeypatched sampler and asserts leaf-exact equality.

**3. Low — Word-oracle artifact is self-sealed only.** `word_audit` (`campaign.py:114-131`) checks the boundary-word list and internal consistency but cannot re-derive the 100,000 sodium probe words; `oracle['sources']` covers only `word-oracle.mjs` and `contract.json` (not `engine.mjs`/`prng.ts`, `word-oracle.mjs:31`). No oracle hash is pinned in the contract. A tampered-but-consistent file passes. *Verify:* have the campaign regenerate the oracle via Node (as `admission()` already does for `describe()`) or pin `word_content_hash` in `contract.json`.

**4. Low — Tie chain consistency not checked.** When tied moves are shuffled twice, `verify_case:93-96` validates each shuffle in isolation (`result == order[w>>31]`) and that the last governs `events[0]`, but never checks the second shuffle's recorded `order` equals the first shuffle's output order. Not required by the finite-word claim; worth an assertion to detect instrumentation errors.

**5. Informational — accuracy draws.** `hitStepAccuracy` calls `randomChance(100,100)` for 100-accuracy moves, consuming an untraced word; `index` gaps are expected. Consistent with "no consumed-word schedule" scope, but the contract text should state it so gaps aren't mistaken for lost draws.

**Mapping checks that pass on inspection:** the `bucket24` 16-bit split proof (`mapping.py:24-50`) is correct and all intermediates <2^21; `178956971 = ceil(2^32/24)` matches `prng.ts:99` with exact doubles; `w>>28`, `w>>31` match `random(16)`/`shuffle` (`prng.ts:150-158`, swap index = `start+(w>>31)`); `tie0/tie1` selection by captured `order` (`campaign.py:95`) matches `first XOR (w>>31)`; role layout in `from_words` matches the contract; 180,000 view-comparison total is consistent with 18/battle.

## Scope limitations
Not in packet: `pokemon_jax.chance.event_key`, `pokemon_binary.loop`, `PlayerProjection`, `engine.mjs`/`describe`, `battle-actions.ts`/`battle.ts` (crit/randomizer/speedSort call sites assumed per upstream), `digest` vs `seedFor` canonicalization, fixtures, and the running campaign's output. I did not run anything.

## Verdict
Findings — no demonstrated correctness flaw in the finite-word mapping or capture passivity; the report overstates two unverified properties (#1, #2) and should be tightened before the artifact is cited.
