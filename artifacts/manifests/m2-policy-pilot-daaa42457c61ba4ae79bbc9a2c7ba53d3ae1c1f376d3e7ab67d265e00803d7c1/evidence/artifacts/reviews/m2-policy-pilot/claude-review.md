**Provisional verdict: no blocking findings in reviewed scope.** Most important observation: the audit's "exact replay" (`campaign.py:117`) is a strict float-equality check across two separate XLA/JIT sessions; it is correct but makes the whole run invalid on any platform-level FP drift (e.g., sealing on a different CPU). Documented as intended ("implementation failure invalidates that run"), so not a bug — flagged as a consequential risk.

## Findings

**1. Medium — Tape-audit strictness vs. portability (campaign.py:117, seal-policy-pilot.py:53–54).**
`a==r['decisions'][turn]` compares `value`, `behavior_logp`, `quick_attack_probability` bit-for-bit, and the sealer re-runs `analyze` and requires equality with the persisted report. Correct for same-host sealing; no tolerance or host binding is recorded. Verification: run `seal-policy-pilot.py --audit-smoke` on the exact host/JAX build used for collection and record that constraint in the receipt.

**2. Low — Final-roster check skipped on loop exhaustion (campaign.py:71–80).**
If all 4 steps execute, `frame` from the last `step` is used without re-asserting `[f['id']...]==ids`; a roster mismatch would surface as `KeyError` rather than a `require` message. Fails safe, but diagnostics are poorer. Suggest moving the roster `require` after the loop as well.

**3. Low — `verify_record` does not check terminal turn equals decision count (campaign.py:95–103).**
A tape with a dropped *last* decision fails via `IndexError` at line 100 (hit turn > `len(decisions)`), not a `ValueError`. Still invalidates the run; add `reduced[0]['header'][0]==len(decisions)` for clarity.

**4. Low — Hard-coded regression expectations (seal-policy-pilot.py:21, 67).**
`SUITES` and `node_pass_count(log)==68` must match the actual suite. Only 7 of the new tests are reported run; no evidence in the packet that a full `npm-test.log` with these counts exists. Not a correctness bug; the seal simply won't complete until produced.

**5. Info — Statistics usage consistent with plan.**
Family of six = 3 seeds × 2 adjusted lower bounds; `alpha/6` passed per seed (campaign.py:147–148); `mean>=0.6` is a point criterion, not an inferential bound. Diagnostics use nominal alpha and are labelled as such. With 4096 replicates the adjusted tail holds ~17 replicates (two-sided) — coarse but acceptable; already disclaimed in `interpretation`.

**Verified as correct (reasoning only):** actor/key addressing (`seed, opponent, pair, seat, turn, role`; random-opponent domain `0x52414e44` disjoint from sampler `DOMAIN`); candidate row = `2*i+seat`; actions ordered `[p1,p2]` matching `history` actor encoding from p1's view; worker alternates commit order with `>p${first+1}`/`>p${2-first}` (both branches correct); views unchanged after first commitment; `admit`-gated teams; smoke/dev seed namespaces disjoint; candidate/baseline pairing shares mechanics seed and random-opponent keys across checkpoints.

## Scope limitations
- `statistics.py`, `provenance` dependencies (`sealed`, `checked_file`, `soak_admission`), `Runner/Executables`, `PlayerProjection`, `engine.mjs` (`checkSeed`, `describe`) not in packet. In particular, whether `'sodium,'+sha256hex` (64 hex) is accepted by `checkSeed`/Showdown's SodiumRNG is unverified here (Showdown pads to 64 hex, so likely fine, and the m1 soak presumably used the same convention).
- Cannot confirm `|turn|` precedes `|request|` in the player channel beyond the passing prefix test on the fixture.
- No production or smoke run output observed.

## Verdict
**No blocking findings in reviewed scope.** Items 1–4 are robustness/clarity issues; none invalidates implementation correctness or the stated statistical claims. Recommend producing a smoke run plus `--audit-smoke` seal on the collection host before the development run.
