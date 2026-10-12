**Provisional verdict: no correctness defects found in reviewed scope; two low-severity items (one operational, one test-strength).** Most important: a stale `.partial` file makes every subsequent `checkpoint.save` at that path fail (`open('xb')`), so a single interrupted checkpoint blocks all later checkpoints until manual cleanup.

## Findings

**1. Low — stale `.partial` blocks later saves** (`checkpoint.py:61-65`)
`temp.open('xb')` raises `FileExistsError` if `state.json.partial` exists. After any interrupted write (injected or real), the next `save(path, …)` to the same path fails. The test (`test_learner.py:104-105`) confirms the `.partial` survives but never attempts a subsequent save, so the gap is untested. Atomicity of the published file is correct (`os.replace` + dir fsync). Verification: call `save` twice with `interrupt=True` then normally; expect failure. Suggested fix: unlink stale partial before `open('xb')`, or use a unique temp name.

**2. Low — first-update bootstrap/value checks are vacuous** (`test_learner.py:85`, `policy.py:63`)
`assertGreater(int((buffer.bootstrap!=0).sum()), -1)` always passes. Moreover `vw`/`vb` initialize to zero, so under the initial parameters every stored value and bootstrap is exactly 0; the first-update `linked_gae` parity (`learner.py:130-133`) therefore exercises only reward propagation, not the γ·V_bootstrap cutoff path. The second-update assertion (`test_learner.py:92`) and the synthetic `targets` test (`:122-129`) do cover it, so coverage exists but the line-85 assertion should be removed or made meaningful.

**Verified as correct (no finding):**
- *Dyadic sampler vs stored/PPO logp:* `act` and `objective` both derive leaf logs from the same `nodes()` conditionals (`policy.py:40-47`, `:75`, `learner.py:92-94`); masks/features identically zeroed on dead lanes in both paths (`learner.py:76`, `:92`). Test `:21-45` checks threshold±1 words select each leaf and `exp(log)` matches the Fraction product under all 16 masks, including floored (|Δ|≥~16.6) nodes. Note: `act` and `objective` are separate jit programs, so epoch-1 ratios need not be bitwise 1 — not a defect.
- *Gradient masking:* invalid rows get logits 0, mask `[T,F,F,F]`, action 0, targets 0 → ratio 1, every mean re-masked (`policy.py:82-90`); `_group`/`binary_parts` zero-gradient on forced/empty nodes. Test `:60-66` asserts zero gradient with NaN padding.
- *Same-player GAE across waits:* reverse scan carries `(next_v, next_adv, next_i)` only on `live` rows (`learner.py:53-62`); hand-verified expected advantages in test `:128` (.59/.58/.4/.2) are correct for λ=.95, γ=1.
- *Terminal vs capped bootstrap:* `terminal = live&last&ended`, `boundary` only when `~ended`; `boot` zeroed for finished games (`learner.py:85`); capped games must be turn 13/phase 0 (`:125`); reward sum == finished games (`:135`). GROUPS=14 = 12 joint + ≤2 replacement groups is consistent with `_mask` allowing double-faint replacement in one group.
- *Observation-only actor:* `observe` → `view_kernel` → `encode(view[:-1])`; 323 = 18+4+8+5+288 ✓.
- *Counter linkage:* `update` advances version+1, step+epochs, next_game+BATCH (`:106`); `validate` enforces the invariant on save and load; budget checked in `run`/`transaction`.
- *Replay/rollback:* fingerprint of full `(candidate, buffer, final, metrics)` compared original vs resumed; fault injection returns before any state publication.

Research-validity note (not a defect): the inherited 2⁻²⁴ floor with zero derivative applies per node, so a leaf can saturate at 2⁻⁴⁸ with dead gradient at both levels; the bandit canary (`:69-81`) only requires >0.9, so this ceiling is untested but is sealed behaviour.

## Scope limitations
Not in packet: `pokemon_learner.core` (`linked_gae`, `ppo_loss_grad`, `require`), `pokemon_learner.checkpoint` (`read_array`, `wire`, `digest`), `pokemon_jax_learner.core._adam_kernel`/`model.forward`, `pokemon_showdown.mapping`, `scripts/verify-fixed-loop.py`, scalar engine/parity tests. No tests were run; no claim about actual 3×64×32 smoke outcomes.

## Verdict
Findings: two low-severity (stale `.partial` blocks re-save; vacuous first-update assertion). No verified correctness defects in sampler/logp parity, masking, GAE linkage, bootstrap logic, counters, or checkpoint replay within the reviewed files.
