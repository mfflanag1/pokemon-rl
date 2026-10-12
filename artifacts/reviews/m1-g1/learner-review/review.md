**Provisional verdict: no blocking findings in reviewed scope; several low-severity gaps.** Most important: the delayed canary's two perspectives share no outcome, so it cannot detect a zero-sum "opponent reward = −own reward" assumption the gate text implies it guards against.

## Findings

**1. Low (research validity) — `trainer.py:60-77`, `ACCEPTANCE_GATES.md:42`.** The "delayed" task is two independent delayed bandits sharing a tape; rewards at positions 3 and 5 depend only on each player's own root choice. A sign-flip of p2's reward would be caught, but a bug deriving p2's reward from p1's outcome (the actual sign-inversion hazard) has no observable consequence here because there is no shared winner. Claim "learns both perspectives without winner/loser sign inversion" is only partially evidenced. Verification: add a canary where one terminal event produces per-player rewards that are not negatives of each other (e.g., both +1, or asymmetric magnitudes) and assert both perspectives learn.

**2. Low (gate coverage) — `trainer.py:45-86`, `core.py:40-41`.** Buffer cuts never occur in the trainer (whole episodes, `bootstrap` always zeros). The gate "rollout truncation bootstraps from the correct perspective" is exercised only by the hand case in `test_core.py:9`, where the caller supplies the bootstrap; `linked_gae` cannot verify whose value it is. Acceptable for G1 scope, but the claim should be labeled "caller-supplied; untested end-to-end".

**3. Low (missing assertion) — `trainer.py:96-102`, `tests/`.** No trainer-level test asserts `diagnostics[0]['max_ratio_error'] <= 1e-5` on the first minibatch of the first epoch (frozen behavior == learner at update start). `test_same_policy_ratios` covers `core` only. Cheap to add.

**4. Low (recovery robustness) — `checkpoint.py:61-65`.** `temp.open('xb')` means a stale `.partial` from an interrupted write makes every subsequent `save` to that path raise `FileExistsError` (not a `ValueError` via `require`). `test_resume.py:62` only asserts the partial exists. After rollback recovery, the resumed job cannot checkpoint to the same path without manual cleanup. Suggest explicit detection with a clear message or documented operator step.

**5. Info (semantics, not error) — `trainer.py:94`, `core.py:84`.** With 384 delayed decisions and minibatch 256, the second minibatch has 128 rows; `ppo_loss_grad` averages per minibatch, so those samples receive 2× per-sample weight. Standard PPO quirk; "mean clipped loss" is per-minibatch. Checkpoint step accounting at `checkpoint.py:99` correctly uses ceil.

**Verified by inspection (no issues):** active-set logic `core.py:89` matches ∂min(rA, clip(r)A) for both signs; entropy gradient `p(logp+H)` sign/coef correct; masked rows get exactly zero probability and zero gradient; terminal forbids bootstrap; forward-link and same-trajectory checks; Adam bias correction and first-step test (±lr·sign) correct; sampler cannot select zero-probability actions; stale-buffer and phase gating; checksum/provenance/counter consistency on load. Both hand GAE cases recompute correctly (second: deltas .16, .24, .6 → A .3895, .51, .6).

## Scope limitations

- No executed tests or canary outputs in packet; the 3-seed ≥99% result is unverified.
- Central-difference test tolerance (5e-5 at ε=.002 in FP32) is tight; cannot confirm it passes.
- Checkpoint exactness is same-runtime only, as declared; cross-platform not reviewed.
- No simulator/Pokémon components; review limited to tabular learner.

## Verdict

**Findings (low severity only).** No analytic-gradient, GAE, mask, frozen-behavior, or checkpoint correctness errors identified. Gaps are in evidence coverage (items 1–3) and recovery ergonomics (item 4).
