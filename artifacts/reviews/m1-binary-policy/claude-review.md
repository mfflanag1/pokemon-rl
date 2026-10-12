**Provisional verdict: no blocking findings in reviewed scope; two low-severity items and one dependence worth verifying.** The integer mapping, boundary handling, and reference oracle are consistent with the contract as written.

## Findings

**1. Low — `reference.py:102-105` forced-row `rare_index` semantics diverge from `policy.py:21-23`.**
For one legal action the oracle returns `rare_index = only` with `rare_mass = 0`; the kernel's `parts` returns `rare_action = 1` (since `safe` is zeroed → `delta = 0 → d>=0` branch) and `rare = 0.5`. `test_reference.py:39-43` compares only `probabilities`/`logp` on these rows, so nothing fails, but the oracle is not a reference for the transport fields on forced/terminal rows. Either document that `rare_index`/`rare_mass` are undefined for `legal < 2`, or align. Not a law mismatch (sampler/log path for forced rows is mask-driven in both).

**2. Low / verify — tie-branch gradient depends on JAX's `abs` JVP at 0.**
`contract.json:10` ("at equal logits use the d>=0 branch"), `reference.py:136`, and `test_policy.py:50-58` (d=0, expected `[0.5,-0.5]`) all assume `d|d|/dd = +1` at `d=0`. This holds iff `lax.abs` lowers its JVP as `select(x>=0, g, -g)` (current JAX behaviour as I understand it) rather than `g*sign(x)`. If a JAX upgrade changes this, the policy silently moves to a zero-gradient tie branch while the oracle does not. Suggested verification: `jax.grad(jnp.abs)(jnp.float32(0.))` pinned in a test, or implement the branch explicitly (`jnp.where(delta>=0, -delta, delta)`) so the contract does not depend on library convention.

**3. Info — oracle independence is partial.**
`dyadic_threshold` (Fraction/`as_integer_ratio`) and `selected_logp_derivative` are genuinely independent of `threshold_words`/autodiff. `_rare_mass` (`reference.py:84-90`) mirrors the kernel's `exp(-|d|)/(1+exp(-|d|))` formula, so the FP32 value tests mainly check NumPy-vs-XLA `exp` agreement under tolerance, not an independent derivation of the mass. Acceptable for the stated scope; do not cite it as an independent check of the FP32 value itself.

## Verified by reading (no issues)

- **Integer mapping** (`policy.py:39-52`): `rare = m·2^(e−150)`, so `rare·2^64 = m·2^(e−86)`; for rare ∈ [2^-24, 1/2], e ∈ [103,126] ⇒ shift ∈ [17,40]. Branch `shift<32`: `high = m>>(32−shift)` (shift count 1..15), `low = (m<<shift) mod 2^32` — correct low word. Branch `shift≥32`: `high = m<<(shift−32)` (≤ 24+8 bits, fits), `low=0`; at shift=40, m=2^23 gives exactly 2^63 for 0.5. Shift counts are clamped so no UB. `raw ≤ 0.5` always (tail ≤ 1), floor via `where(raw > FLOOR,…)` gives exact zero derivative at/below floor, matching `reference.py:134`.
- **Boundary semantics**: both sides choose rare iff `word < threshold` (`policy.py:58-59`, `reference.py:65`); equality → common. Test covers `{0, t−1, t, t+1, 2^64−1}` over all admitted exponents incl. 2^-24 and 0.5. Hand-checked `test_mirrored_tails` expectations `[0,1,1,1,1,0,-1]` / `[1,0,0,0,1,0,-1]` — correct.
- **Log/gradient consistency**: sampler threshold, `log(r)`, `log1p(-r)` all consume the same FP32 `rare`; `fl(1−r)` is only a rendering. Reference derivative `dr/dd = ∓r(1−r)`, `∂log r = dr/r`, `∂log(1−r) = −dr/(1−r)`, logit1 = −logit0 — correct. Remaining differences are ordinary FP32 rounding, not a law mismatch.
- **Masking/leakage**: `safe` zeroing kills logit gradient on forced/padded rows; `masked_loss` sanitizes all inputs before arithmetic; dead rows yield action −1, logp 0, value 0; `act` host-checks legality.
- **Checkpoint isolation**: distinct format string, provenance hashes `pokemon_binary/*.py` + contract dir, plan equality, canonical re-serialisation check, counter bounds (4·u ≤ step ≤ 8·u consistent with 2–4 nonempty minibatches × 2 epochs).

## Scope limitations

Cannot run tests. Did not review `pokemon_jax.engine`, `pokemon_fixed.math.adam_step`, `pokemon_jax_learner.collector`, or the sealed loss for parity claims. Claim that finished games yield an all-false mask (required for the `legal` check) is assumed, not verified. No assessment of the "nominal 1/24 chance sampler" remark.
