**Provisional verdict: findings (no blockers to the bounded 60-s claim; one evidence gap in the sampling audit, several unchecked linkages).**

Most important: the random probe in `sampling_audit` never draws exponent 126, so the mass range [0.25, 0.5) — the regime actually hit for |d| < ln 3 — is covered by a single hand-added value (0.5), while the recorded proof string claims coverage of e=103..126.

## Findings

**1. Medium — sampling probe excludes exponent 126 (`qualification.py:58-59,76`)**
`rng.integers(103,126,…)` has an exclusive upper bound, so random rows have e∈[103,125] only. The `proof` text asserts "e=103..126; shift17..40", and the sealer accepts `rare_mass_cases==n`. Masses in [0.25,0.5) (e=126, shift 40) get exactly one probe (0.5, the boundary). Verification: change to `integers(103,127)` or add explicit e=126 rows, and have the sealer check the exponent histogram rather than only the row count.

**2. Low–Medium — dyadic proof string is not itself conditional (`qualification.py:76`)**
The adjacent field `uniform_word_assumption: True` and contract line 9 carry the conditionality, but the `proof` sentence states unconditional selection semantics. Not wrong (it is about integer comparison, not mass), but a reader of the receipt could lift it as an unconditional mass claim. Suggest appending "mass = threshold/2^64 only under uniform word pairs".

**3. Low–Medium — parity does not independently check the policy (`qualification.py:86,102`)**
The "sealed Python-control collector" receives `actor_fn=act` imported from the same `.policy` module the JAX collector uses. Parity therefore validates environment/GAE/fused-vs-split update, not the binary policy math; only the threshold oracle (`dyadic_threshold`) is independent. The reference string partially acknowledges this but the receipt should say so explicitly.

**4. Low — soak numerics recorded but unbounded by the sealer (`qualification.py:158`, `seal-binary-policy.py:53-59`)**
`minimum_exp_behavior_logp` (actually `minimum_behavior_probability`) and `maximum_update_ratio_error` are never checked against the contract floor 2^-24 / any bound. A soak violating the law would still seal.

**5. Low — hard-coded suite roster cannot be tied to new tests (`seal-binary-policy.py:87`)**
`[7,6,15,13,10,15,13,6,25,17,16,11]` and node=63 are asserted but nothing shows which suite exercises `pokemon_binary` or the fault-injection paths in `checkpoint.py:34-50`. If the new tests live in an existing file, the literal must have been bumped; the packet cannot confirm.

**6. Low — `peak_process_rss_bytes` label (`qualification.py:160`)**
`ru_maxrss` is bytes on darwin, KiB on Linux; the key name is platform-dependent truth.

## Scope limitations
- Not in packet: `.loop`, `.policy`, `.reference`, `pokemon_fixed.campaign`, `wire`/`digest`, `CompileLog`, test files. I cannot confirm `digest` is computed over `wire()` bytes (needed for `content_hash` round-trip, `seal:32`), that `dyadic_threshold` is genuinely independent of `threshold_words`, or that `exe.traces` counts retraces correctly.
- Float equality `decisions_per_second==decisions/elapsed` (`seal:58`) survives only if `wire` preserves repr; a mismatch would false-fail, not false-pass.

## Verdict
Findings. Scope claims are correctly bounded (`full_g2/g3 open`, `new_two_hour_soak False`, `strength_evaluation False`, `complete-for-declared-scope`); source/checkpoint/accounting linkage is sound. The exponent-126 probe gap (Finding 1) should be fixed before relying on the "exact threshold oracle" statement as covering the full legal mass range.
