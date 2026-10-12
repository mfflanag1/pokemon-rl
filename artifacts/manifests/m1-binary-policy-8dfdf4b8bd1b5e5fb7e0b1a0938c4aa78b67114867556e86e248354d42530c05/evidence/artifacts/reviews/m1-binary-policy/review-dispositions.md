# Verified dispositions: binary-policy-1

## Executed checks

- The 11 initial unittest cases passed in 16.305 seconds. They cover an independent
  integer-ratio oracle and analytic derivatives, all admitted exponent strata and
  edge significands, mirrored historical tails, floor/tie/forced gradients, PPO
  finite differences, same-policy ratios, key reordering and partitioning, nine
  compact-collector cases, fused/split updates, poisoned padding, empty minibatch
  skips, exact disk replay and collection/update/publication failure injection.
- The full initial source-bound regression passed 217 tests against 181 source
  files. The later sealer diagnostic test increases this to 218; the final
  verification.json/npm-test.log is authoritative and preserves the earlier log.
- Both campaigns m1-binary-policy-v1 and v2 passed: 100,002 exact scalar-ratio thresholds,
  600,012 injected selections, 24 parity batches, three 100-update seed runs,
  five exact next-cycle disk replays and a 60-second CPU run with real positive
  compilation-log control and zero new compilation events after warm-up.
- Source/runtime/predecessor checks and the final reservation gate passed at both
  campaign endpoints. Previous immutable packages/contracts are unchanged.
- Independent review was tool-free. None of its statements are represented as an
  executed check. `reviewed-source-check.json` records unchanged reviewed files;
  the completion sealer and plan schedule metadata changed after audit, as described below.

## Policy review: 20261010-215520-ac8ba4f1 (Fable)

No blocking finding. The threshold proof, strict integer comparison, log/log1p
law, analytic oracle derivative, masking and checkpoint isolation were checked
by source reading. The coordinator also executed the tests above.

1. Forced/terminal `rare` helper fields differ between oracle and kernel. Confirmed,
   but these are internal inactive transport values, not probabilities of a live
   binary choice. Forced/padded probability/log/action values are equal and checked.
   The handoff documents that helper-field comparison is not supported there.
2. The equal-logit derivative depends on the admitted JAX abs JVP convention.
   Verified indirectly and more usefully by actual selected-log derivative tests
   for both actions at zero difference, matching [+0.5,-0.5] or its negative.
   The environment is pinned and provenance-checked. No source change was needed;
   future runtime migration must rerun these tests.
3. Oracle independence is partial. Confirmed: ratios/Fraction and analytic
   differentiation are independent, whereas the NumPy and JAX mass formulas are
   intentionally the same. The handoff limits the independence claim accordingly.

## Qualification/sealer review: 20261010-215832-73dd8530 (Fable)

No blocking issue with the bounded scope; findings independently adjudicated:

1. Claimed missing exponent-126 interval is **incorrect**. IEEE FP32 .25 has biased
   exponent125; every representable value in [.25,.5) does too. The only admitted
   value with exponent126 is exactly .5, explicitly added. Its next representable
   neighbor is .5000000596046448 and is outside the contract. The executed
   `exponent-audit.json` records these bit patterns. Extending random mantissas to
   exponent126 would introduce invalid probabilities. Kept the correct sampling
   range and proof; the unit test also spans every legal exponent/significand edge.
2. Proof-string conditionality: no mathematical error. The sentence states an
   integer mapping, with uniform_word_assumption=true beside it. Contract and
   handoff explicitly condition mass claims on uniform word pairs. No unsupported
   PRNG independence or empirical rare-frequency claim is made.
3. Collector parity shares the new actor implementation. Confirmed and explicitly
   stated in both report reference string and handoff. It tests control flow,
   observations, trajectories, GAE and fusion; sampler math has separate integer
   and analytic tests. It is not independent whole-policy derivation.
4. Sealer did not bound stored diagnostic values. **Fixed.** It now rejects
   nonfinite/out-of-range minimum behavior probabilities and update ratio errors.
   The floor allows a small exp(log) FP32 rounding tolerance. Observed PPO ratio
   error may exceed the clipping parameter; clipping is not a bound on raw ratios.
   The meaningful absolute ratio bound derives from the policy's probability floor.
   A new test rejects zero/below-floor/>one/NaN/Inf probabilities and negative,
   impossible or nonfinite ratio errors while accepting a valid .21 ratio error.
5. Test roster attribution: the source-bound runner and test files already identify
   all suites. Added explicit extraction and count checking of unittest suite paths,
   including tests/binary-policy, and store the path-to-count mapping in the receipt.
6. ru_maxrss unit: actual admitted runtime is Darwin, where the report's byte label
   is correct. This is a pinned-runtime CPU qualification, not a portable Linux
   measurement. A future OS/runtime migration must adjust the unit conversion and
   rerun admission and performance checks; no Linux result is claimed.

## Independent delegate delivery

Auto routing chose Codex on its fifth comparable exploratory attempt. The candidate
failed ownership enforcement (generated bytecode) and used pytest instead of the
requested unittest/no-new-dependency setup. Controller checks never ran; QA and
adjudication commands both refused a failed run. This remains failed delivery,
not an accepted implementation or routing outcome. See delegate-verification.md.

The coordinator inspected original input hashes, isolated and `git apply --check`ed
only reference.py, and integrated that source under coordinator ownership. No
candidate test or bytecode was integrated. The locally written unittest checks and
Fable policy review above verify the salvaged source. The original candidate and
failed-run artifacts remain available; no model opinion alone counts as success.

## Promotion limits

Only binary-policy numerical qualification is complete. The old two-hour soak
belongs to the old sampling law. The new policy needs its own two-hour stability
qualification once the final chance contract is selected. Full G2/G3, exact
Showdown finite-word chance mapping, 10,000 fresh player-view games, hidden-set
mutation strata, richer mechanics, critic calibration and frozen-opponent strength
remain open. No Prime resources or final reserved evaluation games were used.

## Coordinator plan cleanup

After the first successful 218-test regression, inspection found two inherited
fixed-loop scheduling fields that the new short qualification did not consume.
Removed those unused interval fields and recorded the actual five-boundary save
schedule explicitly. The algorithm and checkpoint mode did not change. Because
the plan is provenance-bound, the qualification was rerun in fresh directory
m1-binary-policy-v2; v1 and its earlier plan remain preserved. The final sealer
archives both campaign directories and records their content hashes. The final
full regression must bind the cleaned plan and updated evidence sealer.

## Completion reporter parser repair

The first seal attempt correctly stopped without publishing a receipt: its Node
summary parser expected TAP `# pass`, while the successful regression log used
the default spec reporter `ℹ pass`. The Python counts were already exact. The
sealer now accepts either explicit summary format and requires zero Node failures.
The existing receipt test covers both, a failing summary and a missing summary;
that targeted test passed. A fresh full source-bound regression binds this parser
and test repair before final sealing. The campaign's numerical sources and plan
did not change and its result does not need to be rerun for this reporting fix.
