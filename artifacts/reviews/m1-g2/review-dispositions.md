# M1/G2 independent review dispositions

Skill: `/Users/maxf/.codex/skills/claude-review/SKILL.md`.

## Bounded reducer implementation and opposite-model QA

Fable run `20261010-173129-67c1f57e`; Codex QA returned revise. See
`delegate-verification.md`, `delegate/qa.md` and `delegate/outcome.json`.
The original run is scored rejected because the coordinator's check command lacked
PYTHONPATH. The candidate's six tests passed when executed with the correct path.
The integrated reducer passed the fresh 1,000-game reference campaign; confirmed
request-validation gaps were fixed locally and covered by regression tests.
Only the owned reducer file was integrated. The candidate input hashes and original
patch remain preserved; later tests extend the original fixtures.

## Independent Fable snapshot review

Run `20261010-173700-1a7e20b0`, completed. Original verdict: low-severity findings;
no mechanics or leakage bug found in the reviewed scope. This is tool-free review,
not an executed test result. See `claude-review.md` and `claude-request.json` for
exact reviewed hashes. Subsequent local changes are accounted for here.

1. **Provenance omissions — fixed.** Added the sealed CPU slice, profile, reference
   and learner Python dependencies, Node reference modules, root lock, learner
   requirements and the sealer to the campaign source hashes. Added x64 runtime
   status and explicitly require x64 disabled at initialization. Existing upstream
   reference seals already bound some of these; explicit admission coverage is clearer.
2. **Imported turn histories — verified and fixed.** The review's proposed
   `[1,2,2,3]` example fails an existing last-turn check, but the related
   `[1,2,2,2]` sequence with consistent HP/PP was reproduced as accepted. Enforce
   history row k belongs to turn floor(k/2)+1; terminal may stop after either actor.
   A regression reproduces the malformed sequence and confirms rejection before step.
3. **Statistical interpretation — clarified.** Confirmed 105 empty Showdown cells,
   35 duplicated turn-one marginal cells and low expected rare-event counts. Added
   expected counts, zero-count p-values and number of cells unable to reject complete
   event absence at the frozen cutoff. The review's blanket 'cannot fail' statement
   is too strong: excessive occurrences can still reject; rare-event absence often
   cannot. Budgets, seeds, family and alpha are unchanged. No retrospective threshold
   tuning or rerun-until-pass procedure occurred.
4. **Information-test strength — strengthened.** Canceled-choice view and actor
   comparisons now cover both seats. Replaced a vacuous current-view RNG test with
   deterministic event-stream purity. Added reachable different-roll/same-public-
   damage pairs, canceled-chance pairs and mixed terminal/live batch isolation.
   View-to-actor identity is described as interface consistency, not a separate proof
   of policy/search information safety. Full G2 stays open.

All changes are local and checked. The final campaign and regression log bind the
corrected source. Earlier failed/pre-correction runs are retained, not overwritten.
No new training-strength, full-format or paid-compute claims are made.

## Finite-word sampler distinction

Inspected pinned `vendor/pokemon-showdown/sim/prng.ts` random/randomChance after
review: it uses floor(uint32 * range / 2^32). For the critical event at range 24,
178,956,971 words yield zero, versus the new rejection sampler's nominal exact
1/24. Absolute difference under uniform words is 1/(3*2^32), about 7.76e-11.
Documented this explicitly in the handoff. The frozen nominal-law tests are not
retrospectively changed and do not certify exact finite-word distribution parity.
Full G2 remains open. This is a reviewed scope distinction, not an unreported claim
of raw PRNG equivalence or a mechanism for resampling to obtain passing tests.
