# Proposed execution gates and experiment ladder

Contribution for the execution plan, 2026-10-10. All numbers below are proposed engineering acceptance criteria, not facts recovered from Jaxcalibur or guarantees of completeness/convergence. A gate combines an explicit supported scope, executable checks, and logged evidence. Passing a statistical screen licenses the next experiment; it does not prove global correctness or optimality.

## Dependency order and safe parallelism

The critical path is: pinned format/data contract → request/observation semantics → reference parity harness → restricted JAX engine → correct small PPO learner → measured learning → expanded mechanics → complete target-format evaluation → search. Keep action IDs, observation schemas, phase names, and checkpoint versions stable early.

Three independent workstreams are safe after the interfaces are agreed:

- **Simulator:** mechanics, reference scenarios, request timing, supported-feature manifest, JAX throughput. Owns `FullState` and transitions.
- **Policy/learner:** small transformer, per-player rollout buffer, PPO/GAE, auxiliary heads, checkpoint/resume. Initially develops against tiny scripted environments and frozen observation fixtures. Owns learner internals, not simulator truth extraction.
- **Data/evaluation:** pinned generator export, sampled-team validation, frozen team seeds, baseline policies, scoring and paired intervals. Owns fixtures and evaluation reports, not engine mechanics.

A separate observation/adapter owner approves interface changes spanning these streams. Belief-model design can run in parallel once observation fixtures exist; connecting beliefs to production action features waits for leakage tests. Search literature/toy-game experiments can run in parallel, but integration waits for a useful validated policy and simulator. Do not launch a large training run while known mechanics divergences remain in its sampled scope.

## Coverage stages: call each game what it is

**Stage A: synthetic canaries.** One-step bandits, deterministic delayed reward, asymmetric forced replacements, and toy hidden-information games. These test algorithms, not Pokémon strength.

**Stage B: restricted Pokémon.** Explicit fixed teams/sets and a small mechanics whitelist, first 1v1 then 3v3/6v6. Use reference battles with those exact teams and documented rules. Do not call a whitelist-filtered training distribution unmodified Gen 9 Random Battles. Excluded features must be impossible to encounter or must fail loudly before sample collection; never implement unsupported effects as silent no-ops.

**Stage C: expanding mechanics.** Add complete mechanic families in batches, with interactions: status/residual damage; stat changes; switching/pivots/trapping; item/ability interactions; field effects; Tera; copying/transformation and other edge cases. Re-run prior fixtures and a newly sampled interaction suite after each batch. Keep the reduced-format baseline checkpoint and compare transfer against training from scratch at the same budget.

**Stage D: intended complete format.** Pin the exact official generator, rules, simulator revision, and data export. Audit all generated sets *and the transitive mechanics they can invoke*; move copying or random move effects can extend beyond listed sets. A closed generator manifest plus a passing harness is evidence of supported scope, not proof that every interaction is correct. Record unknowns and observed unsupported effects as blockers to an unqualified full-format claim. Default train/eval sampling may only include fully supported matches; if exclusions remain, label the result restricted and report rejection rates/distribution shift.

## Quantitative gate ledger

### G0 — Contracts and testability (before learning)

Deliver a reproducible version manifest; phase diagram; observable action-mask contract; full-state/observation separation; deterministic seed protocol; supported-feature manifest; saved reference fixtures; checkpoint schema.

Acceptance: every phase has a fixture for normal entry, terminal exit, cancellation/re-request if applicable, and both player roles. Every action ID round-trips between network, simulator, and adapter. All displayed/consumed observations must be reconstructible from the same player's protocol history and private information. Unknown enum values and unsupported mechanics fail visibly. Zero tolerated silent fallbacks that alter battle rules.

### G1 — Learner canaries (before Pokémon scale)

Status 2026-10-10: passed for the documented whole-episode NumPy CPU canaries;
see [M1/G1 evidence and limitations](M1_G1.md). Cut arithmetic is hand-tested;
end-to-end persistent-game cuts and live-state checkpointing are not yet supported.

Run on CPU/FP32 where practical, using a transparent reference implementation.

- GAE/returns/log-ratios agree with hand-computed short trajectories within absolute 1e-5; terminal and buffer-cut cases are separate. No terminal bootstrap; rollout truncation bootstraps from the correct perspective.
- Same weights/data before an update give PPO ratios of 1 within 1e-5. Single-legal-action decisions have actor gradient/entropy zero within 1e-7, while a forced switch with multiple candidates remains trainable. The scalar surrogate need not be zero (it can equal minus the advantage); alternatively mask those rows out of the actor objective explicitly.
- One-step four-action canary, with exploration penalties disabled: all three learner seeds put at least 99% probability on the known rewarding action within 100,000 sampled decisions. This is a diagnostic target; failure triggers learner investigation rather than immediately expanding training.
- Deterministic delayed-reward canary learns both player perspectives without winner/loser sign inversion. Every example with a known action forbidden by the observable mask produces zero probability for that action. No NaN/Inf in sampled logits, probabilities, losses, gradients, or optimizer state.
- The boundary-checkpoint resume tests below pass before jobs longer than two hours. Mid-collection/mid-epoch tests become required only if those continuation modes are implemented.

### G2 — Restricted engine fidelity and information safety

Acceptance within the declared scope:

- Zero unexplained disagreements in legal request structure, observation/reveal timing, HP/status/PP changes, winner, and phase transitions across all hand-authored fixtures and at least 10,000 independently seeded reference battles. Adapt the battle count to CPU cost, but record the actual count and do not substitute correlated transitions for independent games in a failure-rate claim.
- Deterministic submechanics and forced random outcomes should compare exactly after normalization. For chance mechanics, either enumerate finite outcome distributions or run predeclared probability checks with a family-wise error policy. Same seed alone is not a parity contract if the engines consume randomness differently; inject/check semantically identified chance outcomes.
- At least 1,000 paired hidden-state mutations, stratified over unrevealed species/sets/items/abilities/Tera and request phases: normalized player observations must be identical whenever that player's information is identical. Deterministic inference logits must then match within 1e-6 FP32 (or exactly for byte-identical inputs and deterministic backend). This includes damage features, memory, masks, and cached values.
- For each covered mechanics family, include adversarial boundary cases: HP/KO thresholds, move PP exhaustion, simultaneous fainting, interrupted turns, forced switches, stale requests, and legal-information changes. Random volume cannot replace these targeted tests.

Interpretation: zero failures in 10,000 independent battles implies an approximate 95% upper bound of 0.03% for the tested distribution's per-battle failure probability, *not* for an untested distribution or all mechanics. A discovered divergence blocks training on affected states until resolved or explicitly excluded.

Current restricted progress: the [Showdown-chance campaign](M1_SHOWDOWN_CHANCE.md)
passes exact finite-word mapping checks and 10,000 fresh public-duel player-view
games with zero disagreements. This closes that slice’s fresh coverage target;
full G2 stays open for hidden-set and broader mechanics strata. It does not certify
Showdown seed/PRNG-stream equivalence. The binary-policy/new-chance combination
has only short integration checks and still needs its own two-hour qualification.

### G3 — Stable restricted-policy pilot

Engineering progress: the [fixed-shape loop and two-hour CPU soak](M1_FIXED_LOOP.md)
passed for the public Rattata duel, with zero new compilation events after warm-up
and exact boundary recovery. **Full G3 remains open:** there is no frozen-opponent
strength evaluation or certified device-memory headroom, G2 coverage is incomplete,
and the [new binary policy](M1_BINARY_POLICY.md) has only short qualification. Its
explicit floor and exact dyadic sampling resolve the old tail-law mismatch in this
two-action profile; the selected final policy/chance contract still needs its own
two-hour stability qualification.

Use the smallest practical feed-forward/history-token transformer first; defer recurrence and specialized token MLPs. Measure complete collect/update cycles, not isolated kernel throughput. Proposed pilot: three training seeds, a modest predeclared decision budget, and checkpoint evaluation every fixed budget increment.

Hard gates: zero unsupported effects or observation-mask bugs in training scope; zero NaN/Inf; no unexpected recompilation after warm-up at fixed shapes; a two-hour soak without OOM; peak allocated memory leaves roughly 15% headroom where the selected runtime permits it. Throughput gate is budget-derived: measured decisions/s must project the next experiment within its allocated GPU hours with at least 25% overhead allowance. Report decisions/game as well as games/s.

Learning gate: reliable improvement over initialization against a frozen baseline panel, plus a predeclared starter target such as point score ≥60% against random legal play and lower 95% confidence bound >55%. Include a damage-greedy baseline and old checkpoints as diagnostics. Beating random alone does not authorize a claim of strategic competence. Analyze all three training seeds; a single successful seed is insufficient to conclude the learner is stable. Record KL/clip fraction, entropy, legal-action probability floor, critic calibration, phase-specific losses, battle length, and timeouts.

### G4 — Full intended-format readiness

Expand G2's parity/hidden-info checks to all supported format families and the pinned official team distribution. Run at least 10,000 fresh independent reference battles after the last mechanics/schema change, and an adapter soak of at least 1,000 private/local evaluation games. Require zero unexplained divergences, zero known-forbidden actions, zero unsupported mechanics, and zero server-visible phase/request corruption. Distinguish legitimate hidden-legality rejections from client mistakes.

A production-length run is authorized technically only after a short full-format learner smoke test and resume test pass, and its measured complete-cycle throughput/memory predict the planned budget. Save learning curves against fixed opponents; current self-play score near 50% is not a learning metric.

### G5 — Search integration (after a useful policy)

First baseline: world-averaged one-step lookahead versus an observation-only opponent policy. Then simultaneous deeper search. Before Pokémon integration, test 2×2 games and rock-paper-scissors against exact solutions: at the declared rollout budget, mean-strategy exploitability ≤0.02 on normalized [-1,1] payoffs across a fixed test suite and at least 20 seeds is a proposed target. Report the whole distribution and worst case; do not elevate toy-game success into a Pokémon equilibrium claim.

Require exact pre-reveal hidden-information invariance for any opponent policy component explicitly intended to use only that opponent's information. Test returned action distributions, not only network inputs, because optimized search backups can leak secrets. Root mixed-strategy sampling is the default candidate; argmax is an ablation.

Promote a search configuration only after a predeclared meaningful gain (for example ≥2 percentage points in direct head-to-head score) with a paired lower confidence bound above 50%, no material regression against the fixed opponent panel, and p99 decision latency under the adapter's specified time budget. Use measured latency bounds chosen from the target deployment; do not invent a universal seconds threshold. Search timeouts must return a legal base-policy action and be counted.

## Phase-aware reject/re-request semantics

A policy's action mask represents what the player is allowed to infer from its request, not all legality knowable to the omniscient simulator. A choice may be admissible under that request but rejected when hidden mechanics are checked. Masking it using hidden truth can disclose information.

Implement and test three classes:

1. **Known-invalid choice:** forbidden by the observable mask, stale request ID, malformed mapping, or wrong phase. This is an implementation bug; quarantine/log the affected sample. Do not learn from accidental protocol failures.
2. **Rule-defined hidden-legality rejection:** reproduce the reference rejection message and knowledge update, issue the appropriate new request, retain commitments exactly as the protocol requires, and consume no turn/PP/chance/reward unless the reference does. The attempted choice and resulting information-bearing re-request form a zero-reward transition in that player's decision trajectory; do not silently erase useful feedback or reset GAE as if the game ended.
3. **Transport retry:** replay idempotently against the request ID; do not create a new learner decision or change the opponent's commitment merely because a message was resent.

Fixtures must cover repeated rejected choices, simultaneous submissions, one player awaiting re-request while the other has committed, terminal events before an anticipated request, and a hidden restriction becoming known. Test whether each feedback event is public or private. Keep the rejection/phase state in checkpoints. Retry-count safeguards should surface a bug or explicit failure, not secretly adjudicate a match as lost.

## Resume and on-policy state

Start with synchronous collection: a frozen actor snapshot collects a buffer; PPO updates that buffer; the next collection uses the new actor. Multiple GPUs, async actors, and stale-policy correction can wait.

A continuation checkpoint includes learner parameters, optimizer/scaler state, schedules and update counters; behavior/actor parameters and version; live simulator states; phase/request IDs and pending actions; observations/private memory; all PRNG streams; partially filled buffers with old log-probabilities/values/masks; collection cursor; minibatch permutation and epoch/minibatch cursor (plus accumulated gradients if mid-accumulation saving is supported); and data/schema/source versions. Save atomically with checksums. A manifest must make partial writes distinguishable from valid checkpoints.

Initial resume gate: support completed-update checkpoints at the documented collection barrier, and on a deterministic CPU canary reproduce the next samples and updates from a restore. Live environment/history/request state is required even at that boundary when games persist. Also simulate a crash during collection and during an update: roll back atomically to the last complete checkpoint, discard intervening work and verify loss/re-execution accounting. This is rollback recovery, not exact continuation at the crash point. GPU checks compare counters, actor versions, masks, next PRNG keys and trajectory identities; declare any numerical tolerance without allowing missing or repeated committed data.

Optional extended gate: only after adding exact mid-collection, pre-update-buffer or mid-epoch checkpoint modes, interrupt at each supported point and reproduce the uninterrupted canary's continuation with its complete state above. Prefer bit identity for deterministic operations. A mode that resets unfinished games must be labeled non-exact and report lost samples; it does not satisfy exact continuation. Never stitch a partial buffer to a different behavior-policy version. Raw histories are preferable to unversioned learned recurrent state initially. If recurrence is later added, specify hidden-state recomputation/burn-in across policy updates and restores.

## Lean experiment ladder and statistical promotion

Run one decision-relevant question at a time:

1. Tiny canaries and PPO correctness, no architecture search.
2. Restricted policy baseline, then one exploration/GAE screen (lambda and zero-avoidance settings); keep the best stable cheap configuration.
3. Set-prediction auxiliary head, opponent-action head, then both. Require calibration/hidden-state safety in addition to win-rate gain.
4. Snapshot opponent pool versus current-self-play only; measure cyclic regressions against frozen checkpoints.
5. Expand mechanics and distribution. Repeat the useful baseline before adding model capacity.
6. Correlated generator-based belief candidates versus no belief input; only then learned conditional proposals.
7. One-step search, deeper simultaneous search, and world-count/rollouts/depth tradeoffs at equal inference time. Then test sampled mixture versus argmax and exact HP versus bins.
8. Only after gains survive: specialist MLPs, delayed matchup features, recurrence, distillation, or larger networks. Each competes at both matched sampled decisions and matched GPU-hours; these answer different questions.

Use three independent training seeds for pilot selection and preferably five for consequential comparisons. Match team seeds and swap seats. A team-seed pair is one statistical cluster; do not count both games as independent. Direct head-to-head pilot: 1,000 pairs (2,000 games); promotion: 5,000 pairs (10,000 games), extending by a predeclared plan if precision is inadequate. Worst-case normal-approximation 95% half-widths for bounded pair scores are about 3.1 and 1.4 percentage points, respectively. Actual precision depends on within-pair covariance; bootstrap complete pairs, and account for training seeds separately. These counts describe game-level comparisons: balance evaluation across training seeds and report per-seed results; no quantity of games from one checkpoint replaces independent training runs. If a claim depends on run-to-run reliability, budget evaluation for each seed and estimate the training-seed variation explicitly.

For comparisons against a fixed opponent panel, compute per-pair score differences between candidate and baseline and bootstrap those paired differences, stratified by opponent. Their variance differs from direct head-to-head score variance; do not reuse the same precision claim. Estimate sample size from a pilot using n ≈ (1.96+0.84)^2 s_delta^2 / desired_gain^2 for a rough two-sided 5%, 80%-power plan, then freeze that plan. Correct for multiple comparisons or reserve untouched confirmation seeds. Avoid repeatedly checking ordinary intervals and stopping when favorable.

Common RNG seeds are not automatically common chance outcomes after policies take different actions. If event-indexed random coupling is used, validate it; otherwise pair the teams and seats while allowing independent chance, and state that choice. Preserve full score tables, confidence intervals, generator/simulator/model versions, illegal-action/rejection categories, turn counts, and latency distributions in each promotion report.
