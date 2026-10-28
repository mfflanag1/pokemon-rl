# F7: supervised fit of the sealed F6 decision states

Completed and exhaustively replayed on 2026-10-24. The preregistered probability gates failed for all three seeds, while final argmax accuracy reached 100% on every split. This is a finite-task representability diagnostic after F6's conditional-choice gate failed in all three seeds. The existing F1 encoder, 928→64 tanh→14 policy network and masked dyadic action distribution stay unchanged. Fresh parameters use seeds 7103, 8209 and 9311; no F6 policy is selected or reused.

## Fixed design

Use all 512 player observations in F6's sealed mechanics probe. The label is the unique action maximizing the already recorded one-turn HP-retention score across every legal action. Labels, scores and context identifiers are external to the actor input; the model receives only the existing encoded player view and legal-action mask. There are no new battles. The objective is supervised negative log likelihood with the existing Adam kernel, learning rate 0.0003, gradient norm cap 1.0, full batches of 128, and exactly 1,024 updates per seed. No entropy or value loss is used. The unchanged zero-initialized value head stays at 0.5 and is not a trained critic.

The fixed training set is the 128 train-panel states at episode coordinates 0–7. The same-family probe uses the 128 train-panel states at coordinates 8–15. The variant probe uses all 256 existing variant-panel states. Coordinates are disjoint, but encoded observations can recur; the report inventories unique inputs and every cross-split overlap. The variant species/level combinations have already appeared in F6 development and are not untouched held-out or final evaluation data.

Evaluate all states at updates 0, 128 and 1,024 using the same 128-row inference shape. Keep every checkpoint and seed. The primary gate requires 100% argmax accuracy and at least 95% mean target probability **in every one of the 16 training contexts**, in every seed. The separate same-family gate requires 100% accuracy and 90% target probability per context; the variant gate requires 90% accuracy and 80% probability per context. Accuracy uses NumPy argmax with the smallest action index winning exact ties; gates also require the specified per-context target-probability mass. The report and replay audit record the explicit conjunction of all three training gates as the primary result. No best-checkpoint selection or budget extension is permitted. See [frozen plan](../contracts/decision-fit-v1/plan.json).

The final budget is 3,072 Adam steps, 393,216 training row presentations and 4,608 evaluation predictions. Per seed, the 128-update snapshot has the same optimizer-step count as F6; this is descriptive, not a controlled comparison. The final budget has eight times as many optimizer steps/row presentations as F6, dense oracle labels, repeated static inputs and a different objective. A successful fit shows that this architecture can express the required decisions on these finite states; it does not isolate which RL mechanism failed. An unsuccessful fit under this one fixed optimizer budget does not prove architectural impossibility.

## Verification and boundaries

Before execution: focused data admission, label derivation, context metrics, gradients, masking and original-next-update tests; opposite-model QA of the delegated metrics implementation; independent review of the supervised experiment. After execution: replay every update and metric, all prediction fingerprints and summaries, exact initial/intermediate/final parameters and Adam states, and the original next update loaded from its saved checkpoint. Bind the tested and executed source trees, runtime, plan, F6 receipt and probe digests, measured results and documentation in a repeatable completion receipt. Preserve all F6 source/evidence hashes.

This does not train full battles, measure battle wins or establish transfer, ladder performance or #1 readiness. Full G2/G3/G4 remain open. Final reserved data, public servers and Prime are unused. The GitHub visibility hold remains in force.

## Measured result

The single fixed run completed in **3.378 measured seconds** on local CPU (3.845 seconds for the production process, including startup). It performed **3,072 Adam steps**, **393,216 training row presentations** and **4,608 evaluation predictions**, with **zero new games**. Peak process RSS was **375.5 MiB**. All updates and states were finite; exact initial/intermediate/final checkpoints and all three original-next-update recoveries passed. The fresh replay process exited 0 in 3.670 seconds. Production and audit exit records are preserved.

**Primary result: failed.** Every seed failed the final per-context probability threshold on all three splits. Every seed nevertheless achieved 100% final argmax accuracy on all 512 states: 1,536 correct final predictions across seeds. Accuracy alone does not pass the preregistered gates and does not make stochastic action selection perfect.

| Seed | Split | Initial correct | Update 128 correct | Update 1,024 correct | Final mean target probability | Minimum context mean | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| 7103 | fit_train | 32/128 | 96/128 | 128/128 | 93.68% | 86.31% | Failed probability |
| 7103 | probe_same | 32/128 | 96/128 | 128/128 | 93.58% | 86.31% | Failed probability |
| 7103 | probe_variant | 64/256 | 160/256 | 256/256 | 84.18% | 69.52% | Failed probability |
| 8209 | fit_train | 16/128 | 99/128 | 128/128 | 94.46% | 89.77% | Failed probability |
| 8209 | probe_same | 16/128 | 98/128 | 128/128 | 94.37% | 89.01% | Failed probability |
| 8209 | probe_variant | 64/256 | 173/256 | 256/256 | 85.64% | 73.06% | Failed probability |
| 9311 | fit_train | 16/128 | 96/128 | 128/128 | 94.88% | 89.81% | Failed probability |
| 9311 | probe_same | 16/128 | 95/128 | 128/128 | 94.76% | 89.07% | Failed probability |
| 9311 | probe_variant | 32/256 | 192/256 | 256/256 | 85.35% | 68.48% | Failed probability |

The 128 training rows contain **82 distinct encoded inputs**, the 128 same-family probe rows contain **80**, and the 256 variant rows contain **127**. Training and same-family probe share **44 distinct inputs**; the variant shares none with either. There are **245 distinct inputs** overall. This overlap limits the interpretation of the same-family probe. The variant result concerns these specific, previously developed species/level scenarios, not unseen-team generalization.

The final training negative log likelihoods were 0.066759, 0.057867 and 0.053355 for seeds 7103, 8209 and 9311, respectively, versus 1.927622, 1.960017 and 1.954767 at initialization. The value head remained unchanged at a prediction of 0.5 throughout; value learning, battle-terminal outcomes and win calibration are not measured by this experiment.

All **15 focused tests** pass, including four existing F1 feature/policy checks. The pinned Showdown build and source admission pass. The independent Fable review found no material defect; its two low-severity comments were addressed by an explicit all-seed primary field and documented argmax tie semantics. The bounded metrics implementation was separately checked and reviewed by the opposite model. An initial test-invocation error is retained in `tests-first.log` and corrected before preregistration; the final test/source binding covers 425 files.

The fresh-process audit reproduced **every update and metric**, all 4,608 evaluation prediction fingerprints, every gate and the initial, intermediate, final and original-next-update model/Adam states exactly. A separate NumPy forward/softmax calculation confirmed all 1,536 final argmax labels and the probability-threshold failures, within 1e-5 of the recorded dyadic probabilities. No source, training budget, seed or threshold changed after preregistration. All sealed F6 source/evidence hashes remain unchanged.

This provides a finite greedy-policy representability witness for the unchanged architecture on these observations. It does **not** establish reliable stochastic mastery under the declared thresholds, identify the cause of PPO's failure, or demonstrate full-battle improvement. The higher optimization budget, oracle supervision and static data differ from F6. The failed primary gate is retained without extending training or selecting a different checkpoint.

**Next proposed step, not executed:** preregister a matched-budget comparison of fresh initialization versus each seed's supervised checkpoint under the same PPO loop. Reset optimizer state consistently, retain all seeds, use fresh development mechanics realizations, and measure retention of conditional decisions, sampled target probability and any regression. This can test whether RL preserves the acquired policy before adding more curriculum or full-battle training. It does not require Prime.

Completion receipt: [F7 receipt](../artifacts/reports/f7-decision-fit.json). Evidence: [fixed production report](../artifacts/runs/f7-decision-fit-v1/report.json), [full replay](../artifacts/reviews/f7-decision-fit/production-audit/audit.json), [independent result check](../artifacts/reviews/f7-decision-fit/independent-result-check.json), [review dispositions](../artifacts/reviews/f7-decision-fit/review-dispositions.json). GitHub publication remains on the unresolved visibility hold. Full G2/G3/G4, final reserved evaluation and public ladder strength remain open.

