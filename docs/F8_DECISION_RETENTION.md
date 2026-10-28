# F8: matched-budget PPO after supervised initialization

Completed and exhaustively replayed on 2026-10-24; the primary conditional-choice retention gate failed. F7 showed that the unchanged network can rank the correct action first on its finite diagnostic observations, while all probability gates failed. F8 compares fresh initialization against each seed's sealed F7 step-1,024 parameters under the unchanged F2 PPO loop. This tests local conditional-choice retention and improvement, not full-battle RL or public strength.

## Fixed design

Retain seeds 7103, 8209 and 9311 in both arms. Each arm receives exactly 64 cohorts of 128 fresh one-decision training cases, with two valid-padded Adam updates per cohort: 8,192 cases and 128 Adam updates per arm/seed. Both arms reset Adam moments and the optimizer counter to zero. Both begin with the unchanged zero value head, predicting 0.5; the shared policy trunk differs by initialization. No auxiliary supervised loss, KL anchor, learning-rate change or other PPO change is introduced.

Within each seed, use identical simulator coordinates, actor random roots and optimizer-shuffle roots for both arms. The actions and rewards are collected separately from each arm's current policy. The serialized pre-action player views and provenance must match across arms. Training uses fresh F8 mechanics seeds on the existing train-panel templates. This matches **new RL cost**, not total compute or supervision: the supervised arm already received F7's 1,024 supervised Adam steps and 128 labeled examples.

Evaluate both endpoints on the same 512 fixed fresh coordinates: 16 contexts × 16 mechanics realizations × two panels. Before collecting the policies' evaluation decisions, branch every legal action in the actual pinned simulator to derive and verify each unique optimal HP-retention label. The actor receives only its existing player observation and mask; PPO receives sampled action/reward trajectories, never these labels. Both panel species/level templates are known development cases. Fresh mechanics coordinates can produce repeated encoded observations; measure overlap with F7's supervised training inputs. Final reserved data remain untouched.

Total fixed budget: 49,152 training decisions, 6,144 evaluation decisions, 768 Adam updates, 98,304 valid training-row presentations and 3,584 oracle evaluation branches. There are no complete-battle continuations. A diagnostic terminal flag ends the local one-decision objective; it is not a natural battle terminal. The reward remains exposed-member one-turn HP retention. Value error concerns this local reward, not a battle win probability.

## Preregistered gates

The primary retention gate requires the supervised arm to satisfy all three conditions in **every train/variant context, in every seed**: final greedy accuracy at least 95%; target-action probability at least 60%; target probability no more than five percentage points below its own initialization. With 16 examples per context, the accuracy threshold requires all 16 to be correct. Exact argmax ties use the smallest action index. These are diagnostic thresholds, not confidence bounds or a significance test.

Two separate secondary gates assess the mean target probability in each of the four panel/family groups (train/switch, train/Tera, variant/switch, variant/Tera). Improvement requires at least two percentage points over the supervised arm's own initialization in every group. Final advantage requires at least 15 percentage points over fresh initialization's final policy in every group. Retention, improvement and advantage are reported separately for every seed. Preserve every context and any regression; neither the fresh arm nor a favorable aggregate can rescue a failed primary gate. See [plan](../contracts/decision-retention-v1/plan.json).

The budget, seeds, scenarios and thresholds are fixed before production. No early stop, seed selection, best-checkpoint selection or budget extension. The final run remains a valid completed experiment if any or all learning gates fail.

## Verification and scope

Use focused tests of optimizer reset, exact checkpoint transfer, paired actor-only collection, local reward targets, metadata/counters, metric boundaries and oracle outcome matching. Independently review the experiment and the delegated metrics module. Build and verify the pinned simulator. Freeze the tested source/configuration inventory and runtime before starting the run. Replay every legal evaluation branch, training/evaluation decision, PPO/Adam update, initial/final checkpoint and original next update. Preserve all F7 and F6 source/evidence bytes. Seal measured documentation and results, then independently check every live/archive hash and reseal to identical receipt bytes.

Full G2/G3/G4 and public ladder strength remain open. No public server, final reserved evaluation or Prime use. The existing GitHub visibility hold remains in effect. No automatic follow-on training is authorized by the outcome of this fixed comparison.

## Measured result

The single production run completed in **320.98 measured seconds** (5.35 minutes) on local CPU. It collected **49,152 training decisions** and **6,144 evaluation decisions**, performed **768 Adam updates** with **98,304 valid row presentations**, and verified **3,584 legal evaluation branches**. All three seeds and both arms are retained. Parent-process peak RSS was **512.1 MiB**, excluding the eight Node workers. The fresh exhaustive audit exited 0 in 386.94 seconds. Both initial weights, zeroed optimizer states, all six original-next-update recoveries and final checkpoints verified exactly.

**Primary conditional-choice retention gate: failed.** All three supervised arms preserved **100% final greedy accuracy** on the 512 fresh evaluation coordinates, compared with **25%** for each fresh arm. All three passed the declared final-advantage gate. The retention failures came from action-probability drops in specific variant contexts, not incorrect greedy choices. Overall mean target-action probability increased from 88.97% to 90.09%, 90.09% to 91.47%, and 90.17% to 92.50% for seeds 7103, 8209 and 9311. Each seed failed the requirement for at least two percentage points of improvement in every family; average gains do not replace that gate. The separate improvement and fresh-arm advantage gates are reported below; neither substitutes for retention.

| Seed | Retention | Improvement over supervised start | Final advantage over fresh | Failed retention contexts |
|---|---|---|---|---:|
| 7103 | Failed | Failed | Passed | 2/32 |
| 8209 | Failed | Failed | Passed | 4/32 |
| 9311 | Failed | Failed | Passed | 2/32 |

| Seed | Panel/family | Fresh target mass: initial → final | Supervised target mass: initial → final | Supervised minimum final context mass | Supervised final greedy correct | Supervised final sampled correct |
|---|---|---:|---:|---:|---:|---:|
| 7103 | train/switch | 14.9% → 0.8% | 88.6% → 90.4% | 87.6% | 128/128 | 114/128 |
| 7103 | train/tera | 14.3% → 47.5% | 98.5% → 99.0% | 98.1% | 128/128 | 126/128 |
| 7103 | variant/switch | 14.8% → 0.8% | 76.6% → 78.6% | 66.5% | 128/128 | 102/128 |
| 7103 | variant/tera | 14.3% → 47.5% | 92.2% → 92.3% | 84.1% | 128/128 | 119/128 |
| 8209 | train/switch | 14.4% → 0.9% | 90.4% → 92.8% | 90.6% | 128/128 | 125/128 |
| 8209 | train/tera | 13.8% → 46.9% | 98.3% → 98.7% | 97.6% | 128/128 | 126/128 |
| 8209 | variant/switch | 14.4% → 0.9% | 79.5% → 82.9% | 74.8% | 128/128 | 109/128 |
| 8209 | variant/tera | 13.9% → 47.1% | 92.2% → 91.5% | 83.0% | 128/128 | 122/128 |
| 9311 | train/switch | 14.9% → 0.8% | 90.9% → 93.7% | 88.1% | 128/128 | 116/128 |
| 9311 | train/tera | 13.5% → 47.1% | 98.5% → 99.1% | 98.9% | 128/128 | 127/128 |
| 9311 | variant/switch | 14.9% → 0.8% | 79.0% → 82.6% | 72.2% | 128/128 | 109/128 |
| 9311 | variant/tera | 13.5% → 47.1% | 92.3% → 94.6% | 92.8% | 128/128 | 120/128 |

| Seed | Arm | Mean local reward: initial → final | Local value MSE: initial → final |
|---|---|---:|---:|
| 7103 | fresh | 0.6004 → 0.7514 | 0.097173 → 0.064939 |
| 7103 | supervised | 0.8681 → 0.8952 | 0.207861 → 0.064576 |
| 8209 | fresh | 0.5936 → 0.7682 | 0.104582 → 0.053058 |
| 8209 | supervised | 0.9067 → 0.9298 | 0.209817 → 0.045941 |
| 9311 | fresh | 0.5984 → 0.7645 | 0.097895 → 0.057193 |
| 9311 | supervised | 0.8869 → 0.9212 | 0.206458 → 0.052017 |

Failed retention contexts, without omission: Seed 7103: variant/8, variant/9. Seed 8209: variant/12, variant/13, variant/14, variant/15. Seed 9311: variant/0, variant/1.

All evaluation cases ended the diagnostic while the underlying battle was still ongoing: **6,144 ongoing, zero natural terminals and zero administrative cutoffs**. No complete battles or continuation games were collected. These are not win/loss observations. The critic is fitted to one-turn HP retention; its local MSE is not terminal-outcome calibration.

The fixed evaluation probe's unique optimal targets passed in all 512 contexts/realizations. All training and evaluation records reproduced in the pinned simulator, every PPO update and metric matched exactly, and the saved model/Adam state reproduced the original next update for all six seed/arm combinations. An independent NumPy calculation confirmed all 6,144 endpoint predictions/metrics within 1e-5 of the dyadic law; gate arithmetic was recomputed separately. All **25 focused tests** and the pinned simulator build/admission pass. The reviewed and tested source inventory has 433 files. Cross-model review findings about checkpoint coverage and training-input identity were resolved with byte/fingerprint checks before production; all-branch reward reconstruction was added before final testing. No training source, seed, budget or threshold changed after preregistration.

## Input overlap and limits

Fresh mechanics seeds do not imply new semantic situations or unseen encoded inputs. The scenarios and variant species/levels remain the F6/F7 development templates. F8 evaluation overlap with the F7 supervised inputs:

```json
{
  "fresh_coordinates_not_guaranteed_unseen_inputs": true,
  "fresh_evaluation_unique_inputs": {
    "train": 123,
    "variant": 128
  },
  "overlap_with_supervised_training": {
    "train": 57,
    "variant": 0
  },
  "same_templates_and_known_variant_species": true,
  "supervised_training_unique_inputs": 82
}
```

Overlap between F8's actual on-policy training and its evaluation inputs, inventoried after execution without altering the gates:

| Seed/arm | Training rows | Distinct training inputs | Distinct evaluation inputs | Shared distinct inputs |
|---|---:|---:|---:|---:|
| 7103/fresh | 8192 | 436 | 251 | 120 |
| 7103/supervised | 8192 | 436 | 251 | 120 |
| 8209/fresh | 8192 | 439 | 251 | 118 |
| 8209/supervised | 8192 | 439 | 251 | 118 |
| 9311/fresh | 8192 | 440 | 251 | 117 |
| 9311/supervised | 8192 | 440 | 251 | 117 |

This experiment compares new PPO training under two different initializations, including the shared trunk. It does not identify whether actor loss, value loss, entropy, optimizer conditioning or exploration causes any observed change. It is not a matched-total-compute test: the supervised arm retains F7's additional optimization and oracle supervision. No inference about public strength, unseen teams or full-battle learning follows from these local diagnostic gates.

The practical result is that supervised initialization supplies conditional choices that this short fresh PPO run does not discover, and their greedy ranking survives subsequent PPO. This supports an exploratory transfer test, not a broad promotion or a conclusion about the source of the variant regressions. A proposed next step is a small full-battle transfer pilot that retains the tactical probe, compares against the frozen supervised policy, and tests a declared supervised rehearsal term to limit drift. No follow-on experiment has started. Local CPU remains sufficient for that pilot.

Completion receipt: [F8 receipt](../artifacts/reports/f8-decision-retention.json). Evidence: [production report](../artifacts/runs/f8-decision-retention-v1/report.json), [full replay](../artifacts/reviews/f8-decision-retention/production-audit/audit.json), [independent endpoint and overlap check](../artifacts/reviews/f8-decision-retention/independent-check.json), [review dispositions](../artifacts/reviews/f8-decision-retention/review-dispositions.json).

