# F32 — observed replacement of the incomplete F30 continuation

Status: fixed production and exhaustive replay passed; sealing verification is recorded in the completion receipt.

The user authorized this fresh comparison after F31 qualified the observed worker transport. F30 remains a failed, incomplete execution; its all-worker timeout was not reproduced and its cause is unresolved. F32 starts all six original F29 final states again in a new run directory. It does not resume F30, use its partially trained checkpoints, stitch its completed cells, increase the deadline or retry worker requests.

The scientific question remains whether the nonlinear full-battle actor keeps learning with a larger fixed training budget, before adding search. F29 showed tactical improvement but its final advantage over the trained control passed in only two of three seeds. The replacement preserves all seeds and both arms without selecting a favorable result.

## Preregistered scope

The immutable contract is `contracts/observed-continuation-v1/plan.json`.

- Seeds 7103, 8209 and 9311; control and nonlinear arms. Load each same-arm F29 cohort-256 checkpoint with parameters, Adam moments, step 2,048 and next-episode cursor intact.
- Continue absolute cohorts 256–1,023: 768 new cohorts per cell. Each cohort selects 512 self-play and 512 tactical rows, uses two epochs and eight optimizer steps. Model, public inputs, reward, PPO/GAE, learning rate, selection, training and optimizer namespaces are unchanged.
- Across all six cells: 4,608 new cohorts, 36,864 optimizer steps, 4,718,592 selected rows and 9,437,184 presentations. Each final cell reaches cohort 1,024, cumulative step 8,192 and 1,048,576 selected rows.
- New fixed development seeds: main 1027263201, diagnostic 1027263202, bootstrap 1027263203. Evaluate the continuation starts and endpoints 512 and 1,024 on the same F32 panels. There are 128 paired schedules in both seats per random/base-power/tactical opponent: 13,824 main games plus 576 separate self-play diagnostic games.
- Verify 30 checkpoints and 12 original-next-update recoveries. Exhaustive fresh replay must reproduce every training/evaluation game, action/value prediction, selection, optimizer update and checkpoint.
- Admit all prior histories, including every persisted training and evaluation game from partial F30, the six F31 engineering cells, and F32 preflight. Retain the existing exact-team role gates and final reserved evaluation boundary.

The training schedules intentionally continue the same F29 trajectories as F30. This is a replacement execution, not an independent training replication. Fresh F32 evaluation panels avoid recycling inspected F30 outcomes. Starting checkpoints differ between arms; final nonlinear-minus-control advantage includes inherited differences and is not a difference-in-differences estimate of additional training's effect.

## Gates and interpretation

Retain all 27 dependent paired-schedule contrasts and cutoff-aware bounds. The primary trained-on tactical gates require at least five percentage points of conservative point improvement, a positive registered familywise bootstrap lower endpoint, no detected random/base-power regression and final cutoff rates at most 5%. Report within-arm continuation learning and final nonlinear-minus-control benefit for every seed. All-seed claims require all three seeds. Base-power results are secondary; midpoint, mastery, terminal-only value behavior and pre-update minibatch KL/clip statistics are descriptive.

A completed comparison may fail learning gates. Tactical is trained on, and the development opponents and official generator are familiar. The result cannot establish unseen-opponent, held-out-team or public ladder strength. Full G2/G3/G4 remain open. No policy promotion, public server, final reserved data or Prime is authorized by this workflow.

## Engineering and preservation

The only campaign-body change from F30 is the Pool import selecting F31's observed transport. It retains the 60-second combined write/read deadline and zero retries, records command/phase/byte/resource evidence, and preserves primary errors through cleanup. F31's 37 focused tests and six-cell exact parity are admitted through its sealed receipt. They establish short-run engineering correctness, not the cause of F30's failure or long-duration reliability.

New focused tests cover the replacement boundaries, complete historical admission and immutable receipts. A six-cell engineering preflight compares collection/update results with the sealed F29 function and verifies active watched diagnostics and checkpoint recovery. The pinned Showdown build, independent review and verified dispositions are required before launch. Source/runtime/helper admission binds the qualified tree.

Production: `artifacts/runs/f32-observed-continuation-v1`. Review/replay: `artifacts/reviews/f32-observed-continuation`. One persistent driver runs production, exhaustive replay, measured documentation, copied-evidence rejection checks, first sealing, full integrity checks and identical repeat sealing. On any failure it preserves evidence and stops; no automatic repair, restart, seed changes, cell dropping or budget extension.

Only this new handoff is updated at completion. Historical handoffs, current documents already bound by predecessor receipts, and all sealed sources/evidence stay unchanged. Post-seal logs and integrity records remain outside the sealed review directory. GitHub stays under its existing unresolved visibility hold; no push, upload or visibility change.

Completion receipt: `artifacts/reports/f32-observed-continuation.json`. A valid result requires passed production/replay reports, clean launcher/driver exits and independently verified receipt/source/evidence hashes.

## Launch qualification

All 21 focused tests passed against 741 executable/configuration files. The pinned simulator build passed. All six observed collect/update preflights passed in 125.68 seconds: 232 scheduled engineering games, repeated twice through the observed transport (464 physical executions), 6,144 selected rows and 48 optimizer steps per comparison side. Every Adam counter advanced 2048→2056; selected batches, full ledgers, update metrics, parameters, optimizer state and checkpoint round trips matched. Every worker emitted the required diagnostic stages without drops. These counts exclude production and are not learning evidence.

Independent static review and verified dispositions are preserved in the review directory. F31 remains the separately sealed old-versus-watched transport comparison; this new preflight compares the continuation function to the sealed F29 function through the observed Pool. The launch binds the unchanged source/runtime tree, all helper hashes, the failed F30 receipt and sealed F31 qualification.

## Measured execution

# F32 measured results

Production completed in 7584.90 seconds on CPU. Replay/receipt verification remains a separate requirement.

- training: 177616 games, 177590 natural terminals, 26 cutoffs, 9257583 attempted decisions.
- evaluation: 13824 games, 13820 natural terminals, 4 cutoffs, 758724 attempted decisions.
- selfplay_diagnostics: 576 games, 576 natural terminals, 0 cutoffs, 32824 attempted decisions.

All-seed learning: {'control': False, 'nonlinear': False}; all-seed tactical feature benefit: True; secondary base-power benefit: True. Contrasts: {'improvement': 11, 'inconclusive': 16}.

| Seed | Arm | Base-power wins: initial → final /256 | Tactical wins: initial → final /256 | Learning gate | Familiar final mastery |
|---|---|---:|---:|---|---|
| 7103 | control | 143 → 140 | 28 → 23 | False | False |
| 7103 | nonlinear | 194 → 224 | 58 → 87 | True | False |
| 8209 | control | 142 → 165 | 28 → 35 | False | False |
| 8209 | nonlinear | 198 → 215 | 63 → 85 | False | False |
| 9311 | control | 149 → 156 | 19 → 33 | False | False |
| 9311 | nonlinear | 176 → 213 | 40 → 84 | True | False |

Peak parent RSS 2186.2 MiB (workers excluded). Collection 7191.14s; optimizer calls including compile 131.02s. Finite gradients/likelihood ratios are engineering telemetry, not trust-region or calibration guarantees.

Each arm continues its same-seed F29 final checkpoint with all Adam moments, steps and episode cursors preserved. Absolute cohorts256/512/1024 have cumulative262144/524288/1048576selected rows; the continuation adds786432rows and6144optimizer steps per arm/seed. Selected-row and optimizer budgets match; games and compute are measured, not matched. The existing nonlinear arm has2912 additional actor parameters and permits feature/context interactions with possible shared-trunk feedback; it does not isolate nonlinearity from capacity. Continuation-start reevaluations use the new fixed F32 schedules and contrasts are dependent; all27 stay in the approximate family adjustment. Tactical is trained on and is the primary adaptation gate. Base-power learning and benefit are secondary. Fresh development schedules against recurring generation and known heuristics, not public ladder or held-out-team generalization. All seeds/27 contrasts retained; intervals and cutoff-aware gates remain in measured-review.json. No promotion, final reserved data, public server, Prime or GitHub publishing.

