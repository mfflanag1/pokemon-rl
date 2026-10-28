# F25: own roster identity repair and fixed full-battle matchup comparison

Status: fixed production and exhaustive replay passed; completion sealing follows the measured documentation below.

F23 and F24 both stopped before candidate training. Their source, plans, runs and failure receipts remain immutable. F24 committed 656 reference games before a terminal Illusion observation marked both Zoroark and its benched disguise active. The failed batch is engineering evidence, not a new learning result. F25 is a separately versioned repair and execution of the same comparison, authorized after F24 failed.

The new own reducer keys HP/status, active flags, boosts and persistent Tera reveals by true private roster identity. It processes each player's own private requests in protocol order. Position-addressed public events update the current entry; bench-addressed events update the named private member. Submitted switch slots resolve voluntary/forced entrants, while Revival Blessing is not treated as a switch. Illusion `replace` retains accumulated boosts and condition. Private requests rebase HP/status and active flags, so terminal rendering cannot resurrect stale nickname overlays. The opponent reducer, public dynamic-type tracker, frozen tactical opponent, semantic action space and pinned simulator are reused unchanged.

Unrequested ambiguous entries are buffered until that entry's private request or public reveal resolves identity. An unresolved entry at an observation or before another entry fails explicitly; there is no guessed identity, silent skipped game or disabled singles invariant. Drag candidates exclude the known outgoing member and consider both current and base Illusion ability; submitted non-Illusion entrants must match their public appearance. This is a tested repair, not complete proof of every Gen9 protocol combination. Silent switch-out healing and unsupported boost-transfer messages can still leave terminal supplements stale until a private request; the exact-HP terminal fixture does not establish universal terminal-state fidelity. Full G2 remains open. Neither the engine's private state/end payload nor the other player's private request enters an actor observation.

The actor payload remains `full-showdown-player-v2` with six known own members, public opponent slots and 14 semantic actions. The new transport/frame is `terminal-matchup-worker-v1` / `terminal-matchup-frame-v1`. Both arms receive the same corrected view. The sealed F23 network and PPO learner are reused: 928 base features plus 350 nominal public matchup channels, 60,806 parameters, zero-initialized policy/value residuals. The zero arm validates then zero-masks the extra channels. Nominal features omit ability/item effects, dynamic move type/power and Stellar modifiers; they are not simulator damage. Active capacity and clipping differences are part of the intervention. Historical v1 tactical probes retain their original projection limitations.

All three final F22 flat hosts (7103, 8209, 9311) retain base weights and reset Adam identically. Each arm/seed runs 512 complete-game PPO cohorts, four times the F22 budget: 3,072 cohorts, 24,576 optimizer updates, 3,145,728 selected rows and 6,291,456 presentations. Eight-game chunks continue until 1,024 rows per cohort, capped at 64 games, with same-player GAE before uniform row selection. Games retain the 120-turn administrative cutoff. This matches selected-row/update budgets, not wall time or collected game count. No behavior-cloning rehearsal, history extension, auxiliary heads or opponent-pool change.

Evaluation at cohorts 0, 128 and 512, plus the same-seed frozen F10 reference, uses random/base-power/tactical opponents: 16,128 main games and 576 separate self-play value-diagnostic games. All three seeds and 45 final contrasts remain. Learning requires at least five percentage points of cutoff-aware base-power improvement, positive adjusted approximate-bootstrap lower bound, no detected opponent regression and final cutoffs at most five percent. Feature benefit applies the same threshold to final matchup-versus-zero tactical results and retains random/base-power regression checks. All seeds must pass for an all-seed claim. Midpoints, pooled rates and familiar tactical mastery are descriptive. No automatic promotion.

F25 uses newly fixed train/optimizer namespaces and development seeds. Exact-team historical exclusion includes F24's 656 committed games and all eight failed-batch coordinates, as well as the F23/F22 admitted histories. Historical and engineering exposures are not an exhaustive project-wide inventory. Evaluation is fresh-game development against recurring official generation and known heuristics, not reserved or public ladder generalization.

Qualification covers the captured failure mechanism and exact F24 failed batch in both arms, a real six-KO terminal battle under unrevealed Illusion, bench isolation, status/boost/Tera ownership, request reconciliation, switch hints and fail-closed ambiguity; F24 type fixtures, F23 failed batch, historical probes, learner gradients and complete collect/update/checkpoint linkage are retained. A production result requires exhaustive fresh replay of all games/updates, 48 checkpoint comparisons and 12 original-next-update recoveries. A failure preserves evidence and stops without restarting, changing seeds/budgets or editing frozen sources.

Completion requires measured documentation before sealing, all live/archived source and evidence hashes, and an identical repeat receipt. The workflow lives in `artifacts/reviews/f25-terminal-matchup`; production will use `artifacts/runs/f25-terminal-matchup-v1`. Post-seal logs stay outside the sealed review directory. Full G2/G3/G4 and public ladder strength remain open. No reserved-data use, public server, Prime or GitHub publishing; the existing visibility hold persists.

Qualification binds 669 source/configuration files. The independently reviewed identity fixes and limits are recorded in `artifacts/reviews/f25-terminal-matchup/review-dispositions.md`; earlier draft checks remain as development evidence. The 88 checks and predecessor admission passed before launch.

## Measured execution

# F25 measured results

Production completed in 3549.77 seconds on CPU. Replay/receipt verification remains a separate requirement.

- training: 60832 games, 60792 natural terminals, 40 cutoffs, 3939859 attempted decisions.
- evaluation: 16128 games, 16125 natural terminals, 3 cutoffs, 959163 attempted decisions.
- selfplay_diagnostics: 576 games, 576 natural terminals, 0 cutoffs, 36637 attempted decisions.

All-seed learning: {'matchup': False, 'zero': False}; all-seed feature benefit: False. Contrasts: {'improvement': 8, 'inconclusive': 37}.

| Seed | Arm | Base-power wins: initial → final /256 | Tactical wins: initial → final /256 | Learning gate | Familiar final mastery |
|---|---|---:|---:|---|---|
| 7103 | zero | 134 → 135 | 21 → 32 | False | False |
| 7103 | matchup | 134 → 143 | 21 → 37 | False | False |
| 8209 | zero | 125 → 139 | 19 → 18 | False | False |
| 8209 | matchup | 125 → 132 | 19 → 26 | False | False |
| 9311 | zero | 126 → 123 | 22 → 16 | False | False |
| 9311 | matchup | 126 → 150 | 22 → 24 | False | False |

Peak parent RSS 1939.4 MiB (workers excluded). Collection 3329.17s; optimizer calls including compile 80.47s. Finite gradients/likelihood ratios are engineering telemetry, not trust-region or calibration guarantees.

Nominal public type features; same augmented inventory and zero initialization, different active capacity. Fresh development schedules against recurring generation and known heuristics, not public ladder or held-out-team generalization. All seeds/45 contrasts retained; intervals and cutoff-aware gates remain in measured-review.json. No promotion, final reserved data, public server, Prime or GitHub publishing.

