# Maxcalibur objective review — 25 October 2026

The original goal remains an independent, strong Generation 9 Random Battles agent: correct mechanics and player information, useful self-play RL, scalable experience collection, then beliefs and search where they improve play. The foundation is substantially built. Demonstrated full-battle strength is still far below that goal, and the recent experiment sequence emphasized imitation metrics over demonstrated battle gains.

This review preserves the [initial research handoff](RESEARCH_HANDOFF.md) and its historical claims. It uses the current original technical page, local handoffs, source inspection and admitted evidence; it does not retroactively change gates or declare failed experiments successful.

## What the original project actually did

The author describes an 8.5-million-parameter transformer with event and matchup inputs, win-value and opponent-prediction heads, self-play PPO/GAE, and almost 100 million games over about one H100-week. Search is added at inference, with most strength attributed to the base policy. These are author-reported results, not locally reproduced benchmarks or a required recipe. The public description does not establish that a larger network, more games or search alone will solve our current problems. [Jaxcalibur technical write-up](https://jaxcalibur.github.io/), checked 25 October 2026.

## Where we stand

| Original component | Verified local progress | Remaining gap |
| --- | --- | --- |
| Reference rules, data and information boundaries | Pinned Showdown; complete official teams; versioned actions/player views; request/rejection handling; deterministic replay; prospective train/development/final partitions | Maintain these guarantees as observations and training change. Exact-team splits do not hold out all species or set families. |
| Accelerated simulator | Independent scalar and batched JAX implementations exist for restricted mechanics, with chance mapping and player-view comparisons | Full intended Gen9 Random Battles coverage and production-scale throughput are incomplete. The complete-battle track currently uses Showdown. |
| Full-battle RL | F2/F4/F10 implement actual PPO/GAE and improve over initialization. F10 wins roughly 90% against random, 38–40% against base-power and 4% against tactical in its development panel | Reliable tactical play, stronger opponents, cross-play robustness and sustained useful learning remain unproven. |
| Policy representation | 928 public features; F20's approximately 60,000-parameter shared scorer improves fresh teacher imitation | Current hashed categorical summaries are lossy; no explicit event-token sequence, structured matchup/damage representation or full learned belief system. Architecture is a hypothesis, not an established cause of weak play. |
| Value and opponent predictions | Value learning exists in the full-battle PPO track. Public reveals persist in the adapter | Shifted-opponent value diagnostics are weak; train-distribution calibration needs separate assessment. Hidden-set and opponent-action auxiliary heads are absent. F20 imitation does not train its critic. |
| Search and external strength | Local playable/recoverable Showdown clients exist; search is deferred | No demonstrated public ladder strength, external competitive anchor or information-safe full search implementation. |

Sources: [R7 data foundation](R7_SPRINT.md), [restricted JAX engine](M3_SWITCH_ENGINE.md), [F3 throughput](F3_THROUGHPUT.md), [F10 full-battle RL](F10_LONG_TRANSFER.md), [F20 architecture comparison](F20_SHARED_POLICY.md), and the current [feature encoder](../python/pokemon_full/features.py) and [shared model](../python/pokemon_shared/network.py). The restricted JAX engine is meaningful progress; it is not the complete format.

## What needs to change in our direction

1. **Make battle outcomes the primary learning measure again.** F15–F20 diagnose imitation, fitting and representation. Those probes are useful only when they lead to better complete-battle behavior. F20's 0/3 gate failure stands; relative agreement gains do not remove its Tera-variant failures. Familiar diagnostic mastery and teacher agreement remain supporting metrics.
2. **Keep the Showdown RL track moving independently of the full JAX port.** Using the reference simulator already permits correct full-format experience. A complete accelerated engine is a possible scaling investment, not a prerequisite for every local learning experiment. Its coverage gate must remain distinct.
3. **Test one justified intervention per learning comparison.** Potential changes include shared policy/value scoring, explicit semantic observations, event history and opponent auxiliary heads. Combining all of them would obscure what helped. Prefer complete-battle return, calibrated values and per-opponent regressions over another sequence of tiny fitting targets.
4. **Add robustness and an external anchor.** Historical checkpoint cross-play can expose cycling; a pinned stronger local opponent can broaden the current familiar panel. Neither requires public ladder play. Keep final evaluation data reserved while the design adapts.
5. **Scale useful learning, with measured costs.** The original goal calls for much more sustained learning than these short canaries. Profile simulation, encoding, inference, updates and storage separately, then decide whether more CPU workers, a larger-network GPU workload, or an accelerated engine is the best next investment. CPU sufficiency for a bounded experiment does not settle the eventual training platform.

The replay and provenance work protects conclusions, but engineering completion is not a substitute for a learning result. Future runs should reuse validated components and proportionate regression checks, with full scientific replay at defined integration milestones. The current F21 contract still requires its complete independent replay.

## F21 decision and immediate route back to RL

F21 runs every final F20 flat/shared checkpoint and the F10 rehearsal reference on the same fresh paired schedules. It performs zero updates and retains all seeds, opponents, cutoffs and known mastery failures. It is a development battle test; reference comparisons also differ in training history.

**Measured F21 result:** F21 completed (2026-10-25): evaluated all final F20 flat/shared checkpoints and the F10 rehearsal reference across 6,912 paired-development battle executions in 326.96 seconds on CPU, with zero optimizer updates or checkpoint selection. The shared-versus-flat battle-benefit gate passed in 0/3 seeds. All 27 contrasts remain: 2 improvements, 2 regressions and 23 inconclusive under the registered approximate adjusted bootstrap. Outcomes: 6,908 natural terminals, 4 unresolved cutoffs, 425,793 total attempted decisions and 214,191 learner decisions. All 21 focused tests, exhaustive fresh battle replay and nine immutable checkpoint loads pass. F20 familiar mastery failures remain; no policy is promoted. Full G2/G3/G4 and public strength remain open; no final reserved data, public server, Prime or GitHub transfer.

F20’s stronger teacher imitation did not establish stronger tactical play in complete battles. Descriptive pooled tactical wins were 53/768 for shared (6.90%), 54/768 for flat (7.03%) and 33/768 for the F10 reference (4.30%). Shared-minus-flat tactical differences by seed were 0.00, +1.95 and -2.34 percentage points; all adjusted intervals include zero. The primary battle-benefit gate has 0/3 passing seeds because neither tactical requirement passes in any seed. Shared’s earlier familiar-context failures also reproduce in seeds 8209 and 9311.

The shared model meets the approximate adjusted-bootstrap improvement criterion against flat on random in two of three seeds, with the third inconclusive. Descriptive random wins are 694/768 shared, 611/768 flat and 685/768 reference; there are three unresolved flat cutoffs and one reference cutoff. Against base-power, wins are 324/768 shared, 320/768 flat and 290/768 reference. No shared-versus-reference comparison establishes an improvement. Flat meets the approximate adjusted-bootstrap regression criterion against reference on random in two seeds. All 27 contrasts are retained: 2 improvements, 2 regressions and 23 inconclusive under the approximate adjusted bootstrap; all conservative Hoeffding intervals contain zero. The pooled totals reuse schedules across seeds and are descriptive, not independent-game confidence calculations.

Production completed in 326.96 seconds on CPU, with 6,908 natural terminals, four unresolved cutoffs and no true ties. All 425,793 attempted decisions and 214,191 learner action records are preserved. The 172 unavailable-choice messages were all advertised switch attempts rejected because the active Pokémon was trapped; refreshed-request handling preserves them. No training or optimizer updates occurred. These results do not establish equivalence, a causal explanation of tactical weakness, or competitive/public strength. F10 remains the reference and no policy is promoted.

F21 falls in the mixed/inconclusive decision branch: retain F10 as the reference and do not claim a tactical winner. Next proposed step: return to full-battle RL with a controlled flat-versus-shared PPO comparison, starting from all three matched F20 final checkpoints and retaining F10 as an external reference. The shared scorer remains an unproven candidate: its matched-size implementation and limited random-opponent signal make RL learnability a testable hypothesis, not an established tactical benefit. First qualify the shared actor/value PPO loss, gradients and checkpoint recovery against the existing learner. Then preregister an equal decision/update budget and fixed-opponent learning curves; use battle improvement as the primary outcome, with familiar tactical cases and held-out self-play value diagnostics as supporting measures. Do not extend behavior cloning merely to improve teacher agreement. Structured observations, history, opponent auxiliary heads and historical-opponent cross-play remain separate candidate interventions. Measure complete-cycle cost before selecting a longer CPU or GPU budget. No new training is launched in F21; local CPU sufficed for this evaluation, and Prime remains unprovisioned. See the [F21 handoff](F21_SHARED_BATTLE.md).

| Outcome | Next decision |
| --- | --- |
| Shared improves over matched flat and is competitive with F10 | Treat it as a candidate for a preregistered full-battle PPO comparison. Verify shared actor/critic loss and checkpoint behavior first; do not automatically promote an imitation checkpoint. |
| Both imitation models clearly underperform F10 | Keep F10 as the reference and close the current imitation branch. Any further teacher-based training needs a specific new mechanism and a battle-performance test. |
| Mixed or inconclusive results | Preserve the tradeoff/uncertainty and retain the reference. Do not select the best seed, switch to an earlier endpoint or selectively extend favorable cells. |
| Integrity or execution failure | Preserve evidence and diagnose; make no strength inference. |

The next substantial learning milestone should be a controlled full-battle RL comparison, with a fixed decision/update budget, all three seeds, fixed-opponent evaluation and learning curves. Select its single intervention from F21 and the existing decision audits. Before a larger commitment, use held-out training-distribution value diagnostics and profiling to choose the budget. Auxiliary opponent prediction and historical opponent pools follow as separate ablations; search follows a useful base policy.

## Compute and scope

No Prime resources are needed for F21. A future GPU/Prime pilot could become useful for a larger model or larger batched workload after measured end-to-end throughput and memory justify it. F3's slower Metal result applies to that small tested workload, not every possible model or the M3 Pro GPU in general. No current provider pricing, availability, speedup or training-cost guarantee is asserted.

Full G2/G3/G4 and public competitive strength remain open. No public server, reserved final evaluation or additional training was used for this audit. GitHub remains on the unresolved visibility hold; no transfer is performed.

## Review quality and limitations

A delegated Fable audit and independent Codex QA agreed on the drift toward imitation. The raw draft required substantive corrections: it overlooked restricted JAX work, mischaracterized the familiar variants, and overstated critic, statistical and hardware conclusions. The coordinator checked those claims against source and evidence and rejected the draft as a final deliverable. This corrected review retains the valid findings without treating model opinions as measured evidence. The current engine landscape and external opponents have not been freshly benchmarked in this task.
