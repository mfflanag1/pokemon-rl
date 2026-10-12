# Pokémon battle learning project

[Private GitHub repository](https://github.com/mfflanag1/pokemon-rl) ·
[Data backup and restore](REPOSITORY_STORAGE.md) ·
[Current M3 learning screen](docs/M3_SWITCH_SCREEN.md)

An independent Jaxcalibur-inspired Gen 9 Random Battles project. **R1–R7 are implemented:** a pinned Showdown oracle, versioned Node/Python contracts and action codec, 100,000 validated official teams, 1,000 exploratory reference episodes, player-history features with information-boundary tests, a conservative mechanics catalog, and frozen dataset partitions with provenance gates. A fresh 96-episode campaign provides separate actor/target exports and a local CPU benchmark. **M1/G1 CPU learner canaries pass**, and a fixed-team CPU transition slice matches 10,000 reference cases conditional on recorded chance outcomes. The same fixed duel now has a batched CPU JAX port, exact Showdown finite-word chance maps and player-view parity on 10,000 fresh games. Full G2 remains open. A JAX autodiff learner now passes NumPy parity and all six canaries, and an observation-only collector trains a tiny fixed-duel policy with exact boundary resume; no competitive-strength claim is made.

A [playable Showdown baseline](docs/PLAYABLE_BASELINE.md) now adds a real WebSocket
client and a simple move/switch/Tera heuristic for complete battles. It completed
three local-server games (including a rematch and fresh login), and won 60 of 64
paired development games against a weak random policy. All 2,904 saved decisions
replay exactly. This is a local functional baseline; public login and ladder strength
remain untested, and the custom JAX engine's G2/G3 gates remain open.

The additive [recoverable session](docs/SHOWDOWN_SESSION.md) now completes six
local-server games through seven injected disconnects. All 120 attempted decisions
match the receiving player's reconstructed view, including one deliberately lost
write; an accepted pending action is not replayed. All 303 regression tests pass.
Use `npm run bot:recover` for
this version. [Public deployment requirements](docs/SHOWDOWN_DEPLOYMENT.md) and
account setup remain outstanding; this reliability work adds no playing-strength claim.

The [frozen opponent panel](docs/SHOWDOWN_OPPONENT_PANEL.md) now completes 768
development games and exact replay of all 35,711 decisions. The tactical heuristic
scores 58.59% against the current baseline and 57.03% against the damage heuristic,
but scores slightly worse against random. These descriptive results nominate a
confirmation candidate. The original panel's 338-test regression passes; its
results remain preserved separately from the fresh confirmation below.

The [fresh tactical confirmation](docs/SHOWDOWN_TACTICAL_CONFIRMATION.md) passes
its fixed development screen across **12,000 games**: tactical scores **60.65%**
against baseline, **55.20%** against damage and **95.10%** against random. All
**564,379 decisions** reproduce in fresh simulator replay. Its
paired improvements over baseline against random and damage pass the declared
three-comparison bounds. The opt-in `npm run bot:tactical` client completes six
local-server games through seven injected disconnects, with all 179 attempted
decisions audited. The handoff provides the local launch command. The 380-test regression
passes. Public login and ladder strength remain untested; full G2/G3 remain open,
final data stay reserved and Prime is unnecessary.

The [local network soak](docs/SHOWDOWN_NETWORK_SOAK.md) now completes **1,000 games**
across 20 sessions and 60 planned disconnects. All **46,024 decisions** pass the
independent wire audit, with zero unavailable-choice retries or campaign failures.
Peak sampled client RSS is 445.063 MiB; all **426 tests** pass. This completes the
local adapter volume prerequisite for the frozen tactical client. Public play,
full G2/G3/G4 and competitive ladder strength remain unqualified. The subsequent
M3 engine milestone is recorded below; Prime is unnecessary.

The [M3 switching/hidden-set engine](docs/M3_SWITCH_ENGINE.md) now passes **10,000
fresh Showdown games**, **273,270 player-view comparisons**, and all 72 sealed
reference cases with zero divergence. The four-action categorical learner completes
**192 updates, 6,144 primary games and 140,983 decisions**, with six exact original
next-update replays, nine checkpoint loads and **443 passing tests**. This is a
restricted engineering smoke; its subsequent sustained CPU qualification is recorded
below. A separate learning-strength screen remains next. Full G2/G3/G4 and public ladder strength stay open; final data
remain unused and Prime is unnecessary.

Start with [the runnable R1 guide](docs/R1_REFERENCE.md), then [R2 setup and usage](docs/R2_CONTRACT.md) for the local Node/Python dependencies. Add the CPU learner dependency with `.venv/bin/python -m pip install --only-binary=:all: -r requirements-learner.txt`. Run `npm test`, `npm run smoke`, and `npm run contract:smoke` to verify the reference and contract.

The [R3 corpus guide](docs/R3_CORPUS.md) covers generation, crash recovery, independent Python validation and sampled replay. Read the [inventory handoff](docs/R3_INVENTORY.md) for measured coverage, data locations and the vocabulary revision discovered during collection.

The [R4 campaign handoff](docs/R4_CAMPAIGN.md) records 81,947 verified decisions/transitions and 13 additional fixtures. Use the [episode run guide](docs/R4_EPISODES.md) for generation, replay, validation and the R5 handoff.

The [M3 sustained CPU qualification](docs/M3_SWITCH_SOAK.md) now passes
**7,200.008 seconds**, **1,711,484 updates** and **1,173,453,330 decisions**.
Of 54,767,488 capped games, 50,278,791 terminate and 4,488,697 truncate explicitly.
All **13 original-next-update checkpoint audits** pass, with zero unexpected
compilations and **460.578 MiB** peak process RSS. The unchanged source-bound
regression passes **460 tests**. The [receipt](artifacts/reports/m3-switch-soak.json)
qualifies restricted CPU execution; full G2/G3/G4 and public ladder strength remain
open. Next preregister the separate learning-strength screen. Prime is unnecessary.

The [R5 feature handoff](docs/R5_FEATURES.md) covers the encoder, separate actor/target types, live leakage probes, full-campaign validation and R6 starting point.

**Recommended order:** pin Showdown → establish reference contracts and simulator fidelity → train and evaluate a useful policy → add beliefs and search → scale only demonstrated improvements.

1. [Full implementation plan](docs/IMPLEMENTATION_PLAN.md) — decisions, milestones, architecture, data, learning, evaluation, budget, and schedule.
2. [First sprint](docs/SPRINT_01.md) — seven executable tickets with ownership, dependencies, and acceptance criteria.
3. [Acceptance gates](docs/ACCEPTANCE_GATES.md) — detailed correctness, learning, resume, and evaluation requirements.
4. [Prime Intellect assessment](docs/PRIME_INTELLECT.md) — useful infrastructure and optional later integrations.
5. [Original research handoff](docs/RESEARCH_HANDOFF.md) — source investigation and distinctions between published claims and independent reconstruction.
6. [Review and verification record](docs/REVIEW_RECORD.md) — checks actually performed and unresolved limitations.

The [R6 mechanics handoff](docs/R6_MECHANICS.md) covers 3,509 conservative inventory entries, source-bound dispatch, implementation obligations and 16 scripted reference fixtures.

The [R7 sprint handoff](docs/R7_SPRINT.md) records the frozen 90/5/5 policy, legacy exposure audit, 128 reserved final teams, admission API and benchmark. Use its trusted completion receipt when opening a dataset for training or development.

The [M1/G1 handoff](docs/M1_G1.md) records PPO/GAE checks, three-seed learning, exact checkpoint/repeat results and the restricted Rattata duel. The [M1/G2 handoff](docs/M1_G2.md) records the batched JAX port, chance-law checks and player-view parity. The [JAX learner handoff](docs/M1_JAX_LEARNER.md) records numerical parity, all six canaries, observation-only collection and exact disk recovery. The [fixed-loop handoff](docs/M1_FIXED_LOOP.md) adds a complete fixed-shape executable, a passing two-hour CPU soak, profiling and 37 exact disk-replay comparisons. The [binary-policy handoff](docs/M1_BINARY_POLICY.md) resolves the two-action sampling-tail mismatch with exact integer-word selection, matching PPO logs and a short CPU qualification. The [Showdown-chance handoff](docs/M1_SHOWDOWN_CHANCE.md) adds exact finite-word maps, 10,000 fresh player-view games with zero disagreements, and new-sampler collect/update and checkpoint checks. The selected combination has now passed its own [two-hour CPU qualification](docs/M1_POLICY_SOAK.md). Full G2 and G3 remain open for hidden-set and broader mechanics coverage, frozen-opponent evaluation across training seeds, critic calibration and memory/budget evidence. Local CPU remains sufficient; Prime is unnecessary at this stage. All training compute amounts in the plan are estimates for a future spending decision.

The [restricted-policy development pilot](docs/M2_POLICY_PILOT.md) now passes its preregistered learning screen in all three training seeds. Across 54,000 pinned Showdown games, final scores versus random are 63.65%, 62.35% and 62.30%, with positive adjusted paired improvement bounds over initialization. Scores versus damage-greedy remain about 50%. Every recorded candidate action/value was replayed exactly, and 251 regression tests pass. This is measured learning in the tiny public duel; full G2/G3 remain open, final data remain reserved and Prime remains unnecessary.

The [M3 switching/hidden-set reference scope](docs/M3_SWITCH_REFERENCE.md) is now
complete: 24 constructed two-member teams, four semantic actions, 72 replayed
reference scripts, 1,872 hidden-state invariance comparisons, 5,426 independently
reconstructed actor inputs and 3,456 conditional damage values. All 264 tests pass.
This establishes the reference contract for the scalar/JAX engine and learner
now completed in the subsequent M3 engine milestone above. Full G2/G3 remain open, final data stay reserved and
Prime remains unnecessary. The [M3 receipt](artifacts/reports/m3-switch-reference.json)
binds the source and evidence.

The implementation plan and acceptance gates govern new work; they refine the earlier handoff's proposed schedule and defaults.

The selected combination completed 7,200.007 seconds, 8,158,970 collect/update cycles and 3,272,073,210 decisions in the fixed public duel, with zero unexpected compilations and 13 exact checkpoint replay audits. Peak process RSS was 473.766 MiB. The [completion receipt](artifacts/reports/m1-policy-soak.json) binds the run and unchanged 235-test regression. These are engineering results, not a policy-strength evaluation; final data remain reserved.
