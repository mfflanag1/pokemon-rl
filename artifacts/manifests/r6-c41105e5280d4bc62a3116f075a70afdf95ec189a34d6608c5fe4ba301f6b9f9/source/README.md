# Pokémon battle learning project

An independent Jaxcalibur-inspired Gen 9 Random Battles project. **R1–R6 are implemented:** a pinned local Showdown reference harness, versioned Node/Python artifacts, a semantic action codec, a validated 100,000-team corpus with an inventory, a request-aware runner with 1,000 replayed and independently validated episodes, and verified player-history features with information-boundary tests, and a conservative mechanics manifest with a reference fixture catalog. No custom accelerated simulator or trained model has been created.

Start with [the runnable R1 guide](docs/R1_REFERENCE.md), then [R2 setup and usage](docs/R2_CONTRACT.md) for the local Node/Python dependencies. Run `npm test`, `npm run smoke`, and `npm run contract:smoke` to verify the reference and contract.

The [R3 corpus guide](docs/R3_CORPUS.md) covers generation, crash recovery, independent Python validation and sampled replay. Read the [inventory handoff](docs/R3_INVENTORY.md) for measured coverage, data locations and the vocabulary revision discovered during collection.

The [R4 campaign handoff](docs/R4_CAMPAIGN.md) records 81,947 verified decisions/transitions and 13 additional fixtures. Use the [episode run guide](docs/R4_EPISODES.md) for generation, replay, validation and the R5 handoff.

The [R5 feature handoff](docs/R5_FEATURES.md) covers the encoder, separate actor/target types, live leakage probes, full-campaign validation and R6 starting point.

**Recommended order:** pin Showdown → establish reference contracts and simulator fidelity → train and evaluate a useful policy → add beliefs and search → scale only demonstrated improvements.

1. [Full implementation plan](docs/IMPLEMENTATION_PLAN.md) — decisions, milestones, architecture, data, learning, evaluation, budget, and schedule.
2. [First sprint](docs/SPRINT_01.md) — seven executable tickets with ownership, dependencies, and acceptance criteria.
3. [Acceptance gates](docs/ACCEPTANCE_GATES.md) — detailed correctness, learning, resume, and evaluation requirements.
4. [Prime Intellect assessment](docs/PRIME_INTELLECT.md) — useful infrastructure and optional later integrations.
5. [Original research handoff](docs/RESEARCH_HANDOFF.md) — source investigation and distinctions between published claims and independent reconstruction.
6. [Review and verification record](docs/REVIEW_RECORD.md) — checks actually performed and unresolved limitations.

The [R6 mechanics handoff](docs/R6_MECHANICS.md) covers 3,509 conservative inventory entries, source-bound dispatch, implementation obligations and 16 scripted reference fixtures.

Next is R7 in the first sprint: dataset partitioning, provenance gates and the benchmark report. Local CPU remains sufficient. All training compute amounts in the plan are estimates for a future spending decision.

The implementation plan and acceptance gates govern new work; they refine the earlier handoff's proposed schedule and defaults.
