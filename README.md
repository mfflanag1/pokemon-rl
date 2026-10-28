# Pokémon battle learning project

A research implementation for Gen 9 Random Battles: a pinned Pokémon Showdown reference engine, player-only observation contracts, full-battle PPO, structured policy inputs and observable history, fixed opponent panels, checkpoint recovery, and independently checked experiments.

## Current status — 28 October 2026

**F40 re-verified the saved F39 production in a separate execution and recomputed its measured results, without repeating training.** All 66 prescribed replays, six saved-update recoveries and nine checkpoint comparisons passed. The replay execution reused the original verifier. A separate postcheck confirmed hashes, complete inventories, report accounting and preservation of the original F39 failure; F39 itself has not been relabeled successful.

The verified continuation covered all three seeds, 36,768 training games, 4,608 evaluation games and 6,144 new optimizer steps. **None of the three seeds passed the primary improvement gate.** Against the familiar, training-excluded baseline, wins per 512 games changed 252→269, 236→235 and 248→252. All twelve adjusted contrasts include or touch zero. This does not establish either a reliable improvement or an absence of benefit; no detected regression does not establish noninferiority. See the [F40 measured handoff](docs/F40_VERIFICATION_REPAIR.md) and preserved [F39 failure handoff](docs/F39_ONLINE_CONTINUATION.md).

The [F34 structured-policy screen](docs/F34_POLICY_RESET.md) provides the working policy stack. [F35](docs/F35_TEACHER_ASSISTED.md), [F36](docs/F36_TEACHER_RUNTIME.md) and [F37](docs/F37_TEACHER_LEGALITY.md) failed during teacher acquisition. [F38](docs/F38_TEACHER_COMPATIBILITY.md) found no contained adapter-only repair for the pinned native teacher's revival boundary. That teacher-assisted comparison remains incomplete. Full G2/G3/G4 qualification and public ladder strength remain open; no policy promotion is claimed.

## Repository contents and reproduction

- `packages/`: Node simulator interfaces, player projections, local clients, and opponent adapters.
- `python/`: encoders, learners, collectors, experiment drivers, verification, and process supervision.
- `contracts/`: versioned schemas, fixed plans, vocabularies, and compact catalogs.
- `tests/`: source tests and curated fixtures.
- `docs/`: scientific designs, measured handoffs, failures, and historical reviews.

This source update includes code, tests, contracts, documentation, and dependency configuration. It does not upload new experiment artifacts, raw run logs, checkpoints, external teacher data, vendor checkouts, or environments. Earlier committed artifacts and the historical release remain part of repository history. Evidence links in handoffs can refer to local archives absent from this source snapshot; a source-only checkout cannot reproduce a sealed campaign without its separately preserved evidence and exact runtime. Publication copies normalize local filesystem links while the original frozen files remain unchanged. See [storage and restoration](REPOSITORY_STORAGE.md).

Start with the [R1 setup guide](docs/R1_REFERENCE.md) and [R2 contract guide](docs/R2_CONTRACT.md). The root configuration pins Node and dependencies; the reference checkout is bootstrapped from `showdown.pin.json`. Python dependencies are recorded in `requirements-contract.txt`, `requirements-learner.txt`, and `requirements-jax.txt`. Historical experiment commands and tests may additionally require archived inputs. The [external adapter guide](packages/external-benchmark/README.md) records its separate runtime and upstream license provenance.

The [implementation plan](docs/IMPLEMENTATION_PLAN.md), [acceptance gates](docs/ACCEPTANCE_GATES.md), and [review record](docs/REVIEW_RECORD.md) contain historical chronology. Their older status paragraphs are not the current project status. The [strategy review](docs/STRATEGY_REVIEW_2026_10_27.md) records the motivation for the structured-policy direction.
