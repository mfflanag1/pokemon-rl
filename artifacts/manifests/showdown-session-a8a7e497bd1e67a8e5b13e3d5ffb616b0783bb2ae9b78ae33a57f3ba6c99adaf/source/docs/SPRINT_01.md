# Reference and data foundation: proposed first sprint

Status 2026-10-10: **R1–R7 complete**, documented in [R1_REFERENCE.md](R1_REFERENCE.md), [R2_CONTRACT.md](R2_CONTRACT.md) and [R3_CORPUS.md](R3_CORPUS.md). R3's measured results and remaining boundaries are in [R3_INVENTORY.md](R3_INVENTORY.md). R4’s [episode campaign and handoff](R4_CAMPAIGN.md) records 1,000 replayed and independently validated episodes. R5’s [feature handoff](R5_FEATURES.md) records the verified player-history boundary and full-campaign checks. R6’s [mechanics handoff](R6_MECHANICS.md) records a conservative source-bound inventory and reference fixture catalog. R7’s [sprint report](R7_SPRINT.md) records frozen partitions, exposure exclusions, 96 fresh mirrored episodes, actor admission and the CPU benchmark. The reference/data foundation (M0) is complete; M1/G1 learner canaries are next. Original planning estimate for the whole sprint: approximately 13–18 engineer-days plus integration buffer, or two weeks with two engineers. Its exit deliverable is a pinned, replayable Gen9 Random Battle reference corpus and a stable Python contract. A faithful accelerated battle engine and large RL run are subsequent work.

## Package boundaries

```text
vendor/pokemon-showdown/          exact upstream commit; no implicit updates
packages/reference-node/          owns Dex, format resolution, team factory, BattleStream
packages/corpus/                  official generation, resumable shards, sampled replay
contracts/battle-v1/              JSON Schema + vocabulary + example documents
python/pokemon_reference/         portable validators, action codec, corpus inventory
python/project/data/              future partitions and corpus loaders
python/project/observation/       future player-history -> model features
python/project/engine/            later: canonical state <-> JAX PyTree; reset/step
tests/reference/                 Node reproducibility/contract tests
tests/differential/              later: reference vs JAX at decision boundaries
artifacts/manifests/              immutable content-addressed collection metadata
```

Node is the sole authority for upstream engine calls, compiled Dex extraction, official whole-team generation, and reference state/protocol capture. Python consumes versioned artifacts; it does not import or reinterpret TypeScript callback definitions. JAX owns fixed-shape tensors and accelerated semantics, not the initial team-generation algorithm. Avoid making WebSockets or a fresh subprocess per battle the bulk-collection interface. A long-lived Node JSONL worker can stream records for debugging and collection; training consumes local binary/sharded data derived from those records.

Node exports narrow commands: `describe`, `generate_teams`, `run_episode`, and `replay_episode`. Outputs are JSONL with a request ID and typed success/error records. An episode starts from explicit serialized teams and an explicit reference RNG seed. Actions are submitted against a named decision ID. Malformed, stale or observably forbidden choices are implementation errors; rule-defined hidden-legality rejections are ordinary feedback/re-request transitions. Neither silently selects a default. Rule/profile selection is fixed in the manifest; a custom testing profile must use a distinct profile ID.

Use direct local `BattleStream`/simulator calls for fixed-team replay. Current upstream `getTeam` returns an explicitly supplied team before generation, and `setPlayer` calls it; verify this at the selected SHA. R1/R4 must assert initial canonical team hashes in both seats and record separate generator/mechanics seeds. A network-server route may apply different team validation and requires its own adapter test. [Current Showdown source](https://github.com/smogon/pokemon-showdown/blob/master/sim/battle.ts)

## Portable contract

Use JSON Schema Draft 2020-12 and check documents independently in Node and Python. Every artifact has `schema_version`, `artifact_kind`, `artifact_id`, `manifest_id`, and `content_hash`. Major versions change meaning/required fields; minor versions add optional fields. Unknown major versions are rejected. Define canonical JSON bytes for hashing: UTF-8, sorted object keys, no insignificant whitespace, no NaN/infinity, normalized integers, and precisely specified string escaping. Never use runtime hash functions for IDs.

- **Manifest:** Showdown SHA and dirty status; package lock hash; resolved `gen9randombattle` rules; PotD/server configuration; generator/data hashes; source license hashes; Node/Python versions; extraction version; RNG kind/seed encoding; observation/action vocabulary versions. Store actual source snapshots or content hashes; a branch URL is provenance, not a pin.
- **Team:** team ID; official generator seed and seed kind; ordered six slots; canonical species/form identifiers; level; exact EV/IV values; ability; item; ordered moves; tera type; gender and other generated fields; generator metadata stored separately from policy inputs. Preserve official output instead of rebuilding it from guessed stat defaults.
- **Episode:** episode ID; both team IDs; initial reference RNG seed; policy/opponent IDs; split; input-choice tape; terminal reason and winner. For each decision, record `decision_id`, turn, phase, acting-side set, each side's private request plus visible protocol delta, semantic legal choices, submitted choices, and transition references. Record whether a step is a joint normal decision, forced switch, pivot/replacement decision, or terminal transition.
- **Action:** semantic tagged union, e.g. `move {slot, terastallize}` or `switch {slot}`. Slots are explicitly zero-based in storage. Encode protocol strings only in Node; convert to policy tensor indices only in the Python action codec. Specify exceptional pass/wait/forced-choice behavior; do not invent a normal-action placeholder that leaks into training.
- **Observation:** separate viewer side, public revealed history, own private request, known-value flags, unknown-value markers, and action mask. Numeric zero must never also mean unknown. Opponent exact HP, unrevealed sets, and RNG state cannot enter actor observations. Physical full-state records use a separate schema and storage path.
- **Canonical reference state:** an audit representation of mechanics-relevant fields with stable identifiers, documented units, and null semantics. It is initially comparison-only, not a promise that arbitrary midbattle states can be restored. Keep raw protocol/internal diagnostic traces beside normalized fields so a normalization bug does not erase evidence.

For 64-bit RNG/state values, use tagged hex strings or split uint32 words; avoid JSON numbers outside the exact IEEE-754 integer range. Policy observations, full reference states, and search determinization inputs are distinct types and cannot be passed interchangeably.

## Seven tickets

| ID | Deliverable / owner boundary | Acceptance criterion | Dependencies | Proposed effort |
|---|---|---|---|---|
| R1 | Pin and build Showdown oracle; Node owner | Rebuild from recorded SHA/lockfile; resolve exact format/rules/PotD; one seeded episode's normalized mechanics/protocol replays identically twice; fixed teams retain their hashes in both seats; no network dependency during replay. | None | 0.5–1 day |
| R2 | Contract schemas, semantic choices, vocabularies; joint contract owner | Node/Python validators accept shared golden examples and reject mismatched major versions, unsafe integer seeds, malformed choices, unknown/zero ambiguity. A schema round trip preserves canonical hash. | R1 for exact source facts | 1.5–2 days |
| R3 | Official whole-team factory and corpus inventory; Node/data owner | Generate an initial 100,000 complete teams; every record validates; replay a sampled seed set exactly; emit species/move/item/ability/form/tera/level and joint-set frequencies plus coverage report. Corpus generation has no silently dropped teams. | R1, R2 | 1.5–2 days |
| R4 | Episode runner, request-aware phase state machine, input-tape replay; Node owner | Run at least 1,000 local seeded episodes with legal baseline agents; reproduce normalized mechanics/protocol outputs on replay; fixtures include normal joint choices, pivots, single/double forced replacements, hidden-legality re-requests, and terminal/truncation handling; stale/known-invalid choices fail explicitly. | R1, R2 | 3–4 days |
| R5 | Player-history projection and action codec; Python owner | Node legal choices round-trip through Python semantic/tensor codec; own private data is preserved, unknown opponent fields stay unknown; paired hidden-state perturbation tests leave actor observations/action masks unchanged when the player's history is unchanged. | R2, R4 | 2–3 days |
| R6 | Reachable-mechanics manifest and fixture catalog; engine owner | Every initial corpus move/item/ability/form has a source reference and status. Inventory explicitly expands indirect/generated effects and has no unclassified dynamic-dispatch pathway. Each supported entry has an owning implementation ticket and fixture family; unsupported behavior fails loudly. | R1, R3; R4 for fixture capture | 2–3 days |
| R7 | Dataset partitioning, provenance gates, and sprint benchmark report; data/eval owner | Split deterministically before match pairing; detect shared seeds, exact teams, episodes and mirrored perspectives across splits; prevent privileged fields from actor loader; report collection throughput, bytes/episode, failure rate, and reproducibility results with hardware and counts. Final-evaluation manifest remains sealed from tuning. | R2–R6 | 2–3 days |

Counts are proposed engineering gates, not evidence of existing performance or sufficient mechanics coverage. R6 is an inventory and test design task; it does not claim all mechanics can be implemented within this sprint.

## Fixture capture limitations and controls

V1 reproducibility is **replay from initial teams, reference RNG state, and complete input tape**. A public replay often lacks private requests, unrevealed sets, exact HP, or suitable RNG provenance, so it cannot automatically become a full-state transition fixture. Public logs can still test protocol parsing and known visible transitions; label their uncertainty explicitly.

Normalize only declared nonmechanical fields such as wall-clock timestamps and run identifiers for equality checks, while retaining raw traces. Version and test that normalization; it must never discard a mechanics or information-reveal mismatch merely to make a fixture pass.

An arbitrary Showdown object dump is not a portable restart state. Effects may depend on queue position, callbacks, prior move history, counter provenance, pending choices and hidden condition state. Start with deterministic scripted scenarios replayed from battle initialization. Add a test-only state-construction adapter only for fields and phases that have been audited; label constructed cases separately from natural-play fixtures. Do not patch the production rules or pretend constructed states are reachable.

Matching seeds across Showdown and JAX does not guarantee matching chance events. Initially require deterministic reference fixtures and a trace of random draws/events. Later either preserve the exact pinned PRNG call schedule or introduce a controlled semantic randomness interface for differential cases, with separate statistical tests of stochastic outcomes. A test that merely forces all rolls to one value has inadequate distribution coverage.

## Reachability and support policy

The sampled corpus is a frequency estimate, not a proof of complete support. Inspect all alternatives in the official set/generator inputs, then close the mechanics inventory over move-calling/copying/transforming abilities and moves, form changes, tera-driven behavior, indirect moves, reflected/repeated moves, and conditions they introduce. Examples to classify include Sleep Talk, Metronome-like calls where available, Copycat/Mimic/Sketch where reachable, Transform/Imposter, Instruct/repeat effects if reachable, and Struggle. Do not assume every example is actually present in the pinned format: classify reachable, unreachable-with-reason, or unresolved.

Each mechanics entry records source location/hash, reachability justification, handler ID, interactions, fixtures, and support state: `implemented-tested`, `implemented-unverified`, `unsupported`, or `unreachable`. Compilation/data loading must reject a team/profile requiring unsupported reachable mechanics. Any deliberately reduced training environment receives a different environment ID and separately reported results; it cannot inherit the full Gen9 fidelity claim.

## Split and leakage policy

Proposed partition: approximately 90% train, 5% development, 5% final evaluation by `SHA256(split_version || fixed_salt || canonical_team_content)`, with fixed threshold ranges declared before generation. Use the ordered, mechanically relevant whole-team content as the single grouping key; omit nonmechanical nicknames and generation-seed provenance. Duplicate exact teams always share a partition, including when different seeds generate them. Online generation accepts only the requested partition and records rejection counts. This is a deterministic team holdout, not an individual-set holdout. Match only teams assigned to the same partition; mirrored episodes and both player perspectives stay together. Global rule tables and public generator priors may be shared, but final team samples, opponent checkpoints, outcomes and targeted fixes selected from final failures cannot enter tuning. Freezing the split prevents repeated evaluation from quietly turning the final set into development data.

Keep three independent random streams/namespaces for team generation, battle mechanics, and policy/action sampling. Keep opponent pool version and pairing scheme in each collection manifest. Separately maintain a future-format or later-generator snapshot for drift evaluation; do not silently regenerate the original benchmark after upstream updates.

Privileged state is permitted for transition verification and explicitly declared training targets. The baseline actor and value head both consume only PlayerView; a privileged critic is outside the initial scope. Player histories and masks include only information available at that decision. Future reveals and the opponent's simultaneously selected action cannot appear in features. They may supply explicitly tagged auxiliary labels, never teacher-forced actor inputs. A privileged teacher label is not proof of deployable performance.

## Dependency graph and sprint exit

```text
R1 -> R2 -> R3 -> R6 --+
         \-> R4 -> R5 --+-> R7 -> first-sprint exit
               \-> R6 -+
```

After R2, Node can develop R3/R4 while Python implements loaders/codecs against golden fixtures. R6 can begin from source tables and finish after corpus/runner outputs arrive. Reserve integration time for phase mismatches and hidden-information projection errors. Sprint exit requires reproducibility, validated schemas, explicit support inventory, isolated data splits, and visible failure accounting. It does not authorize or justify a large GPU spend until accelerated engine correctness and end-to-end throughput are measured.
