# R7: frozen partitions, provenance gates and sprint benchmark

R7 completes the reference/data foundation (M0). The project now has deterministic
whole-team partitions, an exposure registry, fresh same-split mirrored episodes,
separate validated actor/target files, a sealed final-team reservation and measured
local collection costs. No trained policy, accelerated simulator or search was
added. Local CPU was sufficient; no Prime resources were provisioned.

## Partition and exposure decisions

The fixed [policy](../contracts/dataset-v1/policy.json) uses a versioned SHA-256
digest of ordered team content, with exact full-width integer thresholds for
90% train, 5% development and 5% final. Team and move order, resolved forms,
abilities, items, Tera, levels, gender and EVs/IVs are retained. Generation seeds,
nicknames, shiny flags and generator bookkeeping do not move a team between
splits. New mechanical fields reject. This holds out exact ordered teams, not
individual sets, base species, unordered teams or every equivalent cosmetic form.

R3/R4 were already inspected. Their files remain unchanged and exploratory.
The exposure audit verified their byte-bound certificates and records **102,000
unique team identities, 105,000 seeds and 1,000 episode keys**. No duplicate whole
team identities were found in these inputs.

| Existing source | Prospective train teams | Development | Final-assigned but exposed |
|---|---:|---:|---:|
| R3: 100,000 teams | 89,957 | 5,035 | 5,008 |
| R4: 2,000 episode teams | 1,799 | 97 | 104 |

Of the 1,000 legacy battles, 182 would mix prospective splits; 812 have two
train-assigned teams, five two development-assigned teams and one two final-assigned
teams. **None is retroactively promoted to a sealed evaluation battle.** All
legacy identities/seeds are excluded from the fresh R7 collection.

## Fresh reservation and campaign

The durable collection plan was written before candidate generation and pairing.
The salt, thresholds and requested counts were not tuned after observing samples.

| Split | Accepted teams | Generator attempts | Rejected for other partition | Battles |
|---|---:|---:|---:|---:|
| Train | 64 | 70 | 6 | 64 |
| Development | 32 | 538 | 506 | 32 |
| Final | 128 | 2,473 | 2,345 | **0** |

No fresh candidate was rejected for an exposed/duplicate team or seed overlap.
Consecutive accepted teams are paired within train/development; each pair is
played in both seats. This produces **48 paired matchups and 96 episodes**, with
**7,782 decisions and 3,438 turns**. Mirrored games share a mechanics seed and
swap policy seeds with their logical teams. They are correlated observations,
not 96 independent strength evaluations.

Every episode completed and reproduced on serialized input-tape replay. The
accepted campaign has **0 collection failures out of 96 episodes**. Two legitimate
hidden-legality rejections were recorded as ordinary request transitions.
The first small preflight failed to start workers because of an inherited Node
stdin flag; that failure was fixed and retained separately, followed by a passing
four-episode canary. It is not hidden inside the accepted campaign's zero count.

The final 128 teams are held separately, and only aggregate counts/hashes appear
in this report. They have **no episodes, outcomes or actor exports**. Automated
schema/provenance checks inspect their bytes; no final matchup was evaluated.
The current actor loader has no final access mode. This is a workflow/API seal
and integrity commitment, not encryption against the filesystem owner.

## Measured CPU benchmark

Host: Apple M3 Max, 14 logical CPUs, 96 GiB RAM, Darwin 25.6.0 arm64.
Runtime: Node v25.6.1, Python 3.14.3. Four long-lived Node collection workers
were used. No other repository test/campaign was intentionally run concurrently
with the measured collection or export stages; this was an ordinary shared desktop,
not a controlled hardware lab.

Collection took **27.391 seconds**, including worker startup, serialized replay,
compression and storage: **3.504 episodes/s and 284.107 decisions/s**. Team
reservation took a separate 0.154 s / 0.657 s / 2.902 s for train/development/final.
These are reference-data collection rates, not PPO, model inference or JAX rates.

Raw episode storage is **6,812,514 compressed bytes** and **167,355,816 canonical
JSON bytes**: approximately **70,964 compressed bytes per episode**. The full
raw files are privileged diagnostic evidence and are never model input.

Python validation/projection/actor export took a separate **267.320 seconds**
with four workers, after **2.135 seconds** of input audit and copying. All 7,782
actor rows passed the strict actor-wire validator. Actor exports occupy
**13,314,669 compressed bytes** (301,560,555 JSON bytes); separate target files
occupy **68,082 compressed bytes**. The one-time legacy exposure audit took
319.773 seconds and is not charged to collection throughput.

| Per-episode latency, milliseconds | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|
| Generation | 395 | 859 | 1,024 | 1,024 |
| Serialized replay | 582 | 1,311 | 1,522 | 1,522 |

These use nearest-rank quantiles. Per-episode replay latency includes serialization
and validation; collection throughput excludes the separately measured team
reservation and Python stages.
Peak memory, GPU speed and learner update cost were not measured.

## Provenance and actor boundary

Node/Python agree on 100 assignment cases and exact threshold edges. The gates
check policy, source/build/vocabulary hashes, reservations, exposure exclusions,
all planned jobs and seed roles. Python independently derives the same pairs,
seat orders and seeds, then checks each raw episode against its exact job.
Missing or duplicate episodes, wrong partitions, final jobs, source drift,
changed bytes and paths escaping the dataset root reject.

Python validates all raw episode semantics through the R4 validator, builds
same-player trajectories through R5, and checks every actor wire before export.
Actor files and target files are physically separate. The actor interface returns
only immutable `ActorInput` objects; policy and value consume the same tensors.
Seeds, full team specifications, privileged states and future outcomes stay outside
that interface. Targets retain action/reward/next-player linkage for a later
learner; they are not actor features.

Admission requires a separately trusted dataset content hash. The completion
report and archive provide that anchor; computing a hash from an arbitrary
untrusted dataset is not authentication. Regression tests reseal bad metadata to
prove semantic checks still reject it, and inject privileged fields into actor
wires to prove schema enforcement is not merely a checksum check.

The complete regression suite passed **59 Node and 66 Python tests (125 total)**.
The R7 Python group includes 15 identity, leakage and integration checks. R5's
bound encoder/contract sources and the root dependency lockfile remain unchanged.
The completion sealer rechecks admission, all exported file hashes and row counts,
actor/target alignment, and the absence of final episodes and actor files before
archiving the result.

## Files and reproduction

- Policy and contract: `contracts/dataset-v1/`.
- Node collection: `packages/dataset/`.
- Python identity, exposure scan, gates and export: `python/pokemon_reference/dataset_*.py`.
- Legacy audit: `artifacts/runs/r7-legacy-v1/`.
- Accepted collection: `artifacts/runs/r7-collection-v1/`.
- Validated dataset and benchmark: `artifacts/runs/r7-dataset-v1/`.
- Completion report: `artifacts/reports/r7-sprint.json`.
- Independent review and logs: `artifacts/reviews/r7/`; [verified dispositions](R7_REVIEW.md).

Accepted dataset content hash:
`38d2048f1ec770ebedc0838dbfb75a347531031338e8d05e47009171a0415761`.
The completion receipt binds the current source tree, prior milestone reports,
legacy registry, accepted dataset and review/test evidence. Its portable snapshot
is under `artifacts/manifests/r7-<completion-content-hash>/`, with source and
evidence separated. Final reservation copies retain file mode `0600`.

Use fresh output directories; this R7 wrapper fails closed on partial runs and
does not overwrite or resume them. Earlier R3/R4 resumable collectors are unchanged.

```sh
PYTHONPATH=python .venv/bin/python -m pokemon_reference.dataset_legacy /tmp/r7-exposures
npm run dataset:collect -- /tmp/r7-collection /tmp/r7-exposures/registry.json
PYTHONPATH=python .venv/bin/python -m pokemon_reference.dataset_build \
  /tmp/r7-collection /tmp/r7-exposures/registry.json /tmp/r7-dataset --workers 4
npm test
```

Replaying the declared namespace recreates the teams and episode content; measured
wall times and the collection-plan timestamp are not reproducibility invariants.
Do not use a new namespace/salt to select final teams based on outcomes.

```python
from pokemon_reference.corpus import read_json
from pokemon_reference.dataset_gate import ActorDataset

receipt = read_json('artifacts/reports/r7-sprint.json')  # trusted retained receipt
rows = ActorDataset('artifacts/runs/r7-dataset-v1',
                    expected_hash=receipt['dataset_hash'], purpose='training')
actor = next(rows.actors('train'))
policy_inputs = actor.policy_inputs()
value_inputs = actor.value_inputs()
```

## Sprint result and next step

R1–R7 now provide the pinned oracle; semantic Node/Python contracts; 100,000
validated official teams; 1,000 replayed exploratory episodes; information-safe
player-history features; the conservative mechanics catalog; and the new split
and admission boundary. Existing R3/R4 timings remain historical measurements
with their documented overlapping workloads; they are not directly comparable
to this smaller four-worker campaign.

The R6 custom-engine gate remains blocked for all unsupported entries. This sprint
has not established accelerated-engine fidelity, exhaustive mechanics coverage,
learned playing strength or readiness for a large GPU spend.

**Next: M1/G1—CPU learner canaries, followed by a deliberately restricted simulator
slice.** Start with hand-checkable returns/GAE, policy-ratio and single-action
cases, bandit/delayed-reward learning, and checkpoint/resume. Define the first
restricted Pokemon profile explicitly and use the R6 fixture obligations for
its differential tests. Keep final data sealed and use development data for
choices. Reassess compute after a measured collect/update loop exists.
