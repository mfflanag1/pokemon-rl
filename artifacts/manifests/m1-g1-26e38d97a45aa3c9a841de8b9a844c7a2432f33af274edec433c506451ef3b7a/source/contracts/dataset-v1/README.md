# Dataset contract R7 v1

This additive layer leaves R1–R6 artifacts and raw R4 `split: unassigned`
envelopes unchanged. R7 admission records assign splits only after a frozen
precollection plan and independent checks. The reference engine remains pinned
Showdown; this contract does not grant custom-engine support.

## Identity and assignment

`policy.json` freezes an ordered whole-team holdout. Identity retains resolved
species/form, ability, item, ordered moves, Tera type, level, gender, EVs and IVs.
The R2 official-generator schema has no explicit nature/happiness fields;
unknown additional fields reject rather than disappear. Nicknames, shiny flags,
role/source-species bookkeeping, generation seeds and raw team hashes are
excluded. Meaningful forms and cosmetic resolved form IDs remain distinct; this
is not a holdout of individual sets, base species, unordered teams or every
mechanically equivalent configuration.

Team key = canonical SHA-256 of `[identity_version, canonical_team]`.
Partition digest = canonical SHA-256 of `[hash_version, salt, canonical_team]`.
Interpret all 256 bits as an unsigned integer x; compare x*10000 against
9000*2^256 and 9500*2^256, with half-open train/development/final ranges. No floating
point, reduced-width truncation or modulo bucketing is used. Node and Python
implement and check this independently.

## Legacy exposures and holdout

R3's 100,000 teams and all R4 episode teams/seeds are exposed exploratory data.
Their assignments are audited prospectively; they are not retroactively clean
holdout data. A content-bound exposure registry excludes them from fresh R7
collection. Final candidate generation is automated and filtering uses only the
fixed policy and prior exposure/duplicate checks. Final teams are retained in a
separate file, with hashes and counts in the public collection index; no final
battles, actor export or outcome-driven selection is performed.

This seal is a workflow/API boundary and integrity commitment, not encryption or
an OS security boundary against the file owner. Reading final team files or
running their battles later requires an explicit evaluation-release procedure.
Current ActorDataset has no final/evaluation access mode. Do not change the salt
or reroll the reservation after looking at final results.

## Collection and pairing

Write `plan.json` before candidate generation. Preserve attempted/accepted and
rejection counts per split. Reserve 64 train, 32 development and 128 final teams
for this initial campaign. Pair consecutive teams within train/development and
play both seat orders: 48 pairs, 96 episodes. Final has no jobs. The two mirrors
share a mechanics seed; policy seeds follow the logical teams when swapped.
Team generation, mechanics and policy seed namespaces are separate. Mirrors are
correlated observations and remain in the same split/group.

A ledger rejects shared team identities, seeds, episode keys or mirror groups
that cross split/group boundaries; seeds cannot cross roles. Mirrored reuse is
explicit and allowed within its group. Jobs are independently re-derived in
Python from the reservations and namespace. A raw episode must match its exact
job/spec and official profile; renamed/relabelled records cannot silently pass.

Generation uses long-lived local Node workers. Each complete episode is serialized
and replayed before its record is committed. Errors retain a failure record and
partial output; no complete index is published. This initial R7 wrapper is a
fresh-directory run, not a resumable collector. The earlier R3/R4 resumable tools
remain available. Never conceal partial results by reusing the same directory.

## Admission and actor separation

Python verifies collection file hashes, policy, reference/source/build provenance,
reservations, exposure exclusion, jobs, mirrors, raw episode contracts and all
record summaries. Export creates physically separate actor and target gzip JSONL
streams. Each actor wire passes R5's strict schema, tensor bounds and feature hash;
full teams, RNG, future outcomes and privileged raw records never become actor
columns. Both policy and value read the same ActorInput tensor interface.

`dataset.json` binds source hashes, collection files, actor/target checksums and
validation counts. `ActorDataset(directory, expected_hash=..., purpose=...)`
requires a separately trusted dataset hash; a self-supplied checksum is not an
authenticity signature. Training can load train actors, development can load
development actors, and final access rejects. The loader returns only ActorInput
objects and rejects privileged/wrong-kind wires and changed bytes.

## Artifact versions and benchmark meaning

- `r7-exposures-1`: prior team keys, seed set and episode keys, linked certificates.
- `r7-collection-plan-1`: fixed policy, counts, namespace, producer and reference.
- `r7-team-reservation-1`: per-split official R2 teams plus their reference entries.
- `r7-job-plan-1`: train/development pairs, seat orders and exact reference specs.
- `r7-collection-1`: complete records, hashes, sampling counts and measured times.
- `r7-dataset-1`: independently validated dataset and separated actor/target files.
- `r7-benchmark-1`: measured wall times, counts, storage and hardware.

Seals use the existing canonical UTF-8 sorted-key integer JSON convention and
hash all fields except `content_hash`. File hashes include exact bytes/newlines.
Source and data paths are confined to their declared roots.

Collection throughput includes worker startup, serialized reference replay,
compression and storage. Per-episode replay latency includes serialization and
validation. Actor validation/projection/export is measured separately. Millisecond
counts and integer rates scaled by 1,000 avoid nonportable numeric encodings.
Latency quantiles use nearest rank. Tera remains disabled in the frozen random
baseline. These measurements are reference-data costs, not PPO throughput,
accelerated-engine fidelity, general mechanic coverage or learned playing strength.
