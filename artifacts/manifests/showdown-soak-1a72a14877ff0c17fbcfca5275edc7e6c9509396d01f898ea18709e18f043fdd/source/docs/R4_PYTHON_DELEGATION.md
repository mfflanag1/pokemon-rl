# R4 independent Python delegation verification

Immutable acceptance evidence, 2026-10-10. The bounded implementation run was
`20261010-134705-46b154c6` in the Pokemon cross-model ledger. Fable 5.1 owned only
`python/pokemon_reference/episodes.py`; Codex QA accepted the frozen candidate.

The coordinator independently reran the exact eight fixed acceptance tests in the
isolated candidate workspace with the project's Python environment: eight passed.
The tests accepted a Node-produced golden bundle without mutation and rejected
rehashed transition, reward, phase, private-view, stale-request, unsafe-cut and raw
evidence corruption. Inspected the candidate patch: only the owned module changed.
`git apply --check` succeeded. Its reconstructed `to_protocol` was checked against
`packages/battle-contract/actions.mjs`: zero-based slot + 1, move/switch/pass and
optional ` terastallize` encoding, with visible admissibility enforced. Offline
schema resolution and no Node subprocess were confirmed from source.

Before integration, selected input hashes were compared with `request.json`.
Only the R4 schema and Node validator had changed under coordinator ownership;
the owned Python stub, fixed tests, golden fixture, R2 dependencies and source
excerpt still matched. Those two intentional changes finalized controller/order
identity, content-bound decision IDs and safe truncation after re-requests.

Accepted the frozen candidate for its stated snapshot. After applying its patch,
the coordinator adapted those new fields/invariants and replaced repeated prefix
parsing with a one-pass history index. The refreshed golden and eight tests passed
again in the main project. This original QA does not cover those subsequent edits;
they are included in the focused R4 follow-up review and campaign validation.
Rework estimate: 8 minutes, including integration and the updated semantics.
