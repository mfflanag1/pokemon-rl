**Provisional verdict: no blocking findings in reviewed scope; two medium/low gaps worth closing before the first full run.**

## Findings

**1. Medium — Producer-drift check during collection covers only 4 files (`packages/dataset/collect.mjs:14-16, 113`).**
`SOURCE_FILES` omits the code that actually produces episodes (`episodes/session.mjs`, `reference-node/engine.mjs`, worker-loaded modules). The reference manifest is captured once at `collect.mjs:62` and passed to workers; Python later re-hashes manifest sources against the *current* tree (`dataset_gate.py:85-91`), not the tree at generation time. A mid-run edit to `session.mjs` that is reverted before export would pass every gate. Not a leakage vector, but it weakens the "jobs are independently re-derived / exact official profile" provenance claim. Verification: include manifest `source_hashes` (or re-call `episodeManifest()` and compare `manifest_id`) in the end-of-run check at line 113.

**2. Low — Node identity's "unknown fields reject rather than disappear" is unverified (`identity.mjs:7-12`).**
Python enforces exact field sets (`dataset_identity.py:24-27, 57-66`); Node relies solely on `validateSchema(payload,'team')`. If the R2 schema lacks `additionalProperties:false` at every level (`pokemon[i]`, `evs`, `ids[i]`), Node silently drops e.g. `nature` and computes a key Python would refuse. Parity test (`dataset.test.mjs:9`) only varies `level`. Since Python is the admission authority, a drift would surface as a Python rejection at gate time rather than leakage — but the collection would be wasted. Verification: add a `nature` fixture to the Node/Python parity test asserting both reject.

**3. Low — Node collector never checks mechanics/policy seeds against the exposure registry (`collect.mjs:29-36`).**
Only team seeds are filtered (`:79`). Python gate does reject (`dataset_gate.py:125`), so this cannot admit a leak, but a collision (possible only if R4 used the same `pokemon-reference-r1` label namespace) would fail after a full run. Cheap to pre-check in `buildJobs`.

**4. Low — Node `partitionDigest` checks only upper bounds (`identity.mjs:20`).**
Correct given the pinned policy, but it implicitly assumes ordered, contiguous ranges from 0; `requirePolicy` only protects callers that invoke it. Python validates lower bounds explicitly. Threshold edge tests on both sides pass per claim, so this is hygiene.

**Verified as correct in snapshot:** half-open exact-integer thresholds (Node/Python agree; `dev = ceil(0.9·2²⁵⁶)` test values are right); mirror seed handling (shared mechanics seed, policy seeds reversed in both `buildJobs` and `expected_jobs`); ledger rejects cross-split/group/role reuse while permitting in-group mirror reuse; rejection accounting (`attempts == accepted + Σrejected`) matches the loop at `collect.mjs:76-85`; record↔job binding via exact `spec` canonical equality, `submission_order`, episode hash and byte digests (`dataset_build.py:33-46`); no final jobs/exports (`worker.mjs:10`, `dataset_gate.py:122,145,154`); nearest-rank quantiles are correctly `ceil(np/100)`; `collection_wall_ms` does start before `fork`, so throughput includes worker startup as advertised; `paired_matchups = episodes/2` is consistent with 96/48.

## Scope limitations
Not in packet: R2 `schema.json`, `seedFor`/`reference_seed` equivalence, `load_actor_input`/R5 wire schema, `validate_episode`, `episode_key` derivation, catalog resolver shapes, `verify_seal`. Independence of mirrors/pairs is acknowledged in the README, not tested here. No tests executed; full collection/export not yet run per task statement.

## Verdict
Findings (no high-severity). Admission logic as written does not admit candidate/record mutation, reused seeds, cross-split mirrors, dropped records, or final data; the main gap is producer-code drift during collection (Finding 1).
