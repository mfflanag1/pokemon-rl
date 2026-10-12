# R1 review and verification

2026-10-10. This records checks actually performed on the local reference harness.

## Results

- `npm run setup` succeeded twice from the upstream lockfile on Node v25.6.1/npm 11.9.0, macOS arm64. Both builds produced compiled-tree SHA-256 `e65fb25c0aeed57ea58d81dc69a2185476975fb0984cc2d857e92277514724db`.
- `npm test`: **19 tests passed**, including saved reference fingerprints, exact replay, upstream player-stream equivalence, private-state isolation, live divergence detection, source/build integrity failures and natural hidden-trapping re-requests.
- `npm run smoke`: **10 completed original battles and 10 exact replays**, 823 original decision records over 359 original turns. Two unavailable switch attempts were retained and replayed. [Machine-readable report](../artifacts/reports/r1-smoke.json)
- The upstream checkout is clean. Dependency install and build are the only network/setup phase; the test/smoke commands ran locally without service calls. No GPU, server deployment or model training was started.

## Independent source audit

Cross-model run: `20261010-112553-890ccff3`, project `Pokemon-83aa7cb62b4e`. Fable inspected the selected pinned `battle-stream.ts`, `side.ts`, `prng.ts` and `teams.ts`; Codex supplied opposite-model QA. The draft is **rejected as a complete contract** because several prescriptions were too broad. Useful findings were independently checked and retained. This quality verdict is unrelated to tool access or approval.

Verified corrections and integration:

1. `Side.emitChoiceError` chooses `Unavailable` when a request update callback reports a change. Inspection of `chooseMove`/`updateDisabledRequest` confirmed that change can include other request metadata; the prefix alone does not prove a newly hidden restriction. The harness now checks request-visible admissibility before submission and records the subsequent reference feedback. A natural trapped-switch fixture checks actual before/after request hashes and same-turn retry behavior.
2. Upstream seed dispatch is permissive. Our wrapper accepts only canonical 64-digit sodium strings, with negative malformed/legacy-array tests. Canonical strings may round-trip literally; the draft's blanket prohibition on literal comparison was rejected.
3. The draft's broad `toID` team-comparison advice was not adopted. The runner preserves structured official generator output and hashes its complete JSON fields. It clones before use, verifies explicit teams in the simulator and tests seat swaps. No packed-string equivalence assumption is needed.
4. Source inspection confirmed owner-only side updates, error-before-updated-request ordering, and loss of request data from omniscient-only logs. The implementation uses explicit per-player projection and verifies a whole battle against `getPlayerStreams`.
5. The sources do not define an external policy RNG contract. The harness owns separate policy seeds; inspection of pinned `RandomPlayerAI` verifies the actual seed/options path.
6. Inputs are deliberately framed: one initial setup chunk and one logical choice per subsequent write. No attempt is made to roll back a partially failed upstream write; the episode aborts. Control commands are not accepted as policy choices or executed from replay artifacts.

## Implementation snapshot review

Fable review: `20261010-113358-bdba8942`. It reported nonblocking test/contract gaps. The coordinator verified the following resolutions:

- Explicitly set the baseline transformation option to zero and assert no Tera selections in the tests and every smoke episode. Pinned upstream constructor/defaults were inspected directly.
- Added negative replay cases with recomputed envelope hashes, so tests reach the live decision-order, observation and final-result mismatch checks rather than merely failing checksum validation.
- Recheck upstream source cleanliness in `describe()`; detect harness-source changes in a long-lived worker and require restart. Build and PRNG dependency fingerprints are checked on process startup. The receipt does not claim a complete hash attestation of all `node_modules` files.
- Reject an unexpected error for a different player during a submission rather than silently attributing it to the acting player.
- Add deliberate temporary source/build integrity sentinels in tests, with cleanup, to exercise rejection of dirty source and altered compiled output.
- Limit the information-invariance claim to the actual initial hidden-item perturbation test. Full-battle stream-routing equivalence is separately verified; comprehensive hidden-information testing remains future work.

The request-visible choice helper and regression additions were introduced after the original snapshot; their relevant upstream semantics were checked directly and the final 19-test suite plus ten-battle replay campaign passed. The review is not represented as a final code execution or exhaustive correctness proof.

## Remaining limitations

The current contract is internal Node R1 only; no Python/JSON Schema compatibility is claimed. Automatic baseline battles exclude Tera. Ten battle seeds and one initial private-item variation do not establish complete mechanics/information coverage. No second simulator exists yet, so replay equivalence is not differential engine fidelity. The saved corpus is privileged reference data; only player projections may enter a future actor.
