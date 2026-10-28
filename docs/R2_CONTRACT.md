# R2: versioned Node/Python contract and semantic actions

Implemented for the pinned Gen9 Random Battle reference. The normative
[contract](../contracts/battle-v1/README.md), [Draft 2020-12 schema](../contracts/battle-v1/schema.json),
[action table](../contracts/battle-v1/actions.json), and
[Dex vocabulary](../contracts/battle-v1/vocabulary.json) are versioned project data.
The [review record](R2_REVIEW.md) records verified findings and scope limits.

## Setup and verification

Use the Node version in `.node-version` and the R1 Showdown build from
[R1_REFERENCE.md](R1_REFERENCE.md). Install contract dependencies locally:

```sh
npm ci --ignore-scripts --no-audit --no-fund
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-contract.txt
npm test
npm run contract:smoke
```

The recorded run uses Node v25.6.1 and Python 3.14.3. Node dependencies have an
exact npm lockfile; Python's direct and transitive versions are pinned in
`requirements-contract.txt`. No GPU or hosted training service is needed.
Set `CONTRACT_PYTHON` to use another Python environment with those dependencies.

`npm test` runs the R1 tests, R2 Node tests with independent Python comparisons,
and the Python unittest suite. Tests run serially because the R1 integrity test
temporarily creates a sentinel inside the upstream source/build tree. Do not run
other oracle jobs concurrently with that integrity test.

`npm run contract:smoke` runs ten seeded battles, exports and independently
validates every artifact in Python, then checks exact equality of the complete
canonical JSONL bytes. The compact report is
[`artifacts/reports/r2-contract.json`](../artifacts/reports/r2-contract.json).
Large evidence streams live under the reported ignored `artifacts/runs/` directory.
This is a contract verification workload, not an RL throughput benchmark.

## Export and consume

```sh
npm run contract -- describe
npm run contract -- export r2-demo artifacts/runs/r2-demo
PYTHONPATH=python .venv/bin/python -m pokemon_reference artifacts/runs/r2-demo/artifacts.jsonl > /tmp/r2-validated.jsonl
```

Choose a fresh output directory for each export. Export writes the R1 reference
evidence plus versioned manifest, two teams, pre-submission player views,
decisions, and an episode index. It never executes an imported input tape.
The existing R1 JSONL worker remains an explicitly separate debugging interface;
its `r1-*` records are not portable v1 contract artifacts.

Node API boundaries:

* `packages/battle-contract/canonical.mjs`: strict parser, canonical serializer,
  SHA-256; no simulator dependency.
* `schema.mjs`, `artifacts.mjs`: structural and semantic validation, identity checks,
  narrow actor loader, known/unknown vocabulary conversion.
* `actions.mjs`: request-only admissibility/masks, semantic↔index mapping, and
  strict semantic↔Showdown command conversion.
* `packages/reference-node/contract-v1.mjs`: pinned Dex vocabulary, provenance,
  team/view/episode adapters and live `submitAction`.

Python uses `PYTHONPATH=python` and imports `pokemon_reference`. It independently
validates the same schema and implements canonicalization, masks, vocabulary
lookups, action indices, and artifact identity checks. It does not import Node or
call the simulator. `load_player_view(text, manifest_id=expected_id)` rejects
team/decision/episode artifacts, mismatched manifests, wrong-owner requests,
inconsistent history boundaries, and masks that disagree with the request.

```python
from pokemon_reference import load_player_view, action_from_index

view = load_player_view(player_view_json, manifest_id=expected_manifest_id)
indices = [i for i, allowed in enumerate(view["action_mask"]) if allowed]
# An empty mask requires explicit wait/pass control handling, not sampling.
if indices:
    semantic_action = action_from_index(indices[0])
```

For a live Node session, call `playerViewArtifact` and retain its `artifact_id`.
`submitAction(session, {schema_version:'1.0.0', side, request_id, view_id, action},
manifestId)` checks the exact view and request before encoding the command.
The underlying R1 session also rejects already-submitted or inactive requests.
An accepted submission is not necessarily a resolved turn: the opponent may still
need to act. An unavailable choice is recorded as a failed attempt followed by a
new request; do not erase it or train it as an ordinary completed transition.

## Verified scope and remaining work

The recorded R2 smoke run covers **10 episodes, 823 decisions, 359 turns, two
natural hidden-trapping rejections, and 1,677 artifacts**. Both languages produce
identical canonical JSONL. Tests separately exercise live Tera, stale submissions,
switch-order changes, and constructed Struggle, recharge, revival, wait and pass
requests. Constructed fixtures are labeled; they are not natural coverage claims.

The shared golden file contains selected examples, not a self-contained episode
dataset: its episode index can reference decisions omitted from the sample.
`npm run contract:fixtures` explicitly refreshes those examples from the pinned
reference. Tests never regenerate expectations automatically.

R2 preserves each player's full visible protocol prefix. This is intentionally
verbose and sufficient for exact decision-boundary inspection. R4/R5 will add
transition linkage, protocol deltas and model features. R2's phase labels distinguish
move/switch/wait, but not pivot versus faint replacement or joint decision groups.
Dataset splitting remains `unassigned`; do not treat this smoke corpus as train,
development, or held-out evaluation data. R7 owns partition enforcement.

An all-disabled advertised move list fails explicitly instead of yielding an empty
normal mask or guessing a fallback. The selected natural episodes do not encounter
that shape. The review record explains why its claimed post-rejection reachability
was not established. Supporting more formats or newly observed request shapes
requires a source audit and versioned fixtures.

**R3 is complete:** see [R3_CORPUS.md](R3_CORPUS.md) for whole-team generation,
validation and inventory. Its pilot discovered missing cosmetic species IDs in the
original vocabulary. The current `showdown-dex-ids-v2` table adds 68 IDs; the original
table remains `vocabulary.v1.json`. Bind numeric tokens to the exact version/hash
and remap by string ID when migrating. Schema 1.0.0 team fields remain unchanged.
The golden fixtures and R2 smoke report were refreshed against this revision.

Next is **R4: request-aware episode phases and transition linkage**. Do not start
policy training before the later simulator, observation and split gates.
