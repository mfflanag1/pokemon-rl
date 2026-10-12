# R5 player-history features and handoff

R5 provides a pure Python player-history encoder, a versioned fixed tensor
interface, request-relative policy actions, and separate training targets. It uses
the pinned Node Showdown oracle and leaves R1–R4 producer/consumer sources intact.
Local CPU is sufficient; no training or hosted compute is involved.

The [feature contract](../contracts/player-features-v1/README.md) documents each
shape, knowledge flag and information boundary. The source audit and independently
checked research corrections are in [R5_SEMANTICS_AUDIT.md](R5_SEMANTICS_AUDIT.md).
The [implementation review](R5_REVIEW.md) records the confirmed absorb-healing
attribution bug, its live regression tests and the corrected interpretation.

## Recorded result: 2026-10-10

The [sealed machine-readable report](../artifacts/reports/r5-features.json) records
the completed run, with all source hashes and per-episode chain receipts. Source,
live fixtures, reviews, logs and receipts are retained in
`artifacts/manifests/r5-d0f4c82bbc85df7ed1cf54eafa2d3f825decfa58577179cb1ba64eb2a26b66b5/`.

| Check | Result |
|---|---:|
| R4 campaign episodes reconstructed | 1,000 |
| Decision inputs validated | 81,947 |
| Python legal-action index round trips | 778,653 |
| Adapter invariant pairs | 1,000 |
| Live hidden-state invariant pairs | 18 |
| Live visible-change positive controls | 2 |
| Live Transform / Illusion / absorb semantic probes | 14 |
| Shared Node/Python request cases / action round trips | 16 / 90 |
| Additional saved R4 fixtures | 13 |
| Fixture decision inputs / truncation bootstraps | 334 / 20 |
| Regression suite | 52 Node + 41 Python tests passed |
| Scan wall time, 8 local workers | 596.7 seconds |

All 28 frozen R1–R4 code/contract files checked against archived hashes were
unchanged; the reference manifest also matched exactly. No custom simulator,
learned model or GPU job was introduced.

Paired corpus boundaries included `{'faint_replacement': 459, 'move': 275, 'revival': 6, 'switch_continuation': 260}`, with
31 rejected-choice boundaries. Seats:
`{'p1': 365, 'p2': 635}`. Memory truncation affected
57,875 decision event windows and
28,811 opponent-appearance windows;
the associated full histories were retained. Byte-row overflows:
0 event rows and 0 fact rows.

## Use the encoder

Set `PYTHONPATH=python` and use the existing `.venv` from R2 setup.

```python
from pokemon_reference.player_history import PlayerHistory
from pokemon_reference.actor_actions import ActionCodec

history = PlayerHistory("p1")                 # one instance per player/game
actor = history.append(own_r4_view)            # strictly verified private delta
policy_tensors = actor.policy_inputs()
value_tensors = actor.value_inputs()           # identical information boundary
codec = ActionCodec(history.request)
if codec.control == "choose":
    legal_index = next(i for i, legal in enumerate(codec.mask) if legal)
    semantic_action = codec.decode(legal_index)
# "wait" emits no policy action; "pass" uses the existing semantic pass control.
```

`ActorInput` is immutable serialized content; returned dictionaries are copies.
Use `load_actor_input` for external feature JSON. Use `assemble_episode(bundle,
manifest)` for offline data; it fully validates the R4 artifact before splitting
actor inputs from action/reward/transition targets. The latter include rejected
attempts, same-player successor links and explicit truncation bootstraps. A future
learner must deliberately handle rejected attempts; R5 does not invent a learning
rule, behavior probability or value estimate for them.

A runnable offline example using the retained short golden episode:

```python
from pathlib import Path
from pokemon_reference.contract import parse_json
from pokemon_reference.feature_trajectories import assemble_episode

golden = parse_json(Path("tests/fixtures/r4-golden.json").read_bytes())
p1, p2 = assemble_episode(golden["bundle"], golden["manifest"])
print(p1.actor_inputs[0].feature_id, p1.targets[0])
assert p1.bootstrap is not None               # explicit truncated continuation
```

## Verification and reproduction

`npm test` runs the full regression suite, including fresh live R5 Node probes
consumed by Python. To regenerate retained observer evidence and scan the corpus:

```sh
mkdir -p artifacts/runs/r5-features-recheck
npm run features:fixtures -- artifacts/runs/r5-features-recheck/live-probes.json
PYTHONPATH=python .venv/bin/python -m pokemon_reference.feature_campaign \
  artifacts/runs/r4-reference-1000-v1 \
  --fixtures artifacts/runs/r4-fixtures-v1 \
  --live artifacts/runs/r5-features-recheck/live-probes.json \
  --out artifacts/runs/r5-features-recheck/validation --workers 8
```

Use a fresh output directory. The scan binds the R5 source/catalog/layout hashes,
R4 manifest/index/validation receipts, exact compressed and decompressed episode
bytes, live-probe file and per-episode feature chains. It reuses R4's prior full
semantic certificate only after those integrity checks; every R5 player view is
validated afresh. The 13 saved fixtures are fully revalidated through the offline
adapter. Report failure and source changes prevent a completed result.

Do not run another Node oracle job concurrently with `npm test`: existing R1 tests
intentionally alter build/source sentinels to verify fail-closed provenance.

The campaign records one paired adapter test per episode. These change privileged
metadata, opponent views/actions, RNG/commitment audits, outcomes and future trace
data while preserving a selected own-player prefix. They are deliberately invalid
privileged carriers, not legal counterfactual episodes. Separate live interventions
change hidden bench fields, RNG, opponent commitments and exact HP, with unchanged
visible history required. Positive controls cross a visible HP bucket. Both seats,
Transform directions and Illusion reveals are covered.

The policy/value probe is a deterministic synthetic tensor consumer. Byte-identical
inputs, masks and probe results pass the present boundary gate; they do not test
learned logits or a future search algorithm. Add those tests when such code exists.
Campaign wall time includes artifact integrity checks, parsing, validation, live
probes, fixture reconstruction and paired re-encoding. It is not model inference
speed or end-to-end RL throughput. Feature chains are retained as hashes rather
than materializing a large duplicated tensor dataset; tensors can be reconstructed
from the frozen sources and R4 episode bytes.

## Next: R6

Build the reachable-mechanics manifest and fixture catalog from the full R3
generator/set inventory and R4/R5 evidence. Classify indirect move calls, Transform,
form changes, ability/item interactions and dynamic dispatch; give each supported
entry a source reference, owning implementation ticket and fixture family. Resolve
unknown pathways before making a full-format accelerated-engine claim.

Retain the R5 input boundary when adding richer memory or a learner. The current
fixed representation keeps six recent encounters and 64 recent mechanical events;
it is a conservative, lossy encoding, not an identified opponent roster or complete
belief state. Older events remain in the verified history. Never fill unknown
opponent fields from the privileged team corpus.

All inspected R3/R4/R5 data remains exploratory and `unassigned`. R7 must introduce
independent partition/provenance gates before final evaluation. Prime Intellect
adds no necessary capability for this CPU stage; revisit GPU infrastructure after
engine correctness and measured collection/update throughput justify it.
