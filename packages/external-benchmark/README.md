# F34 offline Foul Play adapter

This additive adapter runs the pinned official Showdown engine locally and passes
one legitimate player protocol to an isolated Foul Play process. It does not
connect to a Showdown server. Foul Play remains an external evaluation opponent;
the adapter does not collect training labels or start a scientific panel.

## Runtime and provenance

- Foul Play: `8f541f80f91f960603673796559f796b10ccf0fb`, GPL-3.0. Its
  source in `artifacts/external/foul-play` is unmodified.
- `poke-engine==0.0.48`, MIT, built with `poke-engine/terastallization` and
  `--no-default-features`, using the already installed Rust 1.87 toolchain.
- Isolated runtime: `.venv-f34-external`, Python 3.13.7. The existing F32 Python
  3.14 environment is unchanged. Exact package versions, source hashes, source
  distribution hash, and compiled extension hash are in `pin.json`.
- Public Randbats set prior: exact local bytes and MIT license in
  `artifacts/reviews/f34-policy-reset/external-integration`. It is a public
  inference prior, not the opponent's actual team. The prior can differ from the
  frozen Showdown generator.

The adapter checks source/runtime/prior hashes on initialization. Its audit hook
rejects IPv4/IPv6 `socket.connect` calls, including localhost. No websocket client
is constructed. Search subprocesses import the same hook. Unix multiprocessing
pipes remain available.

Initial installation used:

```sh
uv venv --python /opt/homebrew/bin/python3.13 .venv-f34-external
RUSTUP_TOOLCHAIN=1.87 uv pip install --python .venv-f34-external/bin/python \
  requests==2.33.0 websockets==14.1 python-dateutil==2.8.0 poke-engine==0.0.48 \
  --config-settings 'build-args=--features poke-engine/terastallization --no-default-features'
```

For reconstruction, use the retained source distribution and exact dependency
versions in `pin.json`; changing compiled runtime bytes requires an explicit new
pin rather than silently accepting a rebuild.

## Coordinator API

Both processes use JSON lines over standard input/output. Replies are
`{"id":...,"ok":true,"data":...}` or an explicit error. The Python
`pokemon_external_benchmark.transport.JsonWorker` provides bounded write/read
requests and process-group cleanup; it creates no network service.

Start the simulator with `node packages/external-benchmark/worker.mjs`:

```json
{"id":1,"op":"start","run_seed":1027341901,"episode_index":0,"swap":false,"external_side":"p2","tactical_side":null}
```

The simulator extends `StructuredSession`, retaining its player-only
`observable_history[side]` sidecar. Both frozen F32 and candidate evaluations use
this same worker and mechanics. The simulator uses the existing F2 deterministic team and mechanics admission,
explicitly restricted to `development`. It returns:

```text
{ frame: <existing player projections, observable_history by seat, outcome and coordinator provenance>,
  external: { schema_version: "f34-external-player-v1", side: "p2",
              request_id: 1, pending: true, protocol: [<own-channel delta>] } }
```

`step` accepts exactly the pending sides' semantic action indices and request IDs,
just like the existing session. `cutoff` invokes the existing administrative
cutoff at a fresh joint move boundary. `close` destroys the session. `hello`
returns the pinned official reference commit.

Start Foul Play with:

```sh
PYTHONPATH=python .venv-f34-external/bin/python -m pokemon_external_benchmark.foul_play
```

```json
{"id":1,"op":"start","side":"p2","search_time_ms":100,"seed":123}
```

For every simulator bundle, send `{"id":2,"op":"choose","input":<external>}`.
The whole frame, other player projection, team-generation seed, team hashes,
future action, and authoritative engine state must never enter this call. The
worker rejects extra envelope fields, foreign private requests, unsplit protocol,
multiline injection, and replayed request coordinates. It ingests waiting inputs
and returns `{"waiting":true}`; otherwise it returns a semantic action index,
request ID, original Foul Play decision, timing, and nominal search allocation.
The coordinator must validate the resulting action against the current player's
mask before submitting either simultaneous action. Invalid choices are errors;
there is no fallback policy or silent replacement.

Raw simulator requests lack the server's `rqid`. The adapter adds only its local
per-player request counter, matching the existing projection's request ID. Initial
state setup follows Foul Play's `RandomBattleMode.start_battle`: initialize its own
six Pokémon from its private request and parse public opponent switch/events.
Thereafter it uses upstream `update_battle`, `find_best_move`, and
`format_decision`. Move names are translated through the current private request.

## Search budget and reproducibility

The engineering default is `search_time_ms=100`, parallelism 1, threads 1. Upstream
Randbats logic samples four worlds at 50 ms before opponent moves are revealed,
or two worlds at 100 ms later: 200 ms nominal engine budget per decision. This is
not a hard deadline for parsing, spawning workers, or choosing. The response
records sampled-world count, per-world milliseconds, total nominal milliseconds,
and measured search wall time. The coordinator measures full request latency.

Python's hidden-set and final-action sampling is seeded per game. The native Rust
MCTS interface exposes no RNG seed and uses wall-clock stopping, so identical
runs need not choose identical actions. Recorded choices can be replayed exactly
through official Showdown; native search reproducibility is not claimed.

One upstream warning was observed and retained: the public prior encodes Ditto as
`87,choicescarf,imposter,transform,steel`; Foul Play's fixed four-move slice tries to
interpret `steel` as a move and ignores it after a warning. This is a pinned
opponent inference limitation, not an adapter repair. Engine mechanics are also
an approximation; official Showdown remains authoritative.

## Engineering checks

```sh
node --test tests/f34_external_protocol.test.mjs
.venv-f34-external/bin/python tests/f34_external_adapter_test.py
PYTHONPATH=python .venv-f34-external/bin/python -m pokemon_external_benchmark.engineering \
  --output artifacts/reviews/f34-policy-reset/external-integration/new-smoke
```

The smoke driver refuses an existing output directory, records actions, player
protocol, provenance, timing, outcomes and failures, and uses the local tactical
opponent by default. Its optional learner callback receives only that seat's
player projection and a coordinator decision index. It is explicitly an
engineering driver by default; a scientific schedule and budgets must be frozen
separately. The coordinator can pass an explicit `purpose` label. Set
`learner_uses_history=True` to invoke `learner(player,index,own_history)`; otherwise
the callback remains `learner(player,index)`. The driver records illegal choices
and rejected submissions separately by seat, raises on illegal actions, and
retains failure evidence. It never guesses callback signatures or substitutes
action choices.

Engineering exposure includes development run seed `1027341901`, episode indices
0 and 1. Exclude these exact teams from fresh scientific claims. Initial tests
verify player isolation, source pinning, legal conversion, all three frozen F32
checkpoint inputs, and exact official-engine replay from recorded decisions.
