# R4 request-aware episodes

R4 adds a versioned phase/transition layer around the pinned R1 Showdown oracle.
R1–R3 artifacts and interfaces remain usable. It records every submitted semantic
action, simulator feedback, private next request and same-player successor. The
campaign result and evidence locations are in [R4_CAMPAIGN.md](R4_CAMPAIGN.md).

## Run and consume

Use the Node/Python environments from [R2_CONTRACT.md](R2_CONTRACT.md). From the
project root:

```sh
npm run episodes -- run example-r4 --out artifacts/runs/example-r4.json
npm run episodes -- replay artifacts/runs/example-r4.json
PYTHONPATH=python .venv/bin/python -m pokemon_reference.episodes \
  artifacts/runs/example-r4.json --manifest artifacts/runs/example-r4.json.manifest.json

npm run episodes -- campaign --out artifacts/runs/my-r4-campaign \
  --namespace my-r4-campaign-v1 --count 1000 --workers 8
PYTHONPATH=python .venv/bin/python -m pokemon_reference.episode_campaign \
  artifacts/runs/my-r4-campaign --out artifacts/runs/my-r4-campaign/validation --workers 8

node scripts/episode-fixtures.mjs artifacts/runs/my-r4-fixtures
npm test
```

Single-run and fixture outputs must be fresh. Campaign generation is resumable:
repeat the identical output/namespace/count configuration. Worker count may change.
Independent Python validation uses a fresh output directory. Do not run integrity
sentinel tests or edit pinned sources while an oracle campaign is running.

`EpisodeSession` is the programmatic interface for external policies. Call `start`,
get the acting side's `view(side)`, then `submit` with exactly `submission_id`,
`side`, `request_id`, `view_id`, and a semantic `action`. Submissions serialize.
An identical retry returns the original receipt; reusing an ID for different
content fails. Stale, inactive and visibly forbidden actions fail before an engine
write. A failed operation after a write poisons the session. `artifact()` requires
termination or an explicit safe `truncate()`; `close()` releases the oracle.

The built-in baseline is the pinned upstream RandomPlayerAI, move probability 0.9,
with Tera disabled. Its controller is distinguished from externally scripted
choices. Team generation, mechanics and each player's sampling have independent
seed namespaces. The campaign alternates which seat submits first. Hidden actions
are never fed to either baseline policy.

## Request groups and transitions

| Phase | Meaning |
|---|---|
| `joint_move` | Both players have ordinary move/switch choices. |
| `switch_continuation` | One or both active Pokémon remain alive and a switch is requested, including pivots. |
| `single_replacement` | One side must replace a fainted active Pokémon. |
| `double_replacement` | Both sides must replace fainted active Pokémon. |
| `revival` | The request marks a revival target selection. |
| `mixed_switch` | Requested sides have different non-move switch reasons. |
| `terminal` | The simulator ended the battle; pending requests are canceled. |

These group labels are privileged diagnostics. The actor sees only its own R2
`move`/`switch`/`wait` request and visible history. A frame stores per-side request
counters, requested/pending sides, commitment references, and a selected mechanics
audit. The first ordinary submission stays in the same group; resolution opens the
next group. A private unavailable-choice re-request stays in the same group and
retains the other player's commitment. Wait requests also increment the receiving
side's request counter, so request count is not turn or group count.

Every attempted action gets one transition to the **next decision by that same
player**, even across an opponent action, forced switch, or rejected attempt.
Intermediate reward is zero. The final same-player transition receives the
terminal result: win 1, draw 0.5, loss 0. A real terminal never bootstraps.

A decision budget is a soft cutoff: finish the current group, any re-requests and
replacement chain until a fresh, uncommitted joint boundary (or termination).
Only `initial`/`resolved` frames permit a nonterminal cut. Each player's final
transition then references its saved bootstrap view with zero reward. The hard
submission cap raises an incomplete-episode error; it never fabricates a draw.
This supplies linkage for future gamma=1 return/GAE code, not a PPO rollout buffer:
behavior log probabilities, model versions and value estimates remain future work.

## Durability and verification

Persistent Node workers generate a complete episode, serialize it, validate it,
and replay the serialized choices from initial teams. They commit a compressed
file only after exact normalized episode identity matches. A checksummed progress
index tracks exact indices, allowing out-of-order worker completion without holes
in the final index. Resume verifies configuration, manifest and all committed file
checksums. Failure events persist; uncommitted destination files move to `orphans`.
A local PID lock prevents two writers. Dead local owners can be recovered; remote
owners, live/reused PIDs and ambiguous recovery markers require inspection.

Python independently checks every file's compressed/plain hashes and canonical
bytes; schema, channel projection, views and masks; phase/commitment transitions;
action/input tape; outcome/reward/linkage; deterministic seed indices and alternating
submission schedule; and summaries recomputed from the evidence. Its report binds
consumer source hashes and dependency versions. Its replay counts refer to verified
Node receipts, not a second independent battle implementation.

## Boundaries for the next owner

- Bundles and frames are privileged. Consume only per-player views through a new,
  tested R5 feature path; do not pass the whole episode to an actor. R2's typed
  player-view loader rejects phase episodes.
- The audit is selected fields, not a full simulator state or arbitrary checkpoint.
  Replay from initialization is the supported reproducibility contract.
- Constructed probes explicitly manipulate the pinned simulator and add a marked
  protocol message so requests are emitted. They test phase bookkeeping, not natural
  reachability or exact mid-turn timing. The campaign uses no such interventions.
- All data is exploratory and `unassigned`; R7 must split before final pairing and
  seal independent evaluation data. R4 cannot be used as a final held-out benchmark.
- Random baseline replay demonstrates reference plumbing, not policy strength,
  accelerated-engine fidelity, complete mechanics coverage or training throughput.
  Tera is absent from this baseline campaign. R6 owns systematic mechanics coverage.

## R5 handoff

Implement the player-history/feature boundary next, in this order:

1. Add a typed loader for an individual episode view. Reconstruct that player's
   history by appending only its deltas, verify prefix hashes, and keep actor
   outputs separate from frames, full teams, outcomes, input tapes and other views.
2. Define known/unknown flags for opponent species/forms, moves, items, abilities,
   status and displayed HP. Preserve what was observed under Illusion/Transform;
   do not repair apparent identities using the diagnostic truth. Opponent HP must
   retain the precision actually revealed to that player.
3. Map the existing 14 action slots through the current request order. Test normal
   moves, Tera, ordinary switches, faint replacements, pivots, revival, wait and
   hidden-legality re-requests. A new request updates visibility/masking; a transport
   retry does not become a training step.
4. Create paired hidden-state perturbation cases: unchanged private/public history
   must produce identical actor features, masks and policy/value inputs despite
   changed opponent truth, RNG, pending choices, diagnostic group labels or future
   reveals. Keep privileged training labels in a separate loader/type.
5. Validate trajectory assembly using R4's same-player links, terminal rewards and
   bootstrap views. Add frozen-policy log probabilities/values only when a real
   learner is introduced; never invent them for these random reference records.

Reuse the natural double-replacement, revival and hidden-rejection episodes named
in the campaign report. Treat any fixture selected after inspection as development
evidence. Leave the reachable-mechanics inventory to R6 and sealed partitioning to
R7. These steps remain local CPU work; they do not require rented GPU infrastructure.
