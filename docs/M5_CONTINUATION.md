# M5 — bounded simulator continuation

The versioned continuation simulator and fixture campaign passed on 2026-10-12.
Live games now cross fresh turn 13/25/37/49 request boundaries without resetting
their state or history. Scalar, batched JAX and pinned Showdown agree within the
new 60-turn slice. **Learner continuation is not implemented and no training ran.**

This addresses the simulator/state-carry prerequisite from
[M4](M4_BOUNDED_VALUE.md). It does not demonstrate improved play, remove the final
administrative cutoff or establish full-game simulator coverage.

## Measured production result

Run: `artifacts/runs/m5-continuation-v1`; launcher exit code 0. The fixed catalog
contains all 24 own teams × six scripted styles with a fixed opponent permutation
and seed schedule. These are mechanics fixtures, not learned-policy opponents or
a strength screen. Every fixture was independently replayed on unpatched Showdown.

| Check | Result |
|---|---:|
| Reference games | 144 |
| Terminal outcomes | 120 |
| Terminal outcomes after turn 12 | 90 |
| Administrative truncations at fresh turn 61 | 24 |
| Compared simulator frames | 4,038 |
| Compared player views, both seats | 8,076 |
| Decisions / executed events | 7,564 / 7,373 |
| Observed chance draws checked | 6,368 |
| Scalar/JAX/reference mismatches | 0 |
| Serialized live/terminal boundaries checked | 363 |
| Exact original-next-simulator-step replays | 219 |
| Exact frozen-boundary no-ops | 363 |
| Forced replacement frames | 224 |
| Replacements after turn 12 | 78 |
| Replacements on an intermediate window's last turn | 8 (turn 12) |
| Frames with history longer than 32 events | 1,707 |
| Maximum observed history | 120 events |

Fresh boundary coverage was 114 at turn 13,48 at 25,33 at 37,24 at 49 and24 at 61.
Post-turn 12 traces contain 1,052 executed attacks. The24 always-switch fixtures
retain their live game at the administrative ceiling; they are not scored as
draws, terminal rewards or reset episodes.

The source-bound regression passed **510 tests**: 240 Node, 236 existing Python,
21 M4 Python and 13 continuation Python. All **379 source/configuration files**
remained unchanged during final testing and production. The thirteen continuation
tests include actual Showdown replay, scalar/JAX comparison, whole-history views,
batch permutation, mixed terminal/live lanes, frozen-window resumption and
rehashed carry/tape corruption rejection. Hidden opponent PP/speed changes leave
the observer's projected view unchanged; own-view positive controls change.

An additional independent check verified 248 game/group addresses and 1,240 semantic
words across groups 0–61. This supports the preserved event-addressed chance law;
it does not assert identity with Showdown's raw PRNG stream or draw schedule.

## Implementation and boundary semantics

The new code is additive in `python/pokemon_continuation/` and
`packages/continuation/`, governed by
[the contract and fixture catalog](../contracts/continuation-v1/README.md).

The state holds 128 history rows. Sixty joint turns and at most one forced
replacement per side need at most 62 groups and122 executed events. At least 80
initial total PP per member and at most 60 attacks per member rule out total PP
exhaustion/Struggle within this admitted ceiling. Other excluded mechanics are
unchanged. The evidence is limited to this profile and the fixed fixture catalog.

A window pauses only at a fresh joint request after all forced replacements.
The mechanical observation retains its real legal request; the window collection
mask is empty while paused. Paused/terminal lanes are exact no-ops, retaining the
game ID, group counter and event addresses. Resume uses the same state, rather
than constructing a new battle. Complete public history is preserved and both
player projections agree with their received Showdown wire throughout.

The simulator-carry format binds all simulator arrays, game IDs, master seed,
boundary, version and admitted source/runtime. Recovery reproduces the original
next simulator step. Its private fields are audit/simulation state, never actor
input. Host validation rejects malformed shape, counter, phase, padding and event
schema; it does not prove arbitrary supplied state reachable. The scalar oracle
assumes admitted generated states and has weaker host validation. Exclusive carry
writes reject overwrites/truncation; no power-loss durability claim is made.

Reference parity conditions on observed semantic chance outcomes. Separate
address/mapping checks test the independent JAX chance sampler. Carrying the
simulator does not recreate Showdown's sodium PRNG stream from a JAX seed.

## Review, evidence and preservation

The independent scalar implementation passed opposite-model Fable 5.1 QA and
executed checks. A completed full Fable 5.1 audit found no blocking issue. Two low
findings prompted stricter intermediate-replacement coverage and history event
validation. Those fixes passed new corruption cases and the final 510-test
regression. The focused follow-up hit the provider weekly limit before review and
is explicitly **unavailable, not approval**. See
[review dispositions](../artifacts/reviews/m5-continuation/review-dispositions.md).

The initial import-path test-harness failure and the first preflight report-write
failure were preserved. A second preflight passed; the final production run was
then separately executed against the frozen source. No failed attempt is counted
as qualification and no outcomes or seeds were selected away.

M3 and M4 remain unchanged: 1,822 live/archive source and evidence hashes were
verified. Their historical documents are preserved; this new handoff records M5.
The [M5 receipt](../artifacts/reports/m5-continuation.json) binds final source,
fixture tapes, parity, carry replay, test logs, reviews and this handoff. Sealing
replays all 144 reference games and the complete scalar/JAX/window audit again.
Post-seal integrity and repeat-seal records are outside the sealed review folder.

GitHub publication remains on the existing visibility hold; no upload, push or
visibility change was performed. Local CPU remains sufficient and Prime was unused.

## Next — learner continuation

1. Define a new player encoder for expanded history/turn context; the M4
   323-feature encoder and empty-game checkpoint format are incompatible.
2. Carry live simulator lanes, game IDs and request phases between fixed-shape
   collection windows; explicitly distinguish terminal rewards, window bootstrap
   and administrative truncation. Decide terminal-lane refill separately.
3. Preserve same-player return linkage and detached bootstrap values under the
   collecting policy. Add optimizer/buffer/live-game checkpoint identity and exact
   original-next-update replay before a learning campaign.
4. Run short CPU learner parity/canaries first. Sustained qualification and a
   preregistered learning screen are later gates; no strength claim follows from
   these scripted reference fixtures.

The60-turn ceiling, full G2/G3/G4 and public ladder strength remain open. No public
server, final reserved data or strength screen was used.

## Reproduce

Use the recorded runtime and archived source for historical reproduction. Current
source admission intentionally fails after later source additions or edits.

```sh
node packages/continuation/audit.mjs artifacts/runs/m5-continuation-v1/reference
PYTHONPATH=python .venv/bin/python -m pokemon_continuation.parity artifacts/runs/m5-continuation-v1/reference
.venv/bin/python scripts/seal-continuation.py artifacts/runs/m5-continuation-v1
```

`scripts/verify-continuation.py` runs the full regression into a fresh review
directory. Do not restart production into an existing directory or modify sealed
evidence. An unchanged repeat seal must produce identical receipt bytes.
