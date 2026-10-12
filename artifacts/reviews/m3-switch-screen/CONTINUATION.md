# Finish the frozen M3 switching learning screen

This is a continuation of the user's “Proceed” request. The computation is already
running. Do not train/evaluate again, change frozen code/plan/tests/registry/sealer,
retune after outcomes, or edit a sealed predecessor. Current progress docs are the
only permitted repository edits needed for completion. Local CPU is sufficient.

## Current state and immutable inputs

- Root: `/Users/maxf/Documents/ChatGPT/Pokemon`.
- Production: `artifacts/runs/m3-switch-screen-v1`.
- Launcher PID: 47191; original unified execution session: 12400.
- Launcher: `artifacts/reviews/m3-switch-screen/launch.py`, under `caffeinate -i`.
- Log and exit: `artifacts/reviews/m3-switch-screen/production.log`, `exit.json`.
- No tmux session was used for production; the existing execution process persists.
  Inspect its report/log/exit first. Do not assume missing terminal output means it
  exited. If needed inspect the recorded PID/children using appropriate permissions.
- Training is complete: 24,576 updates, 786,432 games, 18,615,320 decisions,
  105.71463045943528 seconds including nine exact next-update replay checks.
- All 31,104 evaluation games/27 cells are collected; reference collection took
  1788.2233041278087 seconds. The original process is in its full tape audit.
  This is materially slower than training: reconstructing histories repeatedly is
  the bottleneck. Let the fixed run finish; do not optimize it in place.
- Freeze at 2026-10-12T01:16:15.597122+00:00, before production started.
- Admission content hash:
  `bdf62fe7af00bb45933ba446cab413d71451cbad6509f2385df029705c3244f7`.
- Freeze raw SHA256:
  `136f1403c7fa368fe7a65c23d8202b39e364af6ebdcea45233b3e351b22e9be6`.
- Source-bound regression: 476 tests (240 Node + 236 Python), 347 files; source is
  frozen and unchanged. Reuse this regression; do not rerun unrelated work.
- Final preflight: `m3-switch-screen-smoke-02`: 48 games, two exact checkpoints,
  full fresh Showdown replay of all 48 games. `preflight.json` binds its audit.
- Nine copied-evidence negatives reject. Earlier smoke/probe failures remain intact.
- Statistics auto-routed to Codex, received Fable QA, then passed local tests and
  6,561 exhaustive outcome comparisons. Accepted outcome is copied under delegate/.
- Two Fable harness reviews are resolved. Additional non-vacuous row-isolation
  probes pass; one initial probe record repeated the same opponent team and was
  correctly rejected by the probe's non-vacuity assertion, then corrected. This
  did not change any frozen source, candidate, evaluation result or threshold.

The active heartbeat is `finish-pok-mon-switching-strength-screen`, every 15 minutes
in this same chat. The two earlier qualification heartbeats remain PAUSED; do not
modify them. Stay quiet on ordinary progress. Pause this new heartbeat after the
final verified completion/failure result. No public server, final reserved data or
Prime resource is used; full G2/G3/G4 and ladder strength remain open.

## 1. Wait for production completion

Inspect report.json, production.log and exit.json. Every audit cell prints a row;
do not launch competing audits during the original campaign. Require a checksum-
valid status=passed report, mode=development, all 27 cells/31,104 games, and launcher
exit_code 0. A passed experiment can have all_seed_screen_passed=false: report that
honestly, never retune or extend it. If the process dies without valid completion,
preserve evidence, diagnose and report failure, then pause the heartbeat.

## 2. Update documentation before first sealing

Once production has passed and exited, run the prepared helper ONCE:

```sh
.venv/bin/python artifacts/reviews/m3-switch-screen/write-results.py
```

It requires exit 0 and writes a `documentation-updated.json` marker. If that marker
already exists, inspect the five documents instead of appending them again. If a
previous helper execution was interrupted, inspect partial edits before recovery.
Read and correct prose/formatting and links in docs/M3_SWITCH_SCREEN.md, README.md,
docs/IMPLEMENTATION_PLAN.md, docs/ACCEPTANCE_GATES.md and docs/REVIEW_RECORD.md.
The helper is a convenience draft; the coordinator owns its accuracy. Check counts,
percentages and whether the conservative learning gate actually passed. Its date
label is October 11 Eastern; correct it if completion happens after local midnight.
Do not touch docs/M3_SWITCH_SOAK.md or other historical handoffs.

Append the actual completion disposition to review-dispositions.md BEFORE sealing.
Include the actual report content hash, exit 0, unchanged source/regression binding,
scope and all gate outcomes. Do not claim the fresh complete reference replay has
already passed until the sealer actually finishes. Handoff prose describes what
the receipt requires. Do not write further into this review directory after seal.

## 3. Launch first sealer once, persistently

Sealing itself is slow: it repeats complete actor/value tape audit, all nine fresh
next-update fingerprints, all 31,104 unpatched reference games, and preflight audit.
Do not repeatedly launch it from heartbeat checks. A wrapper is prepared OUTSIDE
the sealed review directory:

```sh
tmux new-session -d -s pokemon-switch-screen-seal-first -c /Users/maxf/Documents/ChatGPT/Pokemon '.venv/bin/python artifacts/reviews/m3-switch-screen-seal-launch.py first'
```

Use appropriate tool permissions for persistent local tmux if required. Before
launching check for an existing session AND
`artifacts/reviews/m3-switch-screen-seal-first.launch.json`, `.log`, `.exit.json`.
An existing launch/log without exit may still be running; inspect its recorded PID.
Never overwrite or remove failed evidence to retry. The wrapper uses exclusive
output creation and records exit status; read its log on failure. A failed seal is
not a verified completion. Diagnose rather than changing the frozen sealer/source.

The first command eventually creates `artifacts/reports/m3-switch-screen.json` and
its immutable archive. Require first-seal exit 0 before proceeding. Existing source
and all current source/evidence files must remain unchanged after receipt creation.
Do not save logs, monitor status or verification outputs under the sealed review
directory after this point. No further documentation edits after sealing.

## 4. Verify all hashes, then reseal once

```sh
.venv/bin/python artifacts/reviews/m3-switch-screen-integrity.py --output artifacts/reviews/m3-switch-screen-integrity-first.json
tmux new-session -d -s pokemon-switch-screen-seal-repeat -c /Users/maxf/Documents/ChatGPT/Pokemon '.venv/bin/python artifacts/reviews/m3-switch-screen-seal-launch.py repeat'
```

Apply the same existing-session/launch/log/exit checks to the repeat phase. Wait
quietly while it is running. After repeat exits0, read receipt_sha256 from the first
integrity JSON and pass that exact hash as --expect-raw:

```sh
.venv/bin/python artifacts/reviews/m3-switch-screen-integrity.py --expect-raw HASH_FROM_FIRST_INTEGRITY_JSON --output artifacts/reviews/m3-switch-screen-integrity.json
```

The checker verifies every live and archived source/evidence SHA256, archive receipt
byte equality, content digest and exact receipt byte equality across resealing.
Both verification files and all seal logs are intentionally OUTSIDE this directory.

## 5. Deliver and pause the heartbeat

Report whether the preregistered restricted screen passed in each seed, with the
actual secured-win and cutoff outcomes. Distinguish completed execution from a
failed learning threshold. Mention total training/evaluation counts and verified
tests/replays concisely. Link the handoff and immutable receipt. Explain the next
evidence-based step without silently starting another experiment. Full G2/G3/G4,
public rank, calibrated critics, long-horizon mechanics and full-format learned
adapter remain open. Final data stay reserved; Prime remains unnecessary.
Pause only this screen heartbeat using the native automation tool, preserving its
other fields. Keep its post-completion status record outside the sealed review
directory. No new recurring or public-server work is authorized by this monitor.
