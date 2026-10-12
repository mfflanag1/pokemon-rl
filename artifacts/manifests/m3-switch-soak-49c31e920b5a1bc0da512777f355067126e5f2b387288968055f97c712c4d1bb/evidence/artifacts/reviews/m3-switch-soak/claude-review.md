**Provisional verdict: findings — no concrete false-pass path identified in reviewed scope; two material false-negative/fragility defects that will likely block sealing, plus unverifiable dependencies.**

## Findings

**1. Medium — smoke-mode audit bracket is structurally unsatisfiable in the common timing case** (`scripts/seal-switch-soak.py:47-50` vs `python/pokemon_switch_soak/campaign.py:65-81`).
For an interior boundary the sealer requires a telemetry row with `elapsed_seconds > item['elapsed_seconds']`. In `run`, the telemetry row and the boundary replay fire in the *same* iteration when `next_log==next_checkpoint` (smoke: both 16; qualification: 600, 1200, …), with identical `elapsed`. The next row would be scheduled at `seconds` (20), but the loop exits on `time.monotonic()-beg<seconds` before any row at `elapsed>=seconds` can be written (line 65), and no row is written post-loop. So for smoke boundary-02 (16 s), `after` is empty unless telemetry happened to fall behind by one interval. `seal()` requires `audit(preflight_run,smoke=True)` (line 70), so the qualification seal is blocked by preflight. Qualification itself is fine (boundaries end at 6600, rows continue to 7140). Verification: run `--smoke` then `--audit-smoke`; expect `Checkpoint/telemetry bracket` failure.

**2. Low/Medium — telemetry rows are dropped when one iteration spans >1 telemetry interval** (`campaign.py:70-78`). `if elapsed>=next_log` emits one row and advances `next_log` once per iteration. A boundary replay (save+load+2 transactions ≈ 3 cycle latencies) inside the loop can skip a whole interval in smoke (4 s telemetry, 8 s checkpoint). Rows would then be `scheduled=8,12` written late and the gate's `target//interval-1` minimum (4 rows) fails. Not a false pass, but makes smoke preflight flaky. Qualification margin (60 s telemetry, cycles ≪ 7.2 s given `minimum_cycles`) is adequate.

**3. Low — hard-coded kernel constants** (`checkpoint.py:30-31,78`; `campaign.py:37`): `2*updates`, `32*updates`, metric shape `(2,)`, `games=32*cycles` duplicate `learner.CONFIG['epochs']`/`BATCH` rather than referencing them. Harmless while the predecessor is sealed; fails loudly if ever mismatched. Note only.

**No issues found** in: write-once/fsync checkpoint path; `.partial` preservation; checksum-with-duplicate-key rejection; binding includes `mode` (smoke checkpoints cannot be replayed as qualification); audit transactions do not publish counters; sealer independently recomputes warm-up identity and next-update fingerprints in a fresh process; `validate` gate failure correctly flips `status` to `failed` before `content_hash` (lines 116-125); baseline compile count captured after start replay so load→transaction path is covered; counter range 1e7·32 < 2³¹.

## Scope limitations
- `clock_pair`/`check_gap`/`write`/`utc` (pokemon_soak), `CompileLog`/`compile_positive_control`/`memory` (pokemon_fixed), `read_array`, `seal-switch-engine.py` `admission`/`write_once`, and `gates.py` not in packet. `beg=previous[0]` assumes index 0 is monotonic; `check_gap` test is symmetric and cannot confirm ordering. Compile detection relies on the `jax._src.dispatch` logger with `propagate=False` — correctness depends on the positive control truly asserting detected events.
- Cannot execute; cycle latency and RSS units (`ru_maxrss` platform semantics) unverified.
- Sealer test-count literals (237, `SUITES`) assumed to match current suites.

## Verdict
Findings. Core replay/sealer integrity logic appears sound; Finding 1 should be fixed (e.g., write a final telemetry row after the loop, or bracket with `>=` on the row emitted in the replay iteration) before production, since it blocks the preflight-dependent seal.
