**Provisional verdict: findings (no hard blocker in reviewed scope, one gap-detection weakness and several operational risks to address before launch).**

**Most important:** the host-suspension check relies on `time.monotonic()`, which on macOS is `mach_absolute_time()` and does **not** advance during system sleep. A sleep/suspend would therefore be invisible to both `elapsed` and `max_gap`, so "any host-loop gap above 30s fails" is not actually enforced for the suspension case the task calls out.

## Findings

**1. (Medium) Suspension undetected on macOS — `campaign.py:91,95–99,115`.** `beg`, `now`, `previous` all use `time.monotonic()`. Nothing cross-checks wall-clock. `started_at`/`ended_at` (UTC) exist in the report but neither `gates.py` nor the sealer compares `ended_at − started_at` against `compile_seconds + warmup + elapsed`. Suggested fix before freeze: record `time.time()` per cycle alongside monotonic and `require(wall_delta − mono_delta <= small_slack)`, and/or have the sealer bound `(ended_at − started_at) − (soak.elapsed + compile_seconds)` (warmup/replay-start time is unmeasured; use a generous but finite slack). Also `caffeinate -i` does not prevent lid-close sleep; use `-dims` or run with lid open.

**2. (Low) No upper bound on elapsed — `gates.py:196–198`, sealer:42.** `elapsed_seconds` must be ≥7200 but can be arbitrarily large; the per-row ≤30s delay check doesn't bound the tail after the last telemetry row. Add `elapsed <= target + max_gap`.

**3. (Low) Final replay not gap-checked — `campaign.py:114–118`.** The comment says final checkpoint work is included, but `max_gap` is finalized at line 115 before the `soak-final.json` replay at line 118. Minor inconsistency with the timing-scope text in `plan.json`.

**4. (Low, operational) Sealer end-to-end path first exercised on the 2h output.** Smoke produces 2 periodic replays and ~5 rows; sealer hardcodes 11 boundaries (line 55) and 119–120 rows (line 25), so `main()` cannot be dry-run on smoke. `test_campaign.py` covers `validate_telemetry` synthetically but not the replay-roster/fingerprint/history loop (lines 55–69, 85–93). A sealer bug surfaces only after the long run. Consider a parametrized roster/rows expectation derived from the plan so smoke output can be sealed in a non-qualifying mode, or a synthetic fixture test for lines 55–69.

**5. (Low, operational) Stale run directories block sealing — sealer:85–93.** `history` globs `m1-policy-soak-*`; a SIGKILLed/crashed run whose `report.json` is still the line-142 "running" stub has no `content_hash`, so `sealed()` raises and the real run cannot be sealed until the directory is removed. Same for a report with a missing or stale `report.json` after a signal during `write()`.

**6. (Info) Checkpoint↔telemetry linkage is sound.** Telemetry and checkpoint for `k·600` fire in the same iteration (telemetry first), so `rows[i*10-1].counters == boundary counters`; the `<=` bracket at sealer:67–69 holds. Replays don't mutate the runner (verified by `test_original_and_restored_fingerprints`), so primary counters exclude audits as claimed.

**7. (Info) Fail-closed behaviour checks out.** `BaseException` catch covers `InterruptedError` from SIGTERM/SIGINT and `ValueError` from `validate_summary`; `two_hour_qualification` is reset to `False` on any failure; `sealed()` recomputes the hash over the report minus `content_hash`, consistent with line 163.

## Scope limitations
- Cannot execute; macOS `time.monotonic` behaviour is from CPython `pytime.c` knowledge, not verified here.
- `SUITES`, `node_pass_count==65`, suite ordering, and `__init__.py` contents of `pokemon_soak` not in packet.
- `wire`/`digest` JSON handling of numpy floats and `memory_stats()` (`None` on CPU) assumed working from predecessor use.
- Gates.py correctness delegated elsewhere; only its integration reviewed.

## Verdict
Findings. No fingerprint/counter/sealer-binding defect found; address #1 (and ideally #2, #4) before freezing qualification sources.
