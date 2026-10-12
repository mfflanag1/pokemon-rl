Implementation is complete; no further tool calls are needed — all required inputs were read up front and the only owned file has been written.

## Result

Implemented `validate_summary(report, plan)` in `python/pokemon_soak/gates.py` (the only path edited). Stdlib only (`math`); returns `None` on a valid summary, raises `ValueError` for anything invalid or missing (including non-mapping `report`/`plan`).

**Checks performed, in order:**

1. **Contract (`plan`)** – mapping; `version == "policy-chance-soak-1"`; `qualification_seconds == 7200`; positive int `warmup_cycles`, `telemetry_seconds` (=60), `checkpoint_seconds` (=600), `minimum_cycles`; finite nonnegative `maximum_host_gap_seconds`; `full_g2`/`full_g3 == "open"`; `strength_evaluation`/`prime_used` are `False`; `smoke.seconds < target` so the smoke profile can't be confused with the qualification target.
2. **Scope (`report`)** – `status == "passed"`, `mode == "qualification"`, `two_hour_qualification is True`, `full_g2`/`full_g3 == "open"`, `prime_used`/`strength_evaluation` `False`, `final_reservation_unchanged True`. Strict bool checks (`is`), and equality guards against `True == 1` coercion.
3. **Timing** – `target_seconds == plan.qualification_seconds`; `elapsed_seconds` finite, positive, `>= target`; `cycle_seconds` positive, `<= elapsed`; `process_cpu_seconds` positive; `maximum_host_gap_seconds` finite, nonnegative, `<= plan limit`. NaN/inf rejected via `math.isfinite`.
4. **Counter linkage** – all aggregate and start/final counters are nonnegative `int` and not `bool`; `cycles >= minimum_cycles`; `games == cycles*64`; `games*4 <= decisions <= games*8`; `decisions-games <= executed_moves <= decisions`; `initial.updates == warmup_cycles`; `initial.next_game == updates*64`; each `final - initial` delta equals the matching aggregate (`updates→cycles`, `next_game→games`, `decisions`, `executed_moves`).
5. **Compilation controls** – `first_call_compilation_events` positive int; `same_shape_repeat_events == 0`; `warmup_compilation_count` positive; `unexpected_compilations == 0`; `traces` has exactly `{cycle, collect, update}` each `== 1`.
6. **Diagnostics** – `minimum_exp_behavior_logp` finite in `[2^-24·(1−3e-6), 1+3e-6]`; `maximum_update_ratio_error` finite in `[0, 2^24+16]`.
7. **Evidence counts** – derived from the plan rather than hard-coded: `telemetry_rows >= 7200/60 − 1 = 119`; `periodic_replays == 7200/600 − 1 = 11`; `replays == 11 + 2 = 13`; `peak_process_rss_bytes` positive int; `device_memory_headroom_certified` must be `False` (rejects headroom claims).

I traced every case in `tests/soak/test_gates.py` (the two passing fixtures, the 17 scope/compile mutations, the 18 numerical mutations, and the 4 final-counter `+1` mutations) against this logic and each maps to a specific raise or pass path.

## Limitations

- I did **not** execute the test suite (no shell tool available, and the controller runs acceptance checks separately), so the trace above is by inspection only.
- The validator establishes only summary-level consistency; it does not and cannot verify telemetry files, checkpoint fingerprints, or source provenance — that remains with the coordinator sealer, as the docstring states.
- Fields present in the report but not in the spec (e.g. extra keys) are ignored rather than rejected; the task did not ask for strict key sets outside `traces`.
- Plan-level strictness (`telemetry_seconds == 60`, `checkpoint_seconds == 600`) intentionally pins this gate to the frozen 7200/600 contract; a future contract revision would need a deliberate update here.
