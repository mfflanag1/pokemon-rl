The module is written. I traced the supplied fixture and every negative case in `tests/switch-soak/test_gates.py` against the code by hand; I did not execute the tests (no shell tool is available to me), so the controller's acceptance run is the real check.

## Result

Created `python/pokemon_switch_soak/gates.py` (standard library only; `math` is the sole import). Entry point `validate(report, plan, rows)` returns `None` or raises `ValueError`.

**Plan checks** — mapping; `version == "switch-soak-1"`; exact values for qualification_seconds 7200, warmup_cycles 8, telemetry_seconds 60, checkpoint_seconds 600, maximum_host_gap_seconds 30, seed 101, maximum_updates 10 000 000, minimum_cycles 1000, maximum_peak_rss_bytes 2 147 483 648; `smoke` exactly 20/4/8; the eight scope fields must equal their frozen values (type-strict, so `True` ≠ `1`).

**Report header** — `status == "passed"`; `mode` ∈ {qualification, smoke}; `two_hour_qualification` true iff qualification; all eight scope fields type-strict equal to the plan; `final_reservation_unchanged is True`; `compile_control.first_call_compilation_events` finite real > 0, `same_shape_repeat_events == 0`.

**Summary (`report.soak`)** — every numeric field finite, bools rejected everywhere, counts are non-bool ints ≥ 0; elapsed ∈ [target, target+30]; CPU and cycle seconds > 0, cycle ≤ elapsed; gap ∈ [0, 30]; unexpected == 0; warmup compiles > 0; peak RSS in (0, cap]; `-48·ln2 − 1e-5 ≤ min_logp ≤ max_logp ≤ 1e-6`; cycles ≥ 1000 (smoke ≥ 1); games = 32·cycles; 64·cycles ≤ decisions ≤ 896·cycles; terminal+truncated = games; four non-negative action ints summing to decisions; counter triples linked (optimizer_step = 2·updates, next_game = 32·updates, 0 ≤ updates ≤ max); initial updates = 8, final updates = 8 + cycles; `telemetry_rows` ∈ {target//interval − 1, target//interval} and ≥ 1; replays = 2 + ⌈target/checkpoint⌉ − 1 (13 qualification, 4 smoke); throughput `== decisions / elapsed` exactly.

**Rows** — must be a `list` with `len == telemetry_rows`, nonempty; row *i* (1-based) has `scheduled_seconds == i·interval`, `scheduled ≤ elapsed ≤ scheduled+30`, `elapsed ≤ summary elapsed`; same accounting as the summary (cycles ≥ 0); counters = initial + row cycles and linked, every counter ≤ final; stability fields with the same bounds, `memory.peak_process_rss_bytes ≤ summary peak`; elapsed/cycles/decisions strictly increasing; all five count fields plus each action nondecreasing and ≤ summary; last row elapsed ≥ target − interval; if last-row cycles equal summary cycles, all aggregate counts (incl. actions) must equal the summary. Extra fields are tolerated.

## Limitations / judgment calls to flag

- **Not executed.** I have no shell tool, so the tests have not been run by me; the hand-trace is my only evidence.
- **Plan scope fields are pinned to frozen values** (`open`/`unqualified`/`false`), not just mirrored from report to plan. The spec lists only version/numeric fields as "reject any different", so this is slightly stricter than a literal reading; it is consistent with the plan being frozen and passes the supplied tests. Easy to relax if the coordinator wants plan scope to float.
- **"Exactly" throughput** uses `==` on `decisions / elapsed_seconds`; a runner that computes it via a different float path could fail this gate. The spec says "exactly", so I kept it strict.
- **Counts must be Python `int`** (JSON integers); `7200.0` for a count field is rejected. Real-valued fields (elapsed, CPU, gap, logp, scheduled_seconds, compile-control events) accept int or finite float.
- **Replay count** is derived (`2 + ⌈target/checkpoint⌉ − 1`) rather than hard-coded; it yields 13/4 for the frozen plan, so there is no behavioral difference unless the frozen constants change (which the plan check already forbids).
- `rows` must be a `list`; a tuple is rejected. Only `gates.py` was touched; the `python/pokemon_switch_soak/` directory was created implicitly by the write and contains no `__init__.py` (the test loads the module by path, so none is needed; add one if the package is imported elsewhere).
