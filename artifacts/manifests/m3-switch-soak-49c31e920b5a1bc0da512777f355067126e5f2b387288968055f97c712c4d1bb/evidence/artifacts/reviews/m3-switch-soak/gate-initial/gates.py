"""Pure switch soak gate (switch-soak-1).

Standard library only.  ``validate(report, plan, rows)`` returns ``None`` when
the report, frozen plan and telemetry rows are internally consistent and
raises ``ValueError`` for every other input (missing fields, wrong types,
booleans in numeric fields, NaN/inf, wrong scope, instability, broken counter
linkage, under-duration, mismatched telemetry).

This gate checks internal consistency only.  Source/runtime binding, report
checksums, telemetry hashes, checkpoint roster and fresh replay are owned by
the sealer.
"""

import math

__all__ = ["validate", "FROZEN_PLAN", "SCOPE_FIELDS"]

# --------------------------------------------------------------------------
# Frozen plan constants
# --------------------------------------------------------------------------

PLAN_VERSION = "switch-soak-1"

FROZEN_PLAN = {
    "qualification_seconds": 7200,
    "warmup_cycles": 8,
    "telemetry_seconds": 60,
    "checkpoint_seconds": 600,
    "maximum_host_gap_seconds": 30,
    "seed": 101,
    "maximum_updates": 10000000,
    "minimum_cycles": 1000,
    "maximum_peak_rss_bytes": 2147483648,
}

FROZEN_SMOKE = {
    "seconds": 20,
    "telemetry_seconds": 4,
    "checkpoint_seconds": 8,
}

SCOPE_FIELDS = (
    "full_g2",
    "full_g3",
    "full_g4",
    "public_ladder_strength",
    "strength_evaluation",
    "final_evaluation_used",
    "public_server_used",
    "prime_used",
)

FROZEN_SCOPE = {
    "full_g2": "open",
    "full_g3": "open",
    "full_g4": "open",
    "public_ladder_strength": "unqualified",
    "strength_evaluation": False,
    "final_evaluation_used": False,
    "public_server_used": False,
    "prime_used": False,
}

GAMES_PER_CYCLE = 32
MIN_DECISIONS_PER_CYCLE = 64
MAX_DECISIONS_PER_CYCLE = 896
OPTIMIZER_STEPS_PER_UPDATE = 2
GAMES_PER_UPDATE = 32
ACTION_COUNT = 4

LOGP_FLOOR = -48.0 * math.log(2.0) - 1e-5
LOGP_CEILING = 1e-6

COUNT_FIELDS = ("cycles", "games", "decisions", "terminal", "truncated")
COUNTER_FIELDS = ("updates", "optimizer_step", "next_game")


# --------------------------------------------------------------------------
# Primitive helpers
# --------------------------------------------------------------------------


def _fail(message):
    raise ValueError(message)


def _mapping(value, name):
    if not isinstance(value, dict):
        _fail("%s must be a mapping" % name)
    return value


def _get(mapping, key, name):
    if key not in mapping:
        _fail("%s missing field %r" % (name, key))
    return mapping[key]


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _int(mapping, key, name, minimum=0):
    value = _get(mapping, key, name)
    if not _is_int(value):
        _fail("%s.%s must be an integer (not bool)" % (name, key))
    if value < minimum:
        _fail("%s.%s must be >= %r" % (name, key, minimum))
    return value


def _real(mapping, key, name):
    value = _get(mapping, key, name)
    if not _is_real(value):
        _fail("%s.%s must be a finite real number (not bool)" % (name, key))
    return value


def _bool(mapping, key, name):
    value = _get(mapping, key, name)
    if not isinstance(value, bool):
        _fail("%s.%s must be a boolean" % (name, key))
    return value


def _same(actual, expected):
    """Exact equality including type class (bool vs int vs str)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual is expected
    if isinstance(expected, str) or isinstance(actual, str):
        return isinstance(actual, str) and isinstance(expected, str) and actual == expected
    if _is_real(expected):
        return _is_real(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _actions(mapping, name):
    value = _get(mapping, "actions", name)
    if not isinstance(value, list) or len(value) != ACTION_COUNT:
        _fail("%s.actions must be a list of four integer counts" % name)
    out = []
    for index, item in enumerate(value):
        if not _is_int(item) or item < 0:
            _fail("%s.actions[%d] must be a nonnegative integer" % (name, index))
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


def _check_plan(plan):
    _mapping(plan, "plan")
    version = _get(plan, "version", "plan")
    if not isinstance(version, str) or version != PLAN_VERSION:
        _fail("plan.version must be %r" % PLAN_VERSION)
    for key, expected in FROZEN_PLAN.items():
        value = _get(plan, key, "plan")
        if not _is_real(value) or value != expected:
            _fail("plan.%s must be %r" % (key, expected))
    smoke = _mapping(_get(plan, "smoke", "plan"), "plan.smoke")
    for key, expected in FROZEN_SMOKE.items():
        value = _get(smoke, key, "plan.smoke")
        if not _is_real(value) or value != expected:
            _fail("plan.smoke.%s must be %r" % (key, expected))
    for key, expected in FROZEN_SCOPE.items():
        value = _get(plan, key, "plan")
        if not _same(value, expected):
            _fail("plan.%s must be %r" % (key, expected))


# --------------------------------------------------------------------------
# Accounting shared by summary and rows
# --------------------------------------------------------------------------


def _counts(mapping, name, minimum_cycles):
    counts = {}
    for key in COUNT_FIELDS:
        counts[key] = _int(mapping, key, name)
    counts["actions"] = _actions(mapping, name)
    cycles = counts["cycles"]
    if cycles < minimum_cycles:
        _fail("%s.cycles must be >= %d" % (name, minimum_cycles))
    if counts["games"] != GAMES_PER_CYCLE * cycles:
        _fail("%s.games must equal %d*cycles" % (name, GAMES_PER_CYCLE))
    decisions = counts["decisions"]
    if not (MIN_DECISIONS_PER_CYCLE * cycles <= decisions <= MAX_DECISIONS_PER_CYCLE * cycles):
        _fail("%s.decisions outside [%d, %d] per cycle" % (name, MIN_DECISIONS_PER_CYCLE, MAX_DECISIONS_PER_CYCLE))
    if counts["terminal"] + counts["truncated"] != counts["games"]:
        _fail("%s terminal+truncated must equal games" % name)
    if sum(counts["actions"]) != decisions:
        _fail("%s sum(actions) must equal decisions" % name)
    return counts


def _counters(mapping, key, name, maximum_updates):
    triple = _mapping(_get(mapping, key, name), "%s.%s" % (name, key))
    full = "%s.%s" % (name, key)
    out = {}
    for field in COUNTER_FIELDS:
        out[field] = _int(triple, field, full)
    updates = out["updates"]
    if updates > maximum_updates:
        _fail("%s.updates exceeds maximum updates" % full)
    if out["optimizer_step"] != OPTIMIZER_STEPS_PER_UPDATE * updates:
        _fail("%s.optimizer_step must equal %d*updates" % (full, OPTIMIZER_STEPS_PER_UPDATE))
    if out["next_game"] != GAMES_PER_UPDATE * updates:
        _fail("%s.next_game must equal %d*updates" % (full, GAMES_PER_UPDATE))
    return out


def _stability(mapping, name, peak_key_mapping, max_gap, rss_cap):
    gap = _real(mapping, "maximum_host_gap_seconds", name)
    if not (0 <= gap <= max_gap):
        _fail("%s.maximum_host_gap_seconds outside [0, %r]" % (name, max_gap))
    unexpected = _int(mapping, "unexpected_compilations", name)
    if unexpected != 0:
        _fail("%s.unexpected_compilations must be zero" % name)
    minimum_logp = _real(mapping, "minimum_logp", name)
    maximum_logp = _real(mapping, "maximum_logp", name)
    if not (LOGP_FLOOR <= minimum_logp <= maximum_logp <= LOGP_CEILING):
        _fail("%s log probabilities outside admissible range" % name)
    peak_name, peak_mapping = peak_key_mapping
    peak = _int(peak_mapping, "peak_process_rss_bytes", peak_name, minimum=1)
    if peak > rss_cap:
        _fail("%s.peak_process_rss_bytes exceeds cap" % peak_name)
    return peak


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _check_report_header(report, plan):
    _mapping(report, "report")
    status = _get(report, "status", "report")
    if not isinstance(status, str) or status != "passed":
        _fail("report.status must be 'passed'")
    mode = _get(report, "mode", "report")
    if not isinstance(mode, str) or mode not in ("qualification", "smoke"):
        _fail("report.mode must be 'qualification' or 'smoke'")
    smoke = mode == "smoke"
    two_hour = _bool(report, "two_hour_qualification", "report")
    if two_hour is smoke:
        _fail("report.two_hour_qualification must be true only for qualification")
    for key in SCOPE_FIELDS:
        value = _get(report, key, "report")
        if not _same(value, plan[key]):
            _fail("report.%s must preserve plan scope value %r" % (key, plan[key]))
    if _bool(report, "final_reservation_unchanged", "report") is not True:
        _fail("report.final_reservation_unchanged must be true")
    control = _mapping(_get(report, "compile_control", "report"), "report.compile_control")
    first = _real(control, "first_call_compilation_events", "report.compile_control")
    if not first > 0:
        _fail("report.compile_control.first_call_compilation_events must be positive")
    same_shape = _real(control, "same_shape_repeat_events", "report.compile_control")
    if same_shape != 0:
        _fail("report.compile_control.same_shape_repeat_events must be zero")
    return smoke


def _check_summary(soak, smoke):
    name = "report.soak"
    _mapping(soak, name)
    if smoke:
        target = FROZEN_SMOKE["seconds"]
        interval = FROZEN_SMOKE["telemetry_seconds"]
        checkpoint = FROZEN_SMOKE["checkpoint_seconds"]
        minimum_cycles = 1
    else:
        target = FROZEN_PLAN["qualification_seconds"]
        interval = FROZEN_PLAN["telemetry_seconds"]
        checkpoint = FROZEN_PLAN["checkpoint_seconds"]
        minimum_cycles = FROZEN_PLAN["minimum_cycles"]
    max_gap = FROZEN_PLAN["maximum_host_gap_seconds"]
    rss_cap = FROZEN_PLAN["maximum_peak_rss_bytes"]
    warmup = FROZEN_PLAN["warmup_cycles"]
    maximum_updates = FROZEN_PLAN["maximum_updates"]

    elapsed = _real(soak, "elapsed_seconds", name)
    if not (target <= elapsed <= target + max_gap):
        _fail("%s.elapsed_seconds outside [%d, %d]" % (name, target, target + max_gap))
    cpu = _real(soak, "process_cpu_seconds", name)
    if not cpu > 0:
        _fail("%s.process_cpu_seconds must be positive" % name)
    cycle_seconds = _real(soak, "cycle_seconds", name)
    if not cycle_seconds > 0:
        _fail("%s.cycle_seconds must be positive" % name)
    if cycle_seconds > elapsed:
        _fail("%s.cycle_seconds must not exceed elapsed_seconds" % name)
    peak = _stability(soak, name, (name, soak), max_gap, rss_cap)
    warmup_compiles = _int(soak, "warmup_compilation_count", name)
    if warmup_compiles <= 0:
        _fail("%s.warmup_compilation_count must be positive" % name)

    counts = _counts(soak, name, minimum_cycles)
    cycles = counts["cycles"]
    initial = _counters(soak, "initial_counters", name, maximum_updates)
    final = _counters(soak, "final_counters", name, maximum_updates)
    if initial["updates"] != warmup:
        _fail("%s.initial_counters.updates must equal warm-up cycles %d" % (name, warmup))
    if final["updates"] != warmup + cycles:
        _fail("%s.final_counters.updates must equal warm-up + cycles" % name)

    telemetry_rows = _int(soak, "telemetry_rows", name)
    expected_rows_full = target // interval
    if telemetry_rows not in (expected_rows_full - 1, expected_rows_full) or telemetry_rows < 1:
        _fail("%s.telemetry_rows must be %d or %d" % (name, expected_rows_full - 1, expected_rows_full))

    replays = _int(soak, "replays", name)
    periodic = -(-target // checkpoint) - 1  # checkpoints strictly before the final one
    expected_replays = 2 + periodic
    if replays != expected_replays:
        _fail("%s.replays must equal %d" % (name, expected_replays))

    throughput = _real(soak, "decisions_per_elapsed_second", name)
    if throughput != counts["decisions"] / elapsed:
        _fail("%s.decisions_per_elapsed_second must equal decisions/elapsed_seconds" % name)

    return {
        "target": target,
        "interval": interval,
        "max_gap": max_gap,
        "rss_cap": rss_cap,
        "maximum_updates": maximum_updates,
        "elapsed": elapsed,
        "peak": peak,
        "counts": counts,
        "initial": initial,
        "final": final,
        "telemetry_rows": telemetry_rows,
    }


# --------------------------------------------------------------------------
# Telemetry rows
# --------------------------------------------------------------------------


def _check_rows(rows, summary):
    if not isinstance(rows, list):
        _fail("rows must be a list")
    if len(rows) != summary["telemetry_rows"]:
        _fail("rows length must equal report.soak.telemetry_rows")
    if not rows:
        _fail("rows must be nonempty")

    target = summary["target"]
    interval = summary["interval"]
    max_gap = summary["max_gap"]
    rss_cap = summary["rss_cap"]
    summary_counts = summary["counts"]
    summary_elapsed = summary["elapsed"]
    initial = summary["initial"]
    final = summary["final"]

    previous = None
    previous_elapsed = None
    for index, row in enumerate(rows):
        name = "rows[%d]" % index
        _mapping(row, name)
        scheduled = _real(row, "scheduled_seconds", name)
        if scheduled != (index + 1) * interval:
            _fail("%s.scheduled_seconds must equal %d" % (name, (index + 1) * interval))
        elapsed = _real(row, "elapsed_seconds", name)
        if not (scheduled <= elapsed <= scheduled + max_gap):
            _fail("%s.elapsed_seconds outside [scheduled, scheduled+%r]" % (name, max_gap))
        if elapsed > summary_elapsed:
            _fail("%s.elapsed_seconds exceeds summary elapsed" % name)
        if previous_elapsed is not None and elapsed <= previous_elapsed:
            _fail("%s.elapsed_seconds must strictly increase" % name)

        counts = _counts(row, name, 0)
        for key in COUNT_FIELDS:
            if counts[key] > summary_counts[key]:
                _fail("%s.%s exceeds summary" % (name, key))
            if previous is not None and counts[key] < previous[key]:
                _fail("%s.%s must be nondecreasing" % (name, key))
        for j in range(ACTION_COUNT):
            if counts["actions"][j] > summary_counts["actions"][j]:
                _fail("%s.actions[%d] exceeds summary" % (name, j))
            if previous is not None and counts["actions"][j] < previous["actions"][j]:
                _fail("%s.actions[%d] must be nondecreasing" % (name, j))
        if previous is not None:
            if counts["cycles"] <= previous["cycles"]:
                _fail("%s.cycles must strictly increase" % name)
            if counts["decisions"] <= previous["decisions"]:
                _fail("%s.decisions must strictly increase" % name)

        counters = _counters(row, "counters", name, summary["maximum_updates"])
        if counters["updates"] != initial["updates"] + counts["cycles"]:
            _fail("%s.counters.updates must equal initial updates + row cycles" % name)
        for field in COUNTER_FIELDS:
            if counters[field] > final[field]:
                _fail("%s.counters.%s is later than final counters" % (name, field))

        memory = _mapping(_get(row, "memory", name), "%s.memory" % name)
        peak = _stability(row, name, ("%s.memory" % name, memory), max_gap, rss_cap)
        if peak > summary["peak"]:
            _fail("%s.memory.peak_process_rss_bytes exceeds summary peak" % name)

        previous = counts
        previous_elapsed = elapsed

    if previous_elapsed < target - interval:
        _fail("last row elapsed_seconds must be >= target - telemetry interval")
    if previous["cycles"] == summary_counts["cycles"]:
        for key in COUNT_FIELDS:
            if previous[key] != summary_counts[key]:
                _fail("last row %s must equal summary when cycles match" % key)
        if previous["actions"] != summary_counts["actions"]:
            _fail("last row actions must equal summary when cycles match")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def validate(report, plan, rows):
    """Validate a switch-soak-1 report against the frozen plan and telemetry.

    Returns ``None`` on success; raises ``ValueError`` for any invalid input.
    """
    _check_plan(plan)
    smoke = _check_report_header(report, plan)
    summary = _check_summary(_get(report, "soak", "report"), smoke)
    _check_rows(rows, summary)
    return None
