# Bounded gate implementation

Own only python/pokemon_switch_soak/gates.py. Standard library only. Implement
validate(report, plan, rows), returning None or raising ValueError for all invalid
inputs (including missing fields, wrong types, booleans in numeric fields and NaN).
The coordinator owns the runner/checkpoints/sealer and tests. You are not alone;
do not edit other files or recursively delegate.

Frozen plan is switch-soak-1. Reject any different version, target (7200), warmup
(8), telemetry (60), checkpoint (600), host gap (30), seed (101), max updates
(10000000), minimum cycles (1000) or RSS cap (2147483648). Smoke is exactly
20/4/8 seconds. All frozen numeric plan constants require non-boolean int types,
not equal floats. Scope fields are also frozen exactly as the supplied plan;
reject any scope relaxation even if report and altered plan agree. Read report mode qualification or smoke; passed required. Only
qualification has two_hour_qualification true. Both preserve every scope field
from the plan (full_g2/g3/g4, public_ladder_strength, strength_evaluation,
final_evaluation_used, public_server_used, prime_used). Require
final_reservation_unchanged true; positive real compile control first events and
zero same-shape events.

Report.soak has elapsed_seconds, process_cpu_seconds, cycle_seconds,
maximum_host_gap_seconds, unexpected_compilations, warmup_compilation_count,
minimum_logp, maximum_logp, peak_process_rss_bytes, cycles, games, decisions,
terminal, truncated, actions (four integer counts), initial_counters and
final_counters (updates, optimizer_step, next_game), telemetry_rows, replays,
decisions_per_elapsed_second. All numerical fields must be finite and appropriately
typed; counts are nonnegative integers (no bool). Elapsed target..target+30;
positive CPU and cycle seconds; cycle_seconds<=elapsed_seconds; max gap0..30;
unexpected0, warmup compile>0, peak RSS>0<=cap. Behavior log probabilities satisfy
-48*ln(2)-1e-5 <= minimum_logp <= maximum_logp <=1e-6.

Qualification cycles>=1000 (smoke>=1); games=32*cycles;
64*cycles<=decisions<=896*cycles; terminal+truncated=games;
sum(actions)=decisions. Counter triples strictly link optimizer_step=2*updates,
next_game=32*updates, 0<=updates<=maxupdates. Initial updates8;
final updates=8+cycles. Throughput exactly decisions/elapsed.
Replays = start+final+periodic:13 qualification,4 smoke (8 and16seconds).

Rows list must equal telemetry_rows. Count target//interval-1 or target//interval
(nonempty), each row scheduled_seconds=i*interval for 1-based i;
scheduled<=elapsed<=scheduled+30 and elapsed<=summary elapsed. Last row elapsed
>=target-interval. Row fields cycles/games/decisions/terminal/truncated/actions
obey the same accounting, counters equal initial counters plus row cycles and
are no later than final. Every row numerical field must be finite. Row elapsed,
cycles and decisions strictly increase; all aggregate count fields (including
each action) are nondecreasing and <= summary. Rows have maximum_host_gap_seconds,
unexpected_compilations, minimum_logp, maximum_logp, memory.peak_process_rss_bytes
with the same bounds; peak <=summary peak. The runner may include extra fields.
If the last row has the same cycle count as the summary, all its aggregate counts
must be equal. Missing/malformed inputs raise ValueError, never accidentally pass.

This checks internal consistency only. Sealer separately checks source/runtime,
report checksum, telemetry hashes, complete checkpoint roster and fresh replay.
