Implement only python/pokemon_soak/gates.py. You are not alone in the codebase;
do not change or revert coordinator files. Use standard library only. Export
validate_summary(report, plan), return None for a valid qualification summary,
raise ValueError for invalid/missing inputs. The report shape is the good_report
fixture. Reject smoke and promotion/headroom claims. Require production 7200-second
target, finite positive elapsed >= target, positive cycle time <= elapsed,
positive process CPU time, finite nonnegative max host gap <= plan limit,
integer counters with cycles >= minimum_cycles, games=cycles*64, games*4 <=
decisions <= games*8, decisions-games <= executed_moves <= decisions.
Require start updates=warmup_cycles, next_game=updates*64, and final-minus-start
counters match aggregate cycles/games/decisions/moves. All counters must be
nonnegative integers, not bool. Require real compile positive and repeat controls,
positive baseline and zero unexpected events, three entry traces exactly one.
Probability floor allows FP32 exp/log rounding: 2^-24*(1-3e-6) through 1+3e-6.
Ratio diagnostic is finite in [0, 2^24+16]. At least 119 telemetry rows, exactly
11 periodic checkpoints and 13 total replay audits (initial+11+final) for this
frozen 7200/600 contract. Positive integer peak RSS. Do not claim stronger runtime
properties than this summary can establish; later coordinator sealer separately
checks telemetry files, checkpoint fingerprints and source-bound provenance.
Read the contract and tests. Do not add dependencies or modify tests. Do not invoke
other agents or reviewers. Checks run with stdlib unittest and PYTHONPATH=python.
