# Delegated statistics verification

Original Fable run 20261011-131150-d03d0086 was rejected after its controller check failed 1 of 14 tests: duplicate-policy message mismatch. Codex QA confirmed this and asked for literal known-answer coverage. Original logs, packet, patch, QA and rejected outcome are retained.

Fresh Fable repair 20261011-132522-08e04db8 changes only that validation message and adds three literal fixtures, leaving the algorithm unchanged. Its controller check passes all 17 tests. Independent Codex QA accepts the repair and independently recomputes the fixed numerical fixture; the reviewer did not execute the repository suite.

Coordinator checks: both selected input hashes match before each integration; both patches pass git apply --check. The original implementation uses a complete edge/pair/mirror grid, exact 0/0.5/1 scores, pair averaging, common resampled pair indices across edges and the documented symmetric percentile convention. Reversed record orientation complements only score, not the mirror index; the campaign/auditor additionally require canonical matchup roles. I inspected the complete patch and controller checks; only the two owned paths are changed by the delegation. Local node --test tests/showdown-panel/statistics.test.mjs executes and passes all 17 tests after integrating the repair.

I independently computed the first 15 seed-0 uint32 words using Python unsigned arithmetic, yielding indices [2,1,1,0,0,2,2,2,0,0,2,0,1,0,2]. With pair scores [0,0.5,1], the five exact replicate means are [2/3,1/3,2/3,1/3,1/2], giving literal interval [1/3,2/3]. This fixture complements the existing differential oracle and degenerate-pair tests. The original failed candidate is not retroactively accepted; the repaired run alone is accepted.

Remaining scope: descriptive fixed development intervals only. No simultaneous coverage, hypothesis-test/promotion decision, or claim that a degenerate interval implies certain strength. No model review alone is treated as executed evidence.
