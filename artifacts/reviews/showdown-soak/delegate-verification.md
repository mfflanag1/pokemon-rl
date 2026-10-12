# Scheduler delegation verification

Initial Fable run `20261011-145210-26ef7028` completed with 22/23 tests passing.
Coordinator source inspection and opposite-model Codex QA both verified that
negative zero survived in `gameSpec.index`, failing its explicit strict-zero test.
The initial candidate was rejected and never integrated into production paths.

Fresh repair `20261011-145907-d9a18210` changes only negative-zero normalization
and its explanatory comment. The test snapshot is byte-identical. Codex QA accepts
the repair based on the supplied complete patch and controller checks, explicitly
without independent execution in its empty QA workspace. The coordinator checked
all selected-input hashes, patch ownership and `git apply --check`, integrated only
the two owned output files, and executed all 23 tests successfully. The repaired
run was adjudicated accepted; both outcomes remain in the quality ledger.

The final plan subsequently replaces the timing-based opponent delay with explicit
action hold/release and clarifies exhausted reconnect headroom. Scheduler source
and test files remain unchanged; the same 23 tests also pass against the final
plan. The full source-bound regression separately binds that final input.

Detailed requests, checks, patches, QA reports and coordinator verification are
preserved in `delegate-initial/` and `delegate-repair/`. An initial unsupported
`qa --detach` command was corrected; it did not produce or count as a QA result.
