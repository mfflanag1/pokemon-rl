# Verified gate integration

Fresh run 20261011-183417-6d000447 used Codex after the auto router reassessed recorded outcomes; Fable provided opposite-family QA and accepted the corrected candidate. The coordinator compared the complete candidate to the preserved first version: only the two frozen plan/smoke predicates change from _is_real to _is_int. All four request input hashes match the live selected files. git apply --check passed and only the owned new gates.py was integrated.

Controller checks executed six tests successfully. The implementer itself reported no manual test execution; this is consistent with controller-owned checks, not independent execution by Fable. Local coordinator execution passes all 17 focused tests (gate, checkpoint, audit). An additional 36 direct mutations across every frozen plan/smoke numeric field reject equal floats, booleans and strings. Exact restoration, unchanged sealed-kernel identity, injected discard, interrupted writes, clocks and the missing-final-tick bracket are executed tests. These checks, source inspection and opposite-family QA support acceptance; reviewer opinion alone does not.

Initial rejected candidate and its QA/outcome remain preserved. Full source-bound regression and real smoke evidence are separate final admission requirements.
