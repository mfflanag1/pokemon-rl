# Failed delegation disposition

Run 20261011-111535-930b13e8 produced the two requested candidate files, but the
controller rejected ownership before executing checks. The coordinator incorrectly
repeated --owned; argparse kept only tests/switch-reference/test_contract.py in
request.json. The emitted Python file was therefore outside controller ownership.
This is a coordinator launch error, not evidence of a model correctness failure.
The two diff paths match the intended task. No candidate code was integrated.

Opposite-model QA was attempted and returned: "QA requires a completed result".
No tests or QA can be claimed for this delivery. The implementation was completed
locally as a smaller independent two-function codec, covered by live Node/Python
request comparisons and a separate Fable review. That work is not acceptance of
this candidate. Record this run as unverified; preserve the failed evidence.

The controller also rejected adjudication with "Only completed runs can be
adjudicated". Its persisted failed status is the authoritative recorded outcome;
this local disposition is retained as the explanation, not a successful ledger
adjudication. No retry or candidate integration was performed.
