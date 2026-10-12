The first Fable gate candidate passed its three frozen coordinator tests and changed
only its owned module. The coordinator verified all five original file hashes,
checked/applied the patch unchanged, and ran all six local qualification tests.
Codex opposite-model QA found two malformed-input cases violating the promised
ValueError interface. Both were reproduced locally: math.isfinite(10**10000)
raises OverflowError, and sorting mixed invalid trace keys raises TypeError.
They fail closed but do not meet the explicit interface criterion. This delivery
is rejected pending a bounded repair, not counted as accepted. The mapping-object
comment is not a requirement: the interface accepts JSON-decoded dict inputs.
No production two-hour campaign has begun.
