# R7 identity delegation verification

Fable run 20261010-164058-372b5b33 implemented only dataset_identity.py.
Opposite-model Codex QA 59d4686301 correctly identified a defective acceptance
test supplied by the coordinator: reversing Ditto's one-element Transform list
is a no-op. Inspection of the supplied fixture confirmed this. The original
controller check executed normally; there was no quoting error. An earlier
tool-call justification speculated about quoting before reading the log and
was incorrect. No result was accepted on that speculation.

The original run's checks remain failed; the original packet is not rewritten.
It is recorded unverified against its impossible acceptance criterion, not as
a model implementation defect. The local test now reverses the second Pokemon's
multi-move list. All five corrected tests passed against the unchanged isolated
candidate before integration (identity-checked-tests.log). The coordinator read
the code, verified full-digest integer comparisons, explicit field sets and
fresh nested containers. git apply --check passed; only the candidate module
was integrated. Node/Python parity and the broader dataset gates receive separate
integration tests. This record distinguishes the original delivery outcome from
the independently checked code retained locally.
