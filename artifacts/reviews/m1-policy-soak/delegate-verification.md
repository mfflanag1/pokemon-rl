# Verified repaired gate delivery

The initial Fable run 20261011-075123-7e354bba passed its original three frozen tests,
but Codex QA found two malformed-input exception defects. Both were reproduced and
that delivery was recorded rejected. It is retained under delegate-initial.

The bounded repair run 20261011-075637-e175ffb1 completed in 54.02 seconds. It changes
only gates.py: overflow is translated to ValueError with safe error rendering, and
mixed invalid trace keys are rendered before sorting. Codex opposite-model QA
accepted the repair with no findings. The coordinator checked all four live input
hashes before git apply --check/application, inspected the complete patch, and ran
the six local soak tests, including the frozen huge-integer and mixed-key probes.
Valid-input numerical, compilation, scope and counter gates remain unchanged.

The gate operates on JSON-decoded dictionaries. It checks summary consistency;
telemetry, source/provenance and persisted original-next-state fingerprint binding
are separately verified by the coordinator sealer. No claim is made that a model
review executed tests. No candidate source was edited after this QA. Later
coordinator changes to the qualification plan describe paired wall/monotonic clock
checks; they do not alter the gate inputs/bounds accepted by this repaired module.
