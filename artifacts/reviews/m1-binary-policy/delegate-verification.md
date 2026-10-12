# Rejected delivery: 20261010-215113-33f71b3f

Auto routing selected Codex for its fifth-run low-risk exploration. The run failed
ownership enforcement after generating Python bytecode outside its two owned files.
It also wrote pytest tests despite a unittest controller command and the no-new-
dependency requirement. Controller checks never ran. `cross-model qa --engine fable`
refused because QA requires a completed run. These are delivery failures, not a
successful independent implementation or an accepted routing outcome.

The coordinator inspected request hashes (all three original inputs unchanged),
read the source and patch, and isolated only reference.py. `git apply --check` and
integration of that source-only patch succeeded; no bytecode or pytest tests were
integrated. It uses Python ratios/Fraction and arbitrary precision concatenation,
independent from production's JAX bit shifts. Its source is now coordinator-owned
salvage pending local unittest checks and separate Fable snapshot QA together with
the production policy. This record does not certify that pending work. The failed
candidate, response and patch are preserved in delegate/ for audit.

The adjudication command also refused: "Only completed runs can be adjudicated".
The runner retains status=failed, which counts against delivery; no fabricated
accepted/rejected outcome was written into its internal records.
