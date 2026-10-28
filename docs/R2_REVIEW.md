# R2 implementation review and checks

2026-10-10. R2 is implemented; later sprint tickets remain open.

## Independent review

The delegated semantics audit received opposite-model QA, followed by pinned-source
inspection and two executed probes. Its incomplete specification was rejected;
the [audit disposition](R2_SEMANTICS_AUDIT.md) records exactly what was retained.

The completed Fable implementation snapshot review is at
`local-review-archive/20261010-115758-6d1f5398/review.md`.
It was a static review, not an executed test or blanket approval.

* Its medium-severity Revival Blessing finding was false for the pin. Both source
  branches explicitly permit the active slot under revival; an executed upstream
  AI probe confirms the same candidate list. The constructed fixture remains
  labeled, without a claim that the state is naturally reachable.
* Its phase-consistency finding was addressed: conversion rejects an R1 decision
  label that disagrees with the request. A regression rehashes the altered R1
  envelope and proves rejection, so this does not rely only on hash tamper checks.
* Its duplicate-decision concern was addressed with schema uniqueness. Foreign
  reference resolution and dataset closure remain collection-level work; scalar
  validation does not claim to prove them. Team IDs can repeat in mirror matches.
  Remaining-Pokémon counts are not used as rewards or as an alternative win rule.
* The numeric-token scanner now uses a sticky regular expression rather than
  repeatedly slicing the remaining input. The canonical vectors verify behavior.

Additional coordinator check found JSON Schema's `$` anchor can match before a
trailing newline. ID/seed patterns now assert absolute end-of-input in both regex
engines, with shared seed-rejection cases. Node's turn-record check is also strict.

## Executed verification

`npm test` passed **32 Node tests and 7 Python tests**, covering the R1 regression
suite, Node R2 contract tests, independent Python comparisons and the Python
unittest suite. Coverage includes:

* shared artifact examples and exact canonical hashes, Unicode scalar ordering,
  escaped controls, negative zero, safe-integer bounds, duplicate keys, malformed
  UTF-8/surrogates, unsupported schema versions and excessive nesting;
* tagged seed rejection, malformed action rejection, all 14 policy indices,
  known numeric zero versus unknown, and unknown versus known-empty item IDs;
* captured normal/switch/re-request cases and labeled constructed wait/pass,
  Struggle/recharge/revival cases;
* live Tera execution, repeated/stale submission rejection, forbidden switches,
  switch-order changes and hidden-disabled direct Struggle resolution;
* actor-loader rejection of privileged artifacts, wrong manifest/viewer,
  inconsistent masks and mismatched private-history boundaries.

`npm run contract:smoke` verifies ten seeded episodes, 823 decisions, 359 turns,
two natural hidden-trapping rejections, and 1,677 artifacts. Every semantic command
matches R1's visible choices. Python independently validates and reserializes the
entire JSONL stream; its bytes match Node exactly. The current source hashes,
runtime versions, byte count and evidence location are in
[the machine-readable report](../artifacts/reports/r2-contract.json).

The implementation is local and CPU-only. No corpus-scale generation, RL training,
search, Prime Intellect run, or second simulator accuracy claim is included.
R4 owns full phase/transition coverage, R5 owns structured actor features and broad
information invariance, and R7 owns dataset partitions and collection closure.
