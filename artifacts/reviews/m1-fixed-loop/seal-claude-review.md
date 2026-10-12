**Provisional verdict: findings (no blocker to the design; several acceptance gaps that let the seal under-enforce its own stated guarantees).** Most important: the sealer does not require the sealed run to be one of the archived `m1-fixed-loop-v*` directories, nor that the interrupted v1 exists and is recorded as non-passing — so "archive both v1 and v2" is not actually enforced.

## Findings

**1. Medium — v1/v2 archival not enforced (`seal-fixed-loop.py:31,88-89,91`).** `run` is only required to be under `artifacts/runs`; evidence is `glob('m1-fixed-loop-v*')`. A run named anything else is sealed but never archived, and nothing requires a `v1` directory to exist or records its `report.json` status. Fix: `require(run in runs)`, `require(len(runs)>=2)`, and add `{'predecessor_runs': {name: status/absent}}` to the report with `require(status!='passed')` for non-sealed siblings.

**2. Medium — test log unbound to sealed sources (`:77-81`).** `npm-test.log` is accepted from `artifacts/reviews/...` with no link to the hashed `tests/`, `python/`, `packages/` files or to the campaign. A log produced before a later source edit passes. The hard-coded `11` suites / 63 node tests cannot be verified from the packet. Fix: have `scripts/test.mjs` print a manifest hash of the test/source tree and require equality with the sealer's `source_files` digest; or at minimum record the log's mtime and require it exceed every source mtime.

**3. Low/Medium — checkpoint linkage underspecified (`:60-69`).** `load()` validates internal invariants only. The sealer never checks: `initial.seed==final.seed==p['soak_seed']`; each boundary has the soak seed and `initial.version < boundary.version <= final.version`, strictly increasing; `final.executed_moves-initial.executed_moves==soak['executed_moves']`; `final.decisions==soak['final_counters']['decisions']`; `initial counters==soak['initial_counters']`; parity checkpoint `.seed==seed`. A boundary file copied from a parity run would currently pass. All are one-line additions.

**4. Low — parity completeness not asserted (`:44-45`).** Only `comparisons>0`. Should require the exact key set (34 keys as in `parity.json`) and counts (24 per tensor/leaf; 192 for `update/*`; 360 for `fused_vs_separate`). I recomputed 24×15 leaves = 360 and 24×543 = 13032 elements against `parity.json` — consistent.

**5. Low — hardcoded scope field (`:92`).** `'final_reservation_unchanged':True` should be taken from `admitted['prior_gates']['final_reservation_unchanged']` (currently `true` in `admission.json`) and `require`d, not asserted.

**6. Low — telemetry/soak cross-checks.** Add `telemetry[-1]['decisions']<=soak['decisions']` and `telemetry[-1]['updates']==final.learner.version` (or ≤).

What the sealer does get right: content-hash linkage of `report.json`→`parity/profile/soak.json`; `status=='passed'` plus `elapsed>=target==7200`; positive/negative compilation control; re-running `admission()` at seal time (catches drift in `pokemon_fixed`, contracts, sealed predecessors); `write_once` protects old receipts; profile budget/accounting checks pass against the supplied `profile.json` (400804 decisions, 351613 moves within bounds).

## Scope limitations
Cannot verify: `digest/wire/file_hash/require` semantics, `Comparisons.check` tolerance enforcement, existence/contents of v1 directory and the three review `.md` files, actual number of Python test modules (11), plan file hash, or float-equality robustness of `decisions_per_cycle_second` after JSON round-trip. `report.json`, `soak.json`, telemetry and checkpoints do not yet exist; checks on them are reviewed against `campaign.py`'s writer only.

## Verdict
Findings. No false-accept of an incomplete soak is evident, but items 1–3 should be fixed before the seal is run on the completed v2.
