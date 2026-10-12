**Provisional verdict: findings (medium/low); no confirmed false-completion path in the reviewed code.** Most important: the sealer never binds the executed Showdown oracle (node_modules) or the passivity controls to anything it can re-verify.

## Findings

**1. Medium — Executed oracle code not bound by sealer** (`scripts/verify-fixed-loop.py:23`, `packages/switch-engine/reference.mjs:110,131`, `python/pokemon_switch_engine/parity.py:121`)
`source_manifest()` excludes `node_modules`, so the Showdown build actually run by `reference.mjs` is outside admission/`verification.json`. `capture()` records `reference()` into `plan.json` and only self-checks it at end (`reference()===manifest`). `parity.audit` compares `plan['plan']` to the contract but never compares `plan['reference']` to a pinned expectation; the sealer doesn't either. A modified installed Showdown passes seal unless the unseen `m3-switch-reference` receipt covers it.
*Verify:* confirm `m3-switch-reference.json` `source_files` include the Showdown build hashes, or add `compare(plan['reference'], <pinned manifest>)` in `audit`.

**2. Medium — Passivity ("unpatched") controls are count-only evidence** (`reference.mjs:121-123,132`; `parity.py:123`)
`checkUnpatched` leaves no artifact; `report['passivity']` is an integer the sealer only checks equals `min(count,24)`. Unlike `seal-showdown-soak.py:68` (fresh `node audit.mjs --check`), nothing re-runs the unpatched comparison at seal time. An edited `report.json` with `passivity:24` is indistinguishable. Also only 24/10000 games covered; the claim "actual unpatched passivity controls verified" overstates what the sealer checks.
*Verify:* run a Node `--check` of `checkUnpatched` over the 24 cases inside `seal()`.

**3. Low — Rollback audit is tautological** (`seal-switch-engine.py:71-75`; `campaign.py:46-51`; `checkpoint.py:92-103`)
`transaction` is purely functional; "Rollback mutation" compares the fingerprint of a local object that cannot be mutated. No on-disk/publish rollback (e.g. `save(interrupt=True)` leaving `.partial`, then `load` refusing it) is exercised. Counts toward `rollback_audits==6` without testing rollback.

**4. Low — No Node test coverage for new instrumentation** (`seal-switch-engine.py:95` vs `seal-showdown-soak.py:74`)
Node pass count is pinned at 237, identical to the predecessor; only one Python suite was added. `reference.mjs` hooks (shuffle/tie, crit, roll interception) have no regression test in the bound log.

**5. Low — Hard-coded 1152 in `information_checks`** (`parity.py:93`)
`hidden_initial_equal=1152-len(initial)` assumes the comparison count; `n` is never checked against 1152. A truncated fixture inflates the reported invariance count. Also the fixtures path (`parity.py:133`) is bound only via the unseen reference receipt.

**6. Low — Legacy specs unbound to original fixtures in Python** (`parity.py:125`)
`legacy.json.gz` specs/scripts are only asserted against `fixtures.json.gz` inside Node (`expected`); the Python audit re-checks scalar/JAX consistency but not that `legacy[i].spec == old.cases[i].spec`.

Checked and found consistent: receipt/archive hash chain, live-drift exemption via `CURRENT`, preregistration == seal admission, fixed budget (64×32, counters linkage), fingerprint chain from `learner.start(seed)`, checkpoint boundary at 0/32/64, fresh replay of six original-next-updates, metrics fingerprint round-trip, tie/crit/roll word mappings (`w>>>31`, `floor(w·24/2³²)`, `w>>>28` match float-exact Showdown semantics), `smoke is False` and 10000-game gating before training.

## Scope limitations
Not provided: `learner.py`, `scalar.py`, `engine.py`, `views.py`, `switch-reference/*`, `provenance.CURRENT`, the two predecessor receipts, `fixtures.json.gz`. Replay covers only 2 of 64 updates per seed; intermediate rows are chain-linked by fingerprint but not re-executed (by design). Learner math excluded per task.

## Verdict
Findings. No code path lets a run seal with wrong budget, non-original checkpoint replay, or source/runtime drift in the audited Python tree; gaps are in oracle-runtime binding (#1) and non-replayable passivity/rollback evidence (#2, #3).
