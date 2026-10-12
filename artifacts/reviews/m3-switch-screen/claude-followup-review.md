**Provisional verdict: no material replay/provenance findings in reviewed scope; two items need out-of-packet verification.**

## Findings

**1. (Medium – verification needed) Freeze/admission binding depends on `source_manifest()` covering the new screen implementation.**
`provenance.py:33` builds `source_files` via `pokemon_switch.provenance.source_manifest()` (predecessor module, `CURRENT` registry). The comment at line 27 implies the registry is additive, but the packet does not show whether `packages/switch-screen/audit.mjs`, `worker.mjs`, `python/pokemon_switch_screen/*`, and `contracts/switch-screen-v1/*` are registered. If not, `f['admission']==a` (`campaign.py:180`, `seal:35`) and `v['source_files']==a['source_files']` (`seal:69`) bind the predecessor sources but not the replay auditor or inference code, which would make "source-bound freeze" and "regression bound to sources" claims hollow for exactly the new files.
*Verify:* confirm every path above appears in `admission()['source_files']` keys (e.g. dump `admission()` and diff against `git ls-files packages/switch-screen python/pokemon_switch_screen contracts/switch-screen-v1`).

**2. (Low – verification needed) Cross-language hash equality is assumed, not asserted in-packet.**
Stored `view_hashes` are produced by Python `digest()` (`inference.py:80`), checked by Python `digest()` (`campaign.py:95`) and by Node `hashJSON` (`audit.mjs:18`). Replay correctness relies on `battle-contract/canonical.mjs` and `pokemon_learner.checkpoint.digest` sharing canonical JSON (key ordering, float/int rendering, unicode). If they diverge, `audit.mjs` fails closed (good), but a passing seal has no independent in-packet test that the Node side is actually comparing the right thing rather than a coincidental format. *Verify:* existence of a cross-language canonical-hash parity test in the suite (`SUITES`/`node==[237,3]`).

**3. (Low) `frozen_at < started_at` is a lexical string comparison (`campaign.py:180`, `seal:35`).**
Sound only if the freeze writer uses the same `utc()` format as the run. The freeze script is not in the packet. *Verify:* both strings are same-format ISO-8601 UTC with identical precision.

**Checked and found consistent (no finding):**
- Step ordering parity: worker uses `(id+group)%2`, auditor uses `(2*pair+seat+group)%2`; `id=2*pair+seat` per `evaluate_cell:57-59`. Matches.
- Inference batch alignment: `evaluate_cell` batches ids in 32-chunks in id order; `audit_cell.consume` re-batches by record count in 32-chunks, selecting records with `len(fs)>group` — identical membership/order to the live set at collection time. Per-row RNG addresses (`inference.py:64`) are row-position-independent, so re-batching cannot change actions.
- Horizon: `learner.GROUPS=14` loop plus `require(all done)` at `campaign.py:69`; 12 joint turns + ≤2 replacement groups is the correct bound for 2-member teams.
- Tie: `Session.frame` maps `''→null`; `verify_record:86` then requires `winner in ('p1','p2')` → fails closed as documented.
- Greedy: `own[1]` is own exact HP (ability-threshold input only), `23·noncrit+crit` over 16 rolls is proportional to the 1/24 crit expectation; tie-break `-m` picks lowest index.
- `audit.mjs` alone does not check index completeness/sha256/record order, but `seal:50-59` establishes these before invoking it, so the composed claim holds.

## Scope limitations
Not in packet: `source_manifest`/`CURRENT`, `canonical.mjs`, `pokemon_learner.checkpoint.digest`, `views.encode`, `profile.reference()`, the freeze-writing script, test suite contents, `checkpoint.py`. No tests were run; no artifacts inspected.

## Verdict
No findings of material replay or provenance defects in reviewed scope; items 1–3 are unverified dependencies that should be confirmed before freeze, with item 1 being the one that could materially weaken the provenance claim.
