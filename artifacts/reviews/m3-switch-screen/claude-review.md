**Provisional verdict: findings (none blocking on the evidence in the packet; one medium item needs verification before freeze).**

Most important: the damage‑greedy opponent passes the receiver's current HP (`own[1]`) as the third argument to `damage(...)` and passes no attack stat at all, so the README's "expected immediate damage … using own exact HP and stats" may not be what is implemented.

## Findings

**1. Medium — greedy opponent argument may be mis‑specified** (`inference.py:31‑32`)
`views.encode` lays out own rows as `[species+1, hp, maxhp, atk, def, spa, spd, spe, active]` (`views.py:45`). `greedy` calls `damage(SPECIES[own], SPECIES[other], own[1], m, crit, r)` — i.e. passes current HP and never the attacker's `atk`/`spa` (indices 3/5). If `damage`'s third parameter is a stat (the roster encodes stat variants `65+32*v`, `campaign.py:101`), the baseline is computing damage from HP, not stats; the preregistered opponent definition would then be wrong, and every "damage‑greedy" cell (descriptive, but part of the frozen plan) is mis‑labelled. `test_greedy_forced_switch_and_public_type` only asserts `greedy(v)==0` for one team, which would not catch this. *Verify:* inspect `pokemon_switch_engine.damage.damage` signature; add a test where move choice flips with stat variant but not with HP.

**2. Low — hard‑coded 14‑group horizon vs sealed engine `max_groups`** (`campaign.py:61,89,121`; `learner.py:13`)
Evaluation and audit cap at 14 request groups while the engine uses `PLAN['max_groups']`. With 1 initial + 12 turn + up to 2 forced‑switch groups, 15 is reachable unless the engine caps otherwise. Mismatch fails closed (`'Incomplete capped game'`) but would abort a 31k‑game run late. *Verify:* assert `14 == learner.GROUPS` at import, or derive the loop bound from it.

**3. Low — `|tie|` is detected but has no retained outcome category** (`campaign.py:81‑88`)
A tie yields `winner is None, truncated False` → `'Terminal outcome'` raise → whole run fails. Likely unreachable with this moveset, but the contract ("every terminal win/loss and every nonterminal cutoff is retained") is silent on ties. *Verify:* confirm engine cannot produce simultaneous faint, or preregister tie handling.

**4. Low — preregistration freeze is equality‑bound, not time‑bound** (`seal-switch-screen.py:67`)
Seal only checks `REVIEW/preregistration.json == out/preregistration.json`; nothing proves the review copy existed before `main()` wrote the run copy. Consider hashing the review preregistration into the run's `report.json` at start.

**Statistics (checked, no defect):** radius `sqrt(log(family/α)/(2·pairs))` = δ=.05/6 per bound; secured‑win uses wins only (cutoffs count against candidate); worst‑case improvement = win − baseline(win∨cutoff) with 2× radius; all‑cutoff baseline cannot pass (`test_no_false_improvement_from_censoring`); per‑pair mean ∈[0,1]/[−1,1] so Hoeffding over 576 pairs is valid; gate applied per seed, unpooled; smoke forced to `passes_screen=False`. Seat swap keeps the candidate on team A in both seats; chance spec excludes `update`, giving the intended paired design.

**Hidden info / actions (checked):** actor rows use only the receiver's protocol; opponent HP must be `/100`; waits yield `None` value/logp; replay audit rebinds every view hash, action, logp and value; `|win|` lines linked to `winner`.

## Scope limitations
Not in packet: `damage`, `policy.act/select`, `views.actor`, `contract.mask`, `session.mjs`/`choices`, `BaseReference.finish/abort`, `SHAPES`, engine `max_groups`, `write_once`. Cannot confirm test counts (SUITES, `[237,2]`) or that any test ran.

## Verdict
Findings; nothing in the reviewed scope invalidates the censoring‑conservative gate, but item 1 must be resolved before the preregistration is frozen.


# Focused follow-up

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
