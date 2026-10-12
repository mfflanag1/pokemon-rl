**Provisional verdict: findings (none blocking in reviewed scope); most important: the 1152‑start / 1008 / 864 / 576 invariance evidence is hash bookkeeping over Node‑produced hashes, not Python recomputation.**

## Findings

**1. Medium — counterfactual invariance is not independently reproduced** (`python/pokemon_switch/audit.py:59–69`, `94–97`)
`validate` groups `c['initial_hash']`, `c['after_hash']`, `c['opponent_own_hash']` taken verbatim from the fixture and checks equality/inequality within groups. Only the 48 probes (`audit.py:71–79`) are rebuilt through `check_view`/`actor`. The report's `hidden_initial_equal:1008`, `hidden_post_equal:864`, `own_variant_positive_controls:576`, `different_reserve_reveals:24` are literal constants (`audit.py:96–97`), not counted. README line 40–41 ("Python must independently reproduce … reconstruct structured player inputs from full player-only snapshots") is therefore true for the 1826 golden views but not for the comparison roster. Verification: either store the comparison views (or at least the `initial`/`after` views for a sampled subset) and recompute `actor(...).feature_id` in Python, or word the claim as "Node-hashed; Python-verified structure only".

**2. Low — coverage counters are unaudited** (`audit.py`, `campaign.py:25`, `seal-switch-reference.py:26`)
`coverage` (critical_hits 31, speed_tie_turns 138, low_hp_ability_attacks 85, …) is copied from `summary.json` and only checked for equality with itself. No Python code counts `|-crit|`, speed-tie or ≤1/3‑HP attacks from the protocol, and no minimums are required. The sealer would accept a run with `critical_hits:0`. Treat these numbers as producer-reported.

**3. Low — opponent-HP leak check is narrower than the reducer** (`audit.py:17–19`)
Only `switch/-damage/-heal` lines are checked; `drag`, `replace`, `-sethp`, `detailschange` are not. In practice `PublicHistory._hp` (`public_history.py:40`, `percent=True` for non-viewer) rejects a non‑/100 denominator for those too, so coverage is adequate, but the test at `test_contract.py:62–65` only exercises the `switch` path. A `-damage|p2a:…|58/119` negative test would pin the reducer path.

**4. Low — roster/manifest composition unchecked** (`audit.py:26`)
Only `len(roster)==24` is required; distinct species, A/B names and profile variants per roster entry are not verified in Python (they are enforced indirectly by `mask` on each request, so a bad roster would fail downstream, but not at the manifest).

## Codec equivalence (Node ↔ Python)
Compared `actions.mjs:6–41` with `contract.py:22–66` field by field: identical whitelists (request/side/mon/move keys), identical wait/forceSwitch/active branching, same bool-rejection (`type(x) is int` vs `Number.isSafeInteger`), same `[True]` strictness (Python adds explicit bool check to defeat `1==True`), same A/B lookup by ident and 1‑based switch position. No divergence found. `test_contract.py:18–19` correctly infers `switch 2` from mask `[T,T,T,F]`.

## Player boundary
`views.py:26–46` enforces exact key set, rejects foreign-side requests, requires request count/turn/request to match the own channel. `Session.stepOnce:52` checks the pending choice doesn't leak into the other view. Audit fields never enter `actor`. OK as far as visible.

## Scope limitations
`PlayerProjection` (the actual channel splitter), `pokemon_reference.contract`, `dataset_gate`, `profile.mjs`, `fixtures.mjs`, the generator's seeding of comparisons, and the fixture bundle itself are not in the packet; test/run outcomes are as reported, not verified. The running full audit was not observed.

## Verdict
Findings — no correctness defects in the reviewed code; one overclaim risk (Finding 1) should be fixed or reworded before sealing.
