**Provisional verdict: no blocking findings in reviewed scope; two low-severity reporting/claim-fidelity issues and one test-fidelity issue.**

Most important: the campaign's "seven exact resumed updates" per seed rests on a single checkpoint load (after update 0), with updates 2–7 compared against a continuing in-memory `replay` object, not fresh loads at each boundary.

## Findings

**1. Low — Resume claim overstated (campaign.py:146–152).** `replay=load(...)` happens only when `update==0`; later iterations call `replay.collect()/update()` on that same live object. So per seed there is one real disk-resume followed by seven in-process matched updates, plus a final `load` that only compares `state()` without further training (line 158). The code is correct, but "seven exact resumed updates" should read "one resume, seven subsequent matched updates." Verification: load `boundary.checkpoint.json` fresh before each of updates 1–7 (or save/load at every boundary) and compare `content_hash`/diagnostics.

**2. Low — Hardcoded report assertions (campaign.py:160, 163; 135).** `'terminal_score_pair_sum': trainer.games` is tautologically `trainer.games`, not a measurement; `'all_buffers_observation_only': True` and `'actor_has_targets': False` are constants. The underlying properties *are* enforced (`collector.py:116` requires `sum(rewards)==games`; `ActorTape` has no target fields; `encode` only accepts `PlayerView`), but the report fields do not derive from data. Suggest computing `sum(buffer.targets.rewards)` across the trace and listing `fields(ActorTape)` in the report.

**3. Low — `crash_before_linkage` misnamed (trainer.py:39).** The injected crash fires *after* `collect()` returns (linkage, GAE, hashing all complete) and after counters were advanced (line 38). The rollback path is still exercised correctly because `save` refuses non-idle phase and `load` replays from disk; but no mid-collection failure (e.g., between step and record append) is tested. Not a correctness bug; the test label is inaccurate.

**4. Info — Chance stream differs from `chance.sample` (collector.py:78 vs chance.py:56).** Collector folds `0x454e56` into the master key; `chance.sample(seed,...)` does not. Collector chance is deterministic and separated from the action domain (correct), but is not reproducible via the public `sample()` helper. Document or reconcile.

**Verified assumptions (positive):**
- Leakage: features are built only from `PlayerView` (header 11 + relativized public history); terminal/winner header columns are zero for recorded rows (rows filtered by `live`, test asserts `features[:,1:3]==0`); opponent PP hidden; actor is row-independent (linear, tested). Chance keys (`env_key`, roles 0–4) and action keys (`action_key`, roles 0–1) use distinct masters.
- Timing: both seats' views projected before either action is sampled (collector.py:85–89); dead games get mask all-False → action −1 → excluded.
- PPO: `where(active, ratio, stop_gradient(clipped))*A` equals `min(ratio·A, clipped·A)` in all four sign/clip cases and matches the reference `active` gradient rule; targets stop-gradient (tested).
- GAE: whole-episode, bootstrap 0, same (game,seat) forward links, `turn[j]==turn[i]+1`, terminal⇔`next==-1`; reverse scan reads only already-computed `advantages[j]` (j>i).
- Provenance: `buffer is self._buffer`, version, content_hash recompute, policy_hash, and behavior-ratio recomputation (≤1e-6) all enforced before updating.
- Checkpoint: step/m/v/params/counters/ledger serialized; `state(trainer)==b` roundtrip; shuffle keys derived from `(seed, version, epoch)` with no mutable cursor.
- No promotion claims; `full_g2`/`g3` marked open.

## Scope limitations
Not in packet: `pokemon_learner.checkpoint` (`digest`, `wire`, `read_array`), `pokemon_learner.trainer` (Canary base, `shuffle_rng`), `pokemon_slice.engine`, `canaries.json`, `dataset_gate`. Cannot confirm `read_array` dtype enforcement or `digest` canonicalization. No tests executed; reported pass counts (96/384/96, 3×512 games) unverified.

## Verdict
Findings — low severity only; numerical core, leakage boundaries, and buffer provenance appear sound. Tighten the resume claim wording and derive report flags from data.
