Done. `python/pokemon_switch_engine/damage.py` now contains the candidate from `artifacts/reviews/m3-switch-engine/damage-candidate.py`, copied verbatim — zero changes, since nothing in the corrected tests required any.

**Why no changes were needed (manual verification, not test execution):**

- **Speed stats (corrected test, line 25):** swift Fennekin = `(2·60 + 31 + 252//4)·50//100 + 5 = 214·50//100 + 5 = 112` — the candidate's `stats()` produces 112, matching the corrected expectation. Swift Squirtle = `(86+31+63)·50//100+5 = 95`, Bulbasaur plain `[120,69,69,85,85,65]` — both match.
- **Fixture spot-check** (`damage.json` first block, Bulbasaur→Bulbasaur, hp 39, Overgrow active): Vine Whip rolls 0/4/12/13 → 11/10/10/9 and Tackle rolls 0/1/6 → 19/18/17 all reproduce exactly under the candidate's formula (6144/4096 stat boost → base formula → roll → STAB rounding → two floored halvings).
- **Strict arguments:** `_check_int` rejects bools, ranges enforced for `hp` (1..maxhp), `move` (0/1), `roll` (0..15); `_check_bool` rejects `1` for `crit`/`swift`; unknown species raise `ValueError`. All seven rejection cases in `test_rejects_bad_profile_arguments` map to a `TypeError` or `ValueError`.

**Limitations:**
- I did not run the test suite (no execution tool available in this task); the controller's acceptance run is the authoritative check. My verification covers the three `stats` assertions, `max_pp('fennekin')`, the rejection cases, and ~7 hand-computed fixture values out of 3456 — not the full grid.
- No files other than `damage.py` were touched; tests, fixture, contract, and `__init__.py` remain unchanged.
