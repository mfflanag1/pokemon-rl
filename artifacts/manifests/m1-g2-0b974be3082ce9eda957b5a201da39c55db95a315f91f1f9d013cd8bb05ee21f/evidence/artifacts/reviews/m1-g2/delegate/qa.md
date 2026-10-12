Verdict: **revise**

Concrete findings:

- `python/pokemon_jax/views.py:286-322` does not enforce the immutable profile. It validates HP and move IDs, but accepts altered species/details, ability, item, nature, level, gender, IVs/EVs, Pokémon move list, or Tera type in the own request. The contract explicitly requires rejecting profile perturbations.
- Unsupported request mechanics are rejected only when truthy. `teamPreview=False`, `forceSwitch=False`, `wait=False`, or `canTerastallize=False` are accepted, despite the strict contract requiring those mechanics to be absent.
- Embedded `|request|` objects without `side` are accepted. The requirement to check embedded request side identity implies a present, valid `side.id`; otherwise side isolation is not established.
- The supplied test command did not pass—it failed during import with `ModuleNotFoundError`. That may be a test-environment path issue rather than reducer logic, but the criterion “supplied tests pass” remains unverified.
- The candidate’s statement that implementation is complete and no further reads are needed is unsupported. Its own limitations acknowledge untested behavior, while the strict profile-admission requirement was missed.
- Because the review workspace contains no repository or candidate files beyond the supplied evidence, I cannot independently verify that only the owned file changed.

The core protocol replay, relative-seat mapping, KO-capped HP loss, executed-only reveal, terminal stale-request handling, and output shape appear substantially correct. The missing immutable-profile admission checks are material but localized, so **revise**, not reject.