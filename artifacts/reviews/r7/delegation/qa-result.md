Verdict: **revise**

### Finding

- **Blocking:** supplied identity tests fail. In `test_each_mechanical_field_and_order_changes_identity`, this edit is a no-op:

  ```python
  lambda p: p['ids'][0]['moves'].reverse()
  ```

  `ids[0]['moves']` is `["transform"]`, so reversing it cannot change the canonical team or `team_key`. The reported equal digest is therefore correct.

  No pure, specification-compliant implementation can make this particular edit change identity without incorporating excluded metadata, mutating input, or introducing nondeterminism.

### Assessment

The implementation otherwise matches the stated contract:

- Exact payload and nested field sets are enforced.
- Canonical identity includes the requested mechanical fields and preserves ordering.
- Excluded metadata does not leak into identity.
- Nested output containers are independently owned.
- Policy versions, pin, profile, denominator, ranges, and salt are validated.
- Partitioning uses the full 256-bit digest and exact integer arithmetic.
- Cross-partition pairs are rejected.
- Only the owned module is changed in the supplied patch.

The supplied test should reverse moves on a Pokémon with multiple moves, such as `ids[1]`, before this candidate can satisfy the explicit “all supplied tests pass” criterion. Changing only `dataset_identity.py` cannot legitimately resolve the current failure.