Implemented only:

- [reference.py](/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261010-215113-33f71b3f/workspace/python/pokemon_binary/reference.py)
- [test_reference.py](/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261010-215113-33f71b3f/workspace/tests/binary-policy/test_reference.py)

Includes exact ratio-based dyadic thresholds, strict validation, clipped scalar policy, analytic derivatives, and 26 independent tests.

Limitation: [oracle-task.md:5](/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261010-215113-33f71b3f/workspace/artifacts/reviews/m1-binary-policy/oracle-task.md:5) does not prescribe API names or the `rare_index` convention for one-legal-action states. I used explicit public names and return the sole legal action as that index. Coordinator integration tests should verify those interface assumptions.

No external research was required. Controller acceptance remains outstanding.