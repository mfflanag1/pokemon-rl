# Training curriculum v1 contract

`plan.json` fixes the 3,000-episode index schedule, IDs, admission budgets, source hashes, and exact expected counts. `catalog.json` fixes ten constructed, train-admitted six-member pairs and their scripted public-action prefixes. The catalog's canonical hash is checked by the module against the plan. The original R7 policy hash is checked independently; these files do not introduce a replacement partition salt.

Runtime API and semantics: `packages/training-curriculum/index.mjs`; integration, limitations and bounded verification: `docs/C1_TRAINING_CURRICULUM.md`.

These files are coordinator configuration, **not actor features**. No prefix action is a policy training sample. `evidence_requirements` are audit predicates over projected spectator events; no private-state mutation is permitted to satisfy them. All synthetic families/counterfactual groups remain train-only. Any change to scenario IDs, teams, prefix scripts, schedule, candidate budgets or group identities requires a new version and review before another qualification snapshot.
