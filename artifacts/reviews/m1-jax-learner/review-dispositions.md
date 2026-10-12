# Independent review dispositions

Used `/Users/maxf/.codex/skills/claude-review/SKILL.md`.

## Numerical core delegation

Fable run `20261010-180004-f1e0eebe` implemented only `core.py`. Its five declared
controller tests passed, opposite-model Codex QA accepted, and input hashes plus
patch ownership were verified before integration. Further local parity/boundary/
finite-difference tests and the original-budget canaries pass. The core is unchanged
from the candidate; its accepted outcome and actual evidence are preserved in
`delegate-verification.md` and `delegate/`.

## Separate Fable pipeline review

Run `20261010-180934-173c1790` completed: no blocking findings, low-severity evidence
and test-label issues. Original source hashes and packet are retained. It did not
execute tests; all executed evidence below comes from local checks/campaigns.

1. Resume wording: original campaign made one load and compared seven subsequent
   updates, as the initial handoff explained. To remove ambiguity and strengthen
   evidence, it now loads the preceding boundary freshly from disk before each
   compared update. Seven real reloads with following updates per seed; 21 total.
2. Report assertions: terminal-score sums are now measured from buffer rewards;
   actual actor field names are exported; the observation-only summary is derived
   from per-buffer schema, legal-row and zero live terminal/winner-indicator checks.
   These audit fields supplement the source/interface and noninterference tests;
   they do not claim to prove safety against malicious custom actor code.
3. Crash hook: verified that the old `crash_before_linkage` name was inaccurate.
   Renamed it `crash_after_collection`: it exercises failure before trainer buffer
   publication after collection work has completed. No mid-turn failure test is
   claimed. Partial optimizer failure and interrupted atomic publication are tested.
4. Chance namespace: confirmed distinct folded environment master. Documented the
   three domain tags and that plain G2 `chance.sample(seed, ...)` is a different
   sequence, while the same semantic sampler kernel is reused. No raw-stream or
   bit-exact Showdown mapping claim is added.

Coordinator hardening also binds the original canary plan and contract requirements
into admission/provenance, records checker dependency versions and numerical compiler
settings, and preserves all prior sealed simulator/NumPy files. Post-review changes
are localized reporting/admission/checkpoint-test improvements, covered by final
regression and a fresh fixed-budget campaign. Seeds, tolerances, learning budgets,
losses, model and chance sampler are unchanged. Full G2 and G3 remain open.
