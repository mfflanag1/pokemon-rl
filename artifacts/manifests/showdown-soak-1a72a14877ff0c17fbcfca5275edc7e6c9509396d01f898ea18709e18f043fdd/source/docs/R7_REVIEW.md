# R7 review and verification

The authorized `claude-review` workflow reviewed a frozen packet in run
`20261010-165137-63f2484b` (Fable 5.1). It returned findings with no high-severity
issue in the declared scope. This was a static review; the reviewer did not run
tests or inspect files outside the supplied packet.

## Findings and independently checked dispositions

1. **Producer drift at publication.** The collector now re-runs
   `episodeManifest()` and compares the complete runtime manifest before
   publishing its index, in addition to its R7 producer hashes. Inspection of
   existing R4 session/manifest code also confirmed that each worker already
   checks its loaded code against the supplied reference manifest at episode
   startup. A mid-run edit/revert does not hot-reload code into that process;
   the review's suggested scenario was not a demonstrated accepted mismatch.
   The extra end check closes the visible publication-time gap.
2. **Unknown Node mechanical fields.** Inspection confirmed R2 uses
   `additionalProperties: false` on generated Pokemon, IDs and EV/IV objects.
   The added Node `nature` negative test proves rejection; Python's independent
   test covers the same case. Unknown fields are not silently omitted.
3. **Battle/policy seed pre-check.** Before launching workers, the collector now
   checks every planned mechanics/policy seed against exposures and accepted
   team seeds, and rejects role/group reuse. Mirrored reuse in one group remains
   allowed. Python independently checks the same boundary during admission.
4. **Policy range assumptions.** Node now validates supported versions, pin,
   denominator and exact ordered contiguous ranges at import, deeply freezes the
   policy and checks both bounds. Exact boundary tests and 100 Node/Python
   assignment comparisons pass.

These fixes postdate the frozen review packet; their executed tests are separate
evidence, not a claim that the reviewer approved unreviewed changes.

## Bounded identity delegation

Fable implementation run `20261010-164058-372b5b33` owned only
`dataset_identity.py`. Codex QA `59d4686301` found that the coordinator's supplied
acceptance test reversed Ditto's one-element move list. That mutation cannot
change identity. Fixture inspection confirmed the problem. The local test was
corrected to reverse a multi-move Pokemon's list; all five tests passed against
the unchanged isolated candidate before integration, and `git apply --check`
passed. Later Node/Python and threshold tests verify the integrated module.

The original run remains **unverified against its defective acceptance test** in
the routing ledger. Its failed check was not rewritten as success or attributed
to a model implementation bug. The record also corrects an early, unverified
speculation about command quoting: the actual failure was the no-op test.

## Runtime preflight

The first four-episode canary failed before any episode completed because forked
workers inherited the parent's `--input-type=module` stdin flag. Its two worker
failure records and incomplete output are retained. Workers now receive a clean
`execArgv`; the same canary namespace then completed all four episodes and
exported 350 actor rows after independent validation. Neither canary is the
accepted benchmark. The final campaign's failure count is reported separately.

## Integrity and seal limits

Final holdout access is a workflow/API restriction plus content hashes, not
filesystem encryption or protection against the owner editing code. Admission
requires a separately trusted dataset hash. Tests deliberately reseal corrupted
metadata to check semantic rejection, while an unexpected root hash rejects
before loading actor streams. Exact teams are held out, with team/move order and
resolved form retained; this is not species-, set-, unordered-team- or
mechanical-equivalence generalization.

The reserved final teams have no episodes or actor exports. Automated identity
and schema checks may inspect their bytes without exposing team contents or
outcomes in the report. Do not unlock them for development or alter the split
salt after observing results.

Actual final checks, timings, counts, artifact hashes and source snapshots are
recorded in `artifacts/reports/r7-sprint.json` and [R7_SPRINT.md](R7_SPRINT.md).
