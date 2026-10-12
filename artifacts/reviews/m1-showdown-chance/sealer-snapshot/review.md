**Provisional verdict: no blocking findings in reviewed scope.** The sealer binds the campaign report (content hash + admission hash + live `admission()` re-evaluation), the 100 capture shards (first/sha256/count against the sealed index), per-shard accounting that sums exactly to the report, the three checkpoint files (sha256 + load-time provenance/counter checks), the source-bound regression log (manifest equality + log hash), and copies all of it into a hash-named write-once archive. Scope flags (`full_g2/g3 open`, no two-hour, no Prime, reservation unchanged) are enforced, so an altered report cannot be promoted without tripping `validate_scope` or the content hash.

Arithmetic in the snapshot is consistent: shard sums give 10,000 / 31,371 / 55,062 / 180,000 / 62,742; `pending == 2·turns`, `crit == roll == events`, `tie = 31,252 ≤ 2·turns`; suite counts sum to 164 (+65 Node = 229); live comparisons `2·(10,000 + 2·31,371) = 145,484` matches the doc.

## Findings (none blocking)

**Low — `campaign_history` entries are not sealed/status-checked** (`scripts/seal-showdown-chance.py:98`). Predecessor run directories matched by `m1-showdown-chance-v*` have only `content_hash` read; neither `sealed()` nor `status=='passed'` is applied. A failed or hand-edited historical `report.json` would be recorded as history without complaint. Not a false promotion (status is not asserted), but the receipt presents these hashes as campaign history. Verification: add `sealed(...)` to that comprehension and confirm with a v0 directory containing a tampered report.

**Low — the "additional audit" is restore-vs-restore, not an independent check** (`:71–75`). Both runners are loaded from the same bytes with the same `Executables` in the same process, so `state(a)==state(b)` is near-tautological. Original-vs-restore equality rests entirely on the campaign (`campaign.py:164–166`), which the task acknowledges. Suggest not describing this as an independent replay audit in the receipt text.

**Low — review artifacts are presence-only** (`:84–85`). `claude-review.md`, `review-dispositions.md`, `delegate-verification.md`, `delegate/outcome.json` are hashed into evidence but no field (e.g., `outcome.json` status) is validated. For a trusted-source receipt this is acceptable; a one-line status check would close the gap cheaply.

**Info — tie-draw structure not bounded.** `validate_player` only checks `0 < tie ≤ 2·turns`. Since the doc states two shuffles per tie turn, `tie` should be even and equal `2·(tie turns)`; the per-turn structure is enforced in `verify_case`, not at seal. Observed 31,252 is even and plausible (~half of 31,371 turns × 2).

## Scope limitations
- Not inspected: `pokemon_learner.checkpoint.{digest,wire,file_hash}`, `pokemon_reference.dataset_gate.checked_file`, `pokemon_jax.campaign.parity`, `scripts/test.mjs`, the review artifacts, capture `plan.json`/`index.json`, and the verification JSON/log. Claims about their content rely on how the sealer consumes them.
- No tests executed; the 229-test result and verify-fixed-loop binding are taken from the packet.
- Source files outside `python/`, `contracts/`, the pinned package/vendor files, and `provenance()` paths (e.g., unpinned `packages/`, `tests/`) are bound to the regression log but not to the campaign run; this appears harmless because the campaign is Python-side except `describe()`, but I could not confirm `parity()` does not invoke Node.

## Verdict
**No blocking findings in reviewed scope.** Three low-severity hardening items and one informational note; none permits altered counts, missing shards, checkpoint substitution, source drift, or scope promotion to pass the sealer as written.
