# Opponent panel review dispositions

Two Fable 5.1 snapshot reviews and two independent Codex statistics QA results are retained. Snapshot review is inspection only; reviewers did not execute engine or repository tests. The final files include the locally verified repairs listed below; there is no claim of a third snapshot approval.

## Public-view / policy review

1. Illusion memory contamination: confirmed. On reveal, restore the disguise identity memory from entry, and transfer only facts observed during this appearance to the revealed identity. This also fixes the second review finding about pre-existing Water Gun/Eviolite being wrongly attributed to Zoroark. Both a fresh disguise and one with previously known moves/items have dedicated passing tests.
2. Selective/copy/swap stat events: confirmed for reachable event shapes. Clear-negative/positive, copy and swap are supported. Actual pinned Psych Up/Power Swap output is tested, including engine-state agreement. Pinned moves.ts copies target boosts into the first emitted Pokemon for Psych Up; this contradicts SIM-PROTOCOL.md's wording. Unsupported Baton Pass/other callbacks and rare public-state interactions remain explicit approximation limits; no complete public-state/mechanics claim.
3. Entry hazards charged to the active: confirmed and fixed with an explicit entering flag. A known 1/8 Stealth Rock difference tests this.
4. Bench Fake Out/First Impression: confirmed and fixed by assigning the prospective entry turn. First-turn move scores have boundary tests.
5. Accuracy stages: coordinator corrected the combined 3-based stage multiplier and always-hit treatment. Known-ratio tests pass; second reviewer confirms by inspection.
6. Weather sentinel none is harmless for current comparisons; retained. Copied/replaced abilities such as Trace are not fully lifecycle modeled, and known ability memory can over-persist across switches. This and other callback/volatile limitations are documented rather than treated as exact mechanics.

## Evidence pipeline review

1. Source coverage: verified directly with source_manifest(); every new package, test and sealer is included. manifest-coverage.json records hashes and total count. Exact frozen source comparison also gates production sealing.
2. Final-team assertion: sealed developmentTeams recomputes R7 content partition before admitting a unique official team. Audit independently regenerates the exact pool and additionally recomputes each saved team's partition; any non-development team fails. Returned final count now derives from those checked partitions.
3. Score orientation: policy a occupies seat mirror; game.outcome.score[mirror] is therefore a's score for both mirrors. Complete grid and exact whole-game replay verify role pairing. Statistics tests independently check mirror-one interpretation, wins/draws/losses, pair means and malformed grids.
4. Failure/progress check: added explicit rejection of failure.json and any progress count other than 768/768. Tightened Node coverage to exactly 149 tests (114 predecessor + 35 new); expected Python suite sizes remain unchanged at 189.
5. Empty non-force-switch move list: request codec/engine normally provides moves (including Struggle). An unexpected empty actionable list throws and invalidates the whole run, never silently awarding a loss or replacing a pair.
6. Audit deliberately rejects corrupted actor inputs, seed, policy seats, score record, interval and team data. Initial corruption-test implementation treated the sodium seed string as an array; that test failure is retained and repaired without modifying the producer or relaxing rejection. Final targeted suite covers the actual seed representation.

## Delegation

Original statistics delivery failed its own error-message test and was rejected. A fresh bounded repair passed all 17 tests and opposite-model QA. Coordinator checked input hashes, exact patches, local tests and an independently computed Python literal PRNG/bootstrap fixture. See delegate-verification.md and archived original/repair packets, QA, checks and adjudications. Algorithms are unchanged by the repair.

## Scope

Source/plan are frozen before the 768 production games. No outcome-based tuning, cherry-picked matchup replacement or early stopping is allowed. This is a descriptive development panel, not promotion, final evaluation, Elo, learned strength, G2/G3 closure or public deployment. Approximate heuristics intentionally do not implement every mechanic; the pinned Showdown engine resolves real battles. Sealed predecessors stay intact except the six established mutable progress/registry files. Prime remains unnecessary.
