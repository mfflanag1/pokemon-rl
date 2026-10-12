# R2 delegated semantics audit: verified disposition

2026-10-10. This record is evidence for the cross-model outcome and should remain
unchanged after adjudication. Later implementation checks belong in R2_REVIEW.md.

Delegated run:
`/Users/maxf/.local/state/cross-model/Pokemon-83aa7cb62b4e/20261010-114518-c67fa0a8`.
Fable researched the selected pinned `side.ts`, `pokemon.ts`, and random AI source.
Opposite-model Codex QA completed and requested revisions. Neither response alone
was treated as verified evidence. The research result is **rejected as a complete
codec specification**, while the independently verified source facts were retained.

Verified directly against Showdown commit
`942cda71d14bf0618bd57991708a2e2734b6daba`:

* `sim/side.ts:570` rebuilds the move request for slot lookup; `sim/pokemon.ts:964`
  collapses hard locks/recharge and `:1103` constructs an explicit Struggle entry.
  `side.ts:699` returns Struggle before the Tera checks at `:806`. The research's
  claim that Struggle+Tera might hit those later checks is incorrect.
* `sim/tools/random-player-ai.ts:49` and `sim/side.ts:958` both explicitly allow
  the active index under Revival Blessing. Slot zero remains supported in the
  constructed request fixture; this is not a claim of natural reachability.
* `sim/battle-actions.ts:119–133` reorders the live side Pokémon array on switching.
  The live R2 switch test confirms request slot zero becomes the newly active mon.
* `sim/battle.ts:1422–1434` constructs forced-switch requests only when a switch
  flag is present; `sim/battle-actions.ts:1937` consumes Tera eligibility side-wide.
  These files were inspected locally at the pin, replacing the research's unpinned
  master references. The live semantic Tera test confirms request eligibility drops.
* `sim/side.ts:1330` distinguishes pass control; the codec follows the pinned random
  AI's canonical subset. It does not expose every accepted alias or optional pass
  in unusual constructed states. A false forceSwitch flag produces pass.

Codex QA additionally asserted that an all-disabled post-rejection request required
an unconditional inferred Struggle fallback. That reachability was not established.
Inspection found the following branch order: `getMoves()` at `side.ts:627`, empty
live move list → Struggle at `:699`, and only then disabled-choice rejection at
`:729–762`. The last surviving live move cannot be rejected as disabled through
that path once the live move list is empty.

Executed constructed probe (not a natural fixture): started an official-team R1
session, marked the first three live moves disabled and the fourth secretly disabled,
then called the pinned choice parser. Actual result:

```json
{"visible_disabled":[true,true,true,false],"live_moves":0,"chosen_move":"struggle","request_count_changed":false,"error":""}
```

This confirms an advertised attempt may directly resolve to Struggle without
relabeling the prior visible request. It does not prove every all-disabled shape
unreachable. R2 rejects an all-disabled advertised list explicitly until a pinned
reachable fixture demonstrates the required behavior.

Executed second constructed probe: passed the all-fainted revival fixture to the
pinned upstream RandomPlayerAI and captured its candidate switches. Result was
one-based `[1,2,3,4,5,6]`, confirming the codec's zero-based `[0,1,2,3,4,5]` list.

Retained findings: request-relative indices, no hidden-truth mask filtering,
explicit wait/pass handling, strict unsupported-format rejection, and recording
unavailable attempts separately from the updated request. Rejected parts:
overbroad guaranteed-acceptance language, inconsistent Struggle/Tera claim, and
unpinned source as sufficient pinned evidence. QA's speculative fallback was also
withheld rather than applied without a reproduced counterexample.
