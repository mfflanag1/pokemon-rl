# R5 independent review and verified corrections

Fable 5.1 reviewed a frozen implementation snapshot through the
`/Users/maxf/.codex/skills/claude-review/SKILL.md` workflow. Run:
`/Users/maxf/.local/state/claude-reviews/Pokemon-83aa7cb62b4e/20261010-145017-095fa85b`.
It completed in 213.23 seconds and reported findings, not an approval. The reviewer
had no execution tools; the following dispositions are coordinator verification.

1. **Absorb healing credited to the attacker — confirmed and fixed.**
   Pinned `battle.ts:2265–2270` defaults a heal's source from `event.source`;
   `2298–2301` can print that attacker in `[of]`. Water Absorb, Volt Absorb,
   Dry Skin and Earth Eater callbacks call `heal` on their target. New live probes
   dispatch real TryHit events for all four abilities, in both seats, assert the
   emitted healer/attacker fields, and verify only the defending appearance gains
   the reveal. Generic healing with `[of]` is no longer assigned to that source.
   The audited four absorb effects explicitly belong to the healed subject;
   other ambiguous tagged heals remain uninterpreted event bytes. Damage-source
   handling remains separate. This corrects the too-broad damage/heal ownership
   statement in the earlier, immutable research adjudication note.
2. **Carrier pairs test adapter plumbing — accepted limitation, clarified.**
   These regression tests intentionally preserve selected own views and perturb
   privileged carriers. They catch future boundary regressions and are not legal
   simulator counterfactuals. The design, report and handoff distinguish 1,000 such
   pairs from 18 invariant live pairs and two visible HP positive controls. No
   claim of 1,000 distinct reachable hidden worlds is made.
3. **Unmatched structured subjects undercount diagnostics — confirmed and fixed.**
   A bench `|-curestatus|p2: Nick|slp` cannot safely be joined to an active encounter.
   It now increments `unstructured_events`, retains raw history/event bytes and
   leaves active structured memory unchanged. A regression checks all three.

The review found no other actionable issues in transactional append, strict own
request checks, same-input policy/value separation, unknown opponent fields,
Illusion history preservation, Transform public appearance or control actions.
This is a bounded independent review, not a proof of completeness. Reviewed code
was subsequently changed for the verified fixes above and tested locally.

A separate research delegate and opposite-model QA are documented in
[R5_SEMANTICS_AUDIT.md](R5_SEMANTICS_AUDIT.md). Its complete candidate was rejected
and that outcome was recorded in the routing ledger; no unverified recommendation
was treated as source truth.

During campaign startup, a CLI `str`/`Path` mismatch failed before any episode
scan. The failure receipt is retained as `failed-attempt-1/failed.json`; the
corrected run uses a fresh output directory. Earlier development scans also
exposed missing Recharge/As One catalog labels and optional forced-move PP.
Those cases now have explicit representations and regression coverage.
