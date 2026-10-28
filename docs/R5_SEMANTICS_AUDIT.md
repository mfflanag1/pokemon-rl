# R5 source audit and delegation adjudication

Date: 2026-10-10. Pin: `942cda71d14bf0618bd57991708a2e2734b6daba`.

Bounded Fable 5.1 research run:
`local-review-archive/20261010-143114-84ebe47f`.
An independent Codex QA returned **revise**. Coordinator disposition: **rejected
as a complete semantic specification**; use only independently verified facts.
No implementation patch was produced or integrated. Research had no executable
checks; its `checks_passed: false` is not a failed code test.

The supplied source snapshot is `artifacts/reference/r5-information-source.txt`.
The coordinator additionally inspected pinned `sim/battle.ts` and ran controlled
live observer probes through the actual ReferenceSession and channel projection.

Verified corrections to the candidate:

- Its general HP interval formula was wrong at 100. `pokemon.ts:2077–2081`
  rounds upward but changes a non-full 100 to 99. Thus 100/100 means exactly
  full; 99/100 permits fractions strictly greater than 98% and strictly less
  than 100%. R5 stores displayed fractions and makes no exact-HP estimate.
- The initial snapshot omitted serialization evidence for Transform. Additional
  `battle.ts:3091–3114` joins non-function arguments; `pokemon.ts:532–534`
  stringifies a Pokémon to its public ident. The live fixture verifies the
  emitted third Transform field equals the target public ident, for both seats
  and both transformation directions. The encoder resolves only public appearance.
- “Current ability is never certain” overstates the cited silent Skill Swap
  exception, which concerns allies in doubles/triples. R5 is singles and labels
  stored revelations as last-observed evidence, without claiming permanence.
- The candidate's faint-time Illusion identity inference was unsupported.
  `battle.ts:2556` emits `faint` before its Faint event/callback and clearing
  `illusion`. R5 never counts distinct roster members from names or appearances.
- Transform PP is capped with `Math.min(5, move.pp)` (`pokemon.ts:1318`), so
  it is not unconditionally five. R5 uses the current advertised request and
  explicitly marks absent PP unknown, including Recharge and Struggle.
- Own request `stats` are `baseStoredStats`; Transform retains these while the
  actual effective stats may change. Live own-Transform assertions verify the
  encoder preserves request base stats, HP and level.

Evidence retained from the candidate and verified locally: use separate encounter
segments instead of guessed opponent roster slots; preserve pre-reveal inputs;
keep raw public events; never assign item/ability ownership from generic tags.
R5 attributes `[from] item/ability` only for damage/heal messages, whose source
rules are explicit in `battle.ts:2156–2163` and `2295–2302`. Trace/Frisk and
other tags remain raw unless a direct message names the revealed subject.

Public `As One` is a separate ambiguous label, not an inferred choice between
its two private ability IDs. The generated catalog also adds the request-only
Recharge label. A scan of all 1,000 R4 bundles found no unresolved known labels.

The live tests are deliberately constructed simulator interventions, not claims
that each edited state is reachable by legal play. The 1,000 carrier mutation
tests separately check adapter noninterference and are not valid R4 artifacts.
These distinctions are part of the report, not waived acceptance conditions.
