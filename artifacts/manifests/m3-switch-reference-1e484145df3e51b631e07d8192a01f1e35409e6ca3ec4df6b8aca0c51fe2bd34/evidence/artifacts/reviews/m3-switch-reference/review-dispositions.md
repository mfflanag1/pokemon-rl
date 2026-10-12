# M3 verified review dispositions

Two Fable 5.1 frozen-snapshot reviews were collected with the claude-review skill.
They are advisory, not executed verification. Main snapshot:
20261011-112205-7dd4c5e9. Boundary follow-up:
20261011-113056-366204b9. Neither reports a blocking defect within its reviewed
scope; the follow-up identifies evidence gaps that were repaired before production.

1. **Unseen execution evidence:** The main reviewer had only source, not generator
   output. Draft v3 actually completed 72 scripts and the full information matrix;
   draft v4 additionally includes the 3,456-value damage grid. The follow-up packet
   includes v4 summary.json. Final production and source-bound regression are the
   completion evidence, not the reviewer's opinion.
2. **Old encoder incompatible:** Confirmed by executing the first draft adapter.
   The actual reason is the frozen R2/R5 schema's six-member minimum, not the main
   review's conjecture that it was a one-member codec. Preserved that implementation;
   added new strict Node/Python four-action codecs and a structured actor adapter
   using only the old public event reducer. The follow-up found no codec divergence
   or actor-boundary defect. All saved current requests and constructed actor inputs
   are actually replayed by independent Python.
3. **Negative HP checks:** Added a per-player channel check across all natural
   frames, including terminal frames, plus public-reducer HP validation for every
   actor/counterfactual view. Both switch and damage exact-opponent-HP injections
   reject in the focused test. Foreign private requests, outcomes and privileged
   actor envelope keys reject. The profile itself requires non-debug masked HP.
4. **Own speed control:** Removed the tautological simulator-vs-request comparison.
   Grouped actual own views by otherwise identical team choices; 576 pairs require
   distinct own view AND actor hashes for plain versus swift. The test suite also
   changes a consistent own request/history speed variant and requires changed input.
5. **Counterfactual hashes:** Confirmed the follow-up's gap: the previous consumer
   compared only Node-produced hashes for the large matrix. Final producer retains
   all 3,456 underlying initial/post-turn/other-player views. Python recomputes each
   view hash and actor input, then checks the equal/unequal groups. Counts are
   derived from verified groups; expected matrix sizes are also enforced. The
   expanded audit passes 5,426 total actor views. Full producer/consumer source is
   bound to the final run, with a separate exact regeneration in another process.
6. **Coverage and roster:** Confirmed the earlier consumer copied coverage. Final
   Python independently counts every reported coverage metric from frames, private
   audits and public protocol, requires equality with producer counts and positive
   coverage, and enumerates the exact 24-team roster from the declared profile.
7. **Chance and damage hygiene:** Confirmed pinned battle-actions.ts uses
   randomChance(accuracy,100) even at numeric 100 accuracy. Documented that a future
   chance law cannot assume raw stream alignment. Conditional moves now come from
   battle.dex; the intercepted bounded damage draw leaves real integer rounding
   intact. Injected private HP/crit/damage outcomes are diagnostic, never actor data
   or claims of natural chance-frequency coverage.
8. **Delegation:** The codec delegate failed ownership because the coordinator used
   repeated --owned options. No candidate code integrated; controller refused QA
   and adjudication for failed status. Preserved as unverified delivery, with the
   coordinator launch error distinguished from code quality. See
   delegate-verification.md and delegate-failed/.

The follow-up predates the expanded comparison storage, independent recount,
exact roster check and second HP test. These are verified local repairs, not a
claim that the final bytes received a third snapshot review. Reviewed/live source
hash deltas are retained separately. No JAX engine parity, sampler-frequency gate,
policy training, strategic strength, full G2/G3 closure or final-data use is claimed.
