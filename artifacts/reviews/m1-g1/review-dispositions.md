# G1 and slice review dispositions

Fable G1 review `20261010-171112-78fe749d` was a static review, not an executed check.
The core clipped-surrogate gradient, entropy sign, GAE arithmetic, masks and Adam
were confirmed by independent local tests. Verified findings were handled as follows:

1. The first delayed task had independent rewards. Plan v2 now uses one shared
   win/draw/loss event, with explicit p1-win/p2-win/both-correct/both-wrong fixtures.
   Both perspectives still exceed .99 in all three accepted runs. V1 is preliminary.
2. Buffer-cut GAE is hand-tested; the collector supports whole episodes only.
   End-to-end persistent-game cuts and live-state checkpoints remain future work.
3. A trainer-level test now verifies first-minibatch ratios equal one before update.
4. Stale partial checkpoints reject with a clear preservation/recovery instruction;
   the test preserves the partial and successfully writes a new complete checkpoint.
5. Minibatches use their own means; the 384-row delayed buffer ends with 128 rows.
   This convention is explicit and not claimed to be a global per-buffer mean.

Fable slice review `20261010-171531-8691d817` independently checked rounding,
priority, KO cancellation and the four-turn bound by source inspection/arithmetic.
The verifier now checks the declared Showdown pin, reference manifest identity,
selected source hashes and build receipt. It independently derives all mechanics
seeds and action tapes. Winner extraction uses side identity through the fixed names
rather than parsing their spelling. Hard-coded arithmetic is intentionally scoped
to the exact fixed profile and checked against all 32 reference damage outcomes.

These fixes postdate the review packets. Their local checks are separate evidence;
no claim is made that reviewers approved code not in their packets.

Early local failures retained in logs: the coordinator's second GAE hand example
used .04 instead of .24 for `.1 + .9*.6 - .4`; the expected values were corrected
and independently recomputed. The first slice canary needed BattleStream keepAlive
for destruction after terminal; this was corrected before the accepted campaign.
A source packet initially exceeded the delegation file limit; the research run
was relaunched with explicit small excerpts. No model result came from that attempt.

Bounded research run `20261010-170729-a716560c` received opposite-model QA and
was rejected as a complete specification. The verified arithmetic was useful;
raw rolls cannot be reconstructed uniquely from damage logs, format phases needed
to be fixed, and the claim that supplied teams are ignored by the Random Battle
harness was false. See `delegation-verification.md`; no unchecked result was
integrated and the rejection is recorded in the routing ledger.
