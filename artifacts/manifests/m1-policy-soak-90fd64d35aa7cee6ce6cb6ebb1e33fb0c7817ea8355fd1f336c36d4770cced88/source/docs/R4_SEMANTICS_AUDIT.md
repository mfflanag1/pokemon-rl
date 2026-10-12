# R4 delegated lifecycle audit: verified disposition

2026-10-10. Immutable evidence for the cross-model quality ledger; later campaign
results belong in R4_REVIEW.md.

Research run `20261010-133750-ab4cf5b8`, project `Pokemon-83aa7cb62b4e`:
Fable read the pinned battle/side sources and existing R1 session/projection;
Codex supplied opposite-model QA. The candidate is **rejected as a complete
fixture specification**, while the verified lifecycle facts below were retained.
No reviewer executed the fixtures; the coordinator performed the source checks
and runtime verification described here.

At Showdown commit `942cda71d14bf0618bd57991708a2e2734b6daba`:

1. `battle.ts:1381` clears every side's choice for a new typed `makeRequest`,
   rebuilds the requests and marks them for emission. `commitChoices` at `:2996`
   queues the completed choices and clears requests before continuing execution.
   `side.ts:527` instead updates and re-emits only the rejecting side's request.
   It does not clear the opponent's completed choice awaiting group resolution.
2. Request IDs count per-side emissions, including wait requests. Ordinary groups
   emit to both players; a private re-request increments only its recipient's
   counter. The candidate attributed divergence too loosely to wait requests.
3. The proposed trapping fixture with a trapping ability *not selected* was wrong:
   `maybeTrapped` alone does not establish actual trapping. `side.ts:984` checks
   authoritative `pokemon.trapped`. The executed `r1-smoke-8` fixture produces two
   real `[Unavailable choice]` rejections. Running R4 with p2 submitted first
   preserves p2's choice across each p1 re-request; audit hashes confirm unchanged
   HP/PP/RNG/selected mechanics and same group/turn. Both complete tapes replayed.
4. The source turn-limit test is `turn > 1000`; the candidate's exact "turn-1000"
   description was imprecise. R4 records the reference's emitted terminal turn
   without implementing its own adjudication. A submission safety cap is failure;
   an explicit collection cut is labeled truncation with bootstrap observations.
5. The candidate correctly qualified its missing `pokemon.ts` evidence but then
   overstated the emitted revival discriminator in one table. We independently
   read `pokemon.ts:1153–1190`: `reviving` is emitted from the active Pokémon's
   Revival Blessing slot condition. `side.ts:965` accepts fainted revival targets.
   R4 distinguishes revival from replacement and live switch continuation.
6. A parsed but incomplete overall submission returns false with an Invalid-choice
   error (`battle.ts:2972`); it should not be labeled an accepted submission.
   The pinned singles adapter rejects visibly invalid/stale actions before writing.
7. The existing R1 tape includes attempted commands while Showdown's internal
   input log records committed normalized choices. R4 preserves its complete
   external tape and does not substitute the latter. It also leaves all R1/R2/R3
   source files and manifests unchanged and binds constructed test interventions
   to a separate profile and named recipe.

The wrapper's selected audit is privileged, not a complete simulator checkpoint.
Per-player views contain only the corresponding request and visible protocol
delta. New-group, pending-choice and re-request diagnostics are separate fields.
Replay from original teams, seeds and the complete tape supplies the executed
correctness evidence; this source audit alone is not a reachability proof.
