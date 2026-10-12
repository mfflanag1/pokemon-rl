# Verified review dispositions

Two Fable 5.1 snapshot reviews completed. The main review took 297.71 seconds;
the focused completion-sealer review took 94.73 seconds. Neither reviewer ran
code. Their requests, source hashes, statuses and full responses are preserved
in main-snapshot and sealer-snapshot. Local checks establish the results below.

## Main capture, mapping and integration review

1. **Final reservation and fresh seed claims — inspected, no missing gate.**
   campaign.admission calls binary.qualification.admission, then fixed.campaign.admission,
   then jax_learner.campaign.admission_checks. The last function verifies the sealed
   R7 receipt, opens ActorDataset in training mode, checks teams/final.json against
   its pinned collection hash and requires final_episodes == 0. The new campaign
   compares admission before and after execution. The reviewer did not receive
   these predecessor implementations. The reservation flag is consequently backed
   by a real recursive gate. The two excluded namespaces match the source of the
   older 10,000 conditional and 1,000 player reference captures. Binary qualification
   generates no additional Showdown reference corpus. The disjointness claim is
   limited to those prior reference campaigns, not arbitrary future namespaces.

2. **Copied collector and actor stream — verified and strengthened.** A source diff
   shows collect and cycle bodies unchanged from pokemon_binary; the chance import
   and plan path change, while actor, update, GAE and encoder are imported unchanged.
   DYB1 is folded inside binary.policy.action_kernel (via its internal sampler),
   so the checkpoint stream description is correct. The integration test now also
   executes the predecessor collector's actual code with only sample_kernel replaced
   in a private copy of its globals. Three seeds at a nonzero game counter produce
   leaf-exact buffers and terminal states versus the new compiled collector. It does
   not mutate predecessor files or modules. All four campaign tests passed after
   this addition; the full source-bound regression is rerun on the final tree.

3. **Word-oracle self-hash — independently regenerated before sealing.** Re-running
   the actual pinned Node oracle in a temporary directory produced byte-identical
   output for all 100,096 words; see oracle-regeneration.json. Its file SHA256 is
   0321da68a47c637a793821cb4a93dbc699bf3246e09044318cf8e69518013a34.
   Capture admission also binds engine, provenance and selected upstream source,
   compares the current complete describe() manifest and verifies the fixed pin.
   The completion receipt is the trusted content-addressed anchor for these files.
   Self-hashes alone are not authentication against someone who can rewrite all
   source and evidence; that is not the receipt's threat model.

4. **Consecutive shuffle linkage — all captured ties checked.** An independent
   shard pass verifies that the second shuffle's input order equals the first
   shuffle's output for all 15,626 tied turns. Exactly two shuffles occur in each
   captured tied turn. The production validator deliberately validates each
   recorded input ordering and final executed seat; it does not impose a universal
   shuffle schedule on other profiles. See coverage-inventory.json. Both input
   orders, output seats and all sixteen damage rolls occur in this capture.

5. **Untraced word gaps — expected and documented.** Upstream hitStepAccuracy
   invokes randomChance(accuracy,100), including 100-accuracy moves. Capture records
   selected semantic chance roles only, while the monotone raw-word index counts
   all post-setup draws. Gaps do not imply missing critical/roll/tie data. Neither
   the contract nor handoff claims consumed-word schedule equivalence.

The finite-word arithmetic and role layout passed the review's inspection. The
only changed main-packet source after review is the strengthened integration test;
production Python, contracts, capture and oracle source are unchanged.

## Focused completion-sealer review

The reviewer found no blocking evidence-binding or false-promotion defect.
Two inexpensive hardening changes were accepted: historical campaign reports now
have their content hash verified before entering history, and the accepted delegate
outcome must bind the actual local verification record's hash. Failed historical
runs, if present, remain historical evidence; they are not promoted as successes.

The sealer's two restored runners are a deterministic repeat/loadability check,
not an independent original-versus-restore experiment. The three original-versus-
restore comparisons belong to the source-bound campaign. Scope prose does not
count the sealer repetitions as extra independent replay evidence. Review prose is
retained and hashed; its substantive adjudication remains human-readable here.

The sealer requires all 100 ordered shard reports and their hashes, exact aggregate
view/event accounting, mapping-audit equality, source/runtime admission equality,
checkpoint identities/counters and the complete source-bound regression. Dedicated
negative tests reject missing shards, altered counts, hash/identity substitution,
failures and unsupported scope promotion. The final source revision differs from
the sealer review only by the two checks described above.

## Delegation and remaining scope

Fable's bounded mapping implementation passed its frozen tests and coordinator
checks; Codex supplied opposite-model QA. The actual QA verdict was revise due to
a hypothetical mixed-device admission edge, and remains recorded that way. The
coordinator accepted the unchanged code only for the explicitly admitted local CPU
scope after checking the integer proof, actual upstream fixtures, arbitrary-precision
probes and result devices. See delegate-verification.md and delegate/outcome.json.
No accelerator or mixed-device qualification is inferred.

The 10,000-game campaign has zero disagreements. Full G2/G3 remain open. There was
no new two-hour soak, strength evaluation, final-data use, GPU run or Prime resource.
