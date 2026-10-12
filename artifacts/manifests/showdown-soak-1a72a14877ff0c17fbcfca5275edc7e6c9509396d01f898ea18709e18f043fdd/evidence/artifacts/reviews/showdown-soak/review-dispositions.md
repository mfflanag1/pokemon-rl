# Local network soak: review dispositions

Fable snapshot `20261011-145757-d1ad452f` reviewed the initial harness, auditor,
plan, sealer and frozen clients. It did not execute tests. The following claims
were independently checked and addressed before any production game.

1. Accepted-choice timing: a fixed 800 ms opponent delay could allow the turn to
   resolve before recovery. The instrumented baseline transport now holds at most
   one action until the tactical client's committed restoration. The trace records
   the original attempt and separate actual delivery. Both smoke fault-seat cases
   pass; a dropped delivery record fails the audit. Neither frozen policy changes.
2. Received teams: outgoing upload alone was insufficient evidence of the exact
   simulator roster. The first smoke run attempted the existing `/showteam` path;
   pinned chat.ts requires registration in temporary rooms, and the failure is
   preserved. An additive local-only plugin now calls `RoomBattle.getTeam` with the
   real requester, returns privately and denies target arguments/nonparticipants.
   Pinned room-battle.ts and battle-stream.ts show the returned packed `side.team`.
   The second-preflight auditor compared every field after the pinned challenge
   validator's normalization. Early expectations missed default Serious nature and Ogerpon's
   event Hidden Power metadata; diagnostic failures remain preserved. The corrected
   audit passes all eight games, both actual policy seats and all 302 decisions.
3. Sealing: the sealer now regenerates teams and runs the complete auditor with
   `--check`, without writing a new audit, and requires equality with the saved
   audit. It also binds the installed local receipt plugin to its tested source.
4. Reconnect budget: all three recoveries per session are intentional; no spare
   retry is promised. The plan explicitly fails any unplanned loss.
5. Retired session traffic: observers ignore a socket once its session has rotated
   or the clients have been retired. Current-session connection history stays in
   the evidence; only stopped clients are retired.
6. Additional audit checks: before-choice/lost-write restorations must be
   uncommitted; undelivered choices cannot receive a server sentchoice marker;
   uploads precede challenge/accept on the same connection; held baseline actions
   are released only after committed restoration. Timeouts have an explicit game
   index and deadline message, and failure paths settle outstanding listeners.
7. JSON fidelity: source inspection confirms the frozen TacticalView stores plain
   objects and arrays (temporary Sets are spread back to arrays). Both capture and
   reconstruction now reject non-JSON object types, undefined and non-finite fields,
   and tests prove Map/Set and other lossy values cannot pass vacuously.

The initial 41 focused tests and 421-test full regression pass with 296 tested
source/configuration files unchanged. Opposite-model scheduler QA and both
initial/repaired delivery dispositions are recorded separately.

## Focused follow-up

Fable snapshot `20261011-151410-15a3a3b6` reports no blocker and identifies one
useful additional delivery-control check. The plan promises at most one held
baseline action: if no baseline action is attempted until after tactical recovery,
no hold is necessary. Requiring an actual hold in that case would introduce an
artificial scheduling failure. The invariant that matters is now checked directly:
in every accepted-choice fixture, no baseline action can be delivered until the
tactical committed restoration. A new corrupted trace changes a held action into
an early direct delivery and is rejected. Recorded held attempts still require
one matching release. This is a coordinator-verified repair, not a third snapshot.

The serialization finding prompted a full production-pool preflight. It exposed
an important format mismatch: 164 of 1,000 official generated teams fail constructed-team
legality (Raging Bolt's Attack IV requirement is one example), while the normal
Random Battle format bypasses per-set legality. No team is replaced or modified.
The final local format accepts only the exact regenerated official pool and keeps
all resolved battle rules. The actual private receipts retain all packed fields.
The first serialization preflight also incorrectly demanded identity before the
first move-name canonicalization; the corrected check requires stable canonical
roundtrips. Diagnostic logs are retained. Full-pool results are recorded separately
in production-team-normalization.json. All 1,000 exact-pool admissions are mutation-free, with 1,000 stable canonical pack
and receipt roundtrips. The v3 network preflight passes 324 decisions
and six faults; its trace replaces the unsealed v2 regression fixture, whose bytes
are preserved in the review directory.
JSON validation applies to the structured-cloned snapshot; class prototypes are
intentionally removed first. It is not a generic proof for future class-based
actors or sparse arrays. The frozen actor's fields are plain data; Map/Set/date,
undefined and non-finite values are explicitly rejected in the executed cases.
The existing client's 30-second recovery failure remains a visible failure event;
a held baseline action cannot turn that into a successful game or indefinite wait.

Final source-bound regression is rerun after the new audit guard and negative test.
No production qualification is claimed until that regression, the frozen run,
complete audit, all evidence hashes and repeat sealing pass.


## Format admission review

Fable snapshot `20261011-152759-3cc015c7` identifies no blocking defect in the
allowlist or set preservation; its evidence gaps were checked against pinned
source rather than accepted as defects. No reviewer executed tests.

- `sim/pokemon.ts:329–330` uses exactly the implemented predicate and baseSpecies
  replacement for the original set nickname. The full-pool canonical roundtrips
  and actual v3 receipts supplement that primary-source check; the unit test alone
  is not independent simulator evidence.
- `config/config-example.js:87–89` explicitly documents `subprocesses: 0` as
  disabling subprocesses. `server/index.ts:103` defines global Teams, and even
  `team-validator-async.ts:100` defines it in a child. The suggested undefined
  global defect is not present in this pinned runtime; no speculative require
  change or extra server rerun was made.
- `sim/dex-formats.ts:637–650` loads and merges the exact copied custom format.
  Its sanitized name equals the sealed `DEV_FORMAT` ID; raw display-name equality
  suggested by the reviewer would be incorrect. The actual private own-team
  receipts, uploaded pool, loaded tier and full file equality are independently
  checked. The pinned standard list contains no competing development format.
- The complete original format declaration (`config/formats.ts:28–35`) has no
  battle hooks or Tera-preview override. Mod/rules are copied, game type is singles;
  description and best-of UI default are intentionally outside this single-game
  local format. The executed test verifies the full resolved rule/value tables.

Source excerpts and hashes are in pinned-format-evidence.md. These checks close
the review's stated gaps without changing tested implementation after the review.


Final validation before production: all 46 focused tests and the full 426-test
regression (237 Node + 189 Python) pass on 298 unchanged source/configuration
files. The fixed production run was launched only after those results and all
three snapshot dispositions. The format review's implementation hashes remain
unchanged; only its supplementary pinned-source evidence was expanded afterward.
Initial/follow-up source deltas are recorded explicitly. They include coordinator
repairs covered by the final tests and the focused format snapshot, not an implied
blanket approval of every final byte by every reviewer.


## Production disposition

The unchanged admitted tree completes all 1,000 games, 60 faults and 46,024 audited
decisions with zero unavailable-choice retries or campaign failures. Elapsed time
is 169.097 seconds including setup; peak sampled client RSS is 445.063 MiB. All
1,523 unchanged copied upstream files match. Eight production-evidence negatives
fail at their intended gates and cannot write a receipt/archive; a changed saved
audit is rejected by an actual fresh full reconstruction. No test-suite rerun is
needed after the passing source-bound regression because implementation, tests,
plan and runtime remain unchanged. Only current progress documentation and
verification evidence are updated before sealing. Full G2/G3/G4 remain open.
