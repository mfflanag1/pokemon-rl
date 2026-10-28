# C1: deterministic train-only battle curriculum

The additive module completed a 16-game mixed learning canary on 2026-10-15. It has not measured strength or curriculum benefit, or accessed sealed final data. It provides actual six-member battle starts and ordinary scripted actions in the pinned Showdown engine; it does not patch HP, statuses, teams, or other private simulator state after battle initialization.

## Interface and boundaries

`packages/training-curriculum/index.mjs` exports:

- `curriculumEpisode({episodeIndex, runSeed})`: deterministic coordinator-only plan containing two admitted team records, canonical mechanics seed, scripted `prefix`, public-event `evidence_requirements`, and provenance.
- `curriculumSlot(episodeIndex)`: cheap schedule lookup without generating teams or running games.
- `PLAN`, `CATALOG`: immutable frozen schedule and scenario roster.
- `admitTrainingTeam(entry, source)`: verifies content hash/six-member structure and applies the original R7 partition function. Returns `null` for non-training candidates; caller labels cannot override admission.
- `verifyPublicEvidence(spectatorLines, requirements)`: checks that the required public mechanics occurred, or rejects the handoff. It rejects private request/unprojected split channels.

The full-Showdown coordinator can call `FullShowdownSession.start({episodeIndex, runSeed, plan: curriculumEpisode(...)})`, or use the worker's `curriculum: true` integration. That session owns requests, action encoding, unavailable-choice handling, and eventual rewards. Validate each prefix group against **all and only currently pending players**; a U-turn prefix includes a one-sided replacement group. Stop on script/evidence failures rather than substituting another seed or scenario. After the prefix, continue the live battle with the caller's actor until the ordinary terminal result; do not create an artificial terminal reward at handoff.

Only the player's projected protocol and own current private request may enter the actor. Full team plans, `scenario_id`, family IDs, expected events, hidden reserve details, and coordinator provenance are not actor features. Fixed prefix actions have no sampled behavior probability and are explicitly marked `prefix_actions_trainable: false`; exclude them from PPO and imitation targets. Initialize the actor's visible history/memory by replaying its projected prefix before it chooses its first action.

## Fixed schedule and budgets

The `contracts/training-curriculum-v1/plan.json` schedule is determined solely by zero-based episode index. `runSeed` separates mechanics and official-team random streams; it does not change the scenario schedule or constructed roster.

| Episode indices | Targeted slots per block of 20 | Targeted episodes | Official complete battles |
|---|---:|---:|---:|
| 0–999 | First 10 | 500 | 500 |
| 1000–1999 | First 5 | 250 | 750 |
| 2000–2999 | First 2 | 100 | 900 |
| Total | — | 850 | 2150 |

Targeted slots cycle through ten frozen scenario IDs: 85 instances of each, 170 instances of each family. Indices outside 0–2999 fail; extension requires a new reviewed plan rather than implicit repetition. The official branch has a maximum of 128 generator candidates per seat and rejects an identical opponent team key. Admission exhaustion fails the episode without outcome-based replacement. No win, loss, value estimate, or learner state influences scheduling, seeds, or admission.

This is a schedule contract, not authorization to execute 3,000 training episodes. The executed mixed canary used indices 0–15 only; the complete 3,000-index schedule has not been run.

## Realized target states

Every constructed team has six distinct species with legal Gen9 sets, checked by the pinned `TeamValidator('gen9anythinggoes')`. Battles execute the pinned `gen9randombattle` battle rules with explicit constructed teams. This **does not make those teams official Random Battles samples**. Official slots call the unmodified whole-team generator and then apply the train partition gate.

| Family / scenario IDs | Fixed warmup and required public evidence | Handoff limitation |
|---|---|---|
| `switching-p1-v1`, `switching-p2-v1` | Scizor uses U-turn; the resulting replacement request selects Ting-Lu; public U-turn and Ting-Lu switch events must appear. | Learner starts after the scripted pivot, not at an unsolved replacement decision. Later switching remains under actor control. |
| `hazards-p1-v1`, `hazards-p2-v1` | Ting-Lu sets Stealth Rock; opponent switches to Snorlax; public hazard setup and attributed entry-damage events must appear. | One hazard family and one entry example, not broad hazard coverage. |
| `status-p1-v1`, `status-p2-v1` | Amoonguss uses Spore after Slowbro's harmless action; a public sleep event appears and the current request confirms sleep. | Sleep only; other statuses need new scenario versions. |
| `tera-p1-v1`, `tera-p2-v1` | Slowbro publicly Terastallizes to Fire while its opponent Protects. | Learner responds to revealed Tera; this does not itself teach the initial Tera decision. The other side still has its own Tera option. |
| `hidden-reserve-a-v1`, `hidden-reserve-b-v1` | Identical visible lead actions; Corviknight remains unrevealed. The variants differ in its item, Tera type, and fourth move. | A controlled hidden-information invariance check, not a calibrated posterior over general opponent teams. |

`tests/training-curriculum/prefix-evidence.json` records the fixed seeds, team hashes, submitted prefix choices, matched events, and normalized public logs from all ten actual engine runs. Reproduce it without writing files using `node packages/training-curriculum/audit.mjs`. Timestamp normalization uses the existing reference projector. No public outcome or final-team record is needed.

## Data isolation and family leakage

Both official and constructed teams pass `packages/dataset/identity.mjs` with the unchanged R7 policy/salt. Constructed roster admission considered only deterministic last-reserve HP-EV candidates (up to 64), before any battle results; the accepted exact teams and hashes are now fixed in `catalog.json`. Rejected candidate team bytes never enter battles, actor inputs, saved fixtures, or the catalog. Tests construct temporary counterexamples assigned to development/final only to verify rejection; they do not read the sealed reservation or run those teams.

Synthetic leakage uses a separate `training-curriculum-v1/family/...` namespace. Mirrored scenarios share a family group; the two hidden variants additionally share a counterfactual group. All groups are training-only. Never relabel a mirrored/counterfactual variant as development/final, or use those related scenarios to claim independent generalization. Any future targeted evaluation needs separately designed held-out families. General strength must be evaluated on untouched official-team comparisons using the existing evaluation controls.

The repeated small constructed catalog can be memorized from legitimate observations even without explicit scenario IDs. Official mixing and withheld metadata reduce obvious shortcuts but do not prove transfer. Curriculum usefulness requires a later matched-budget ablation against official-only training. The present fixtures establish executable setup, data admission, actual mechanics, and information boundaries only.

## Verification performed

Run `node --test tests/training-curriculum/curriculum.test.mjs` from the project root. Checks cover exact 3,000-index roster/counts, six-member legality, frozen R7 admission/rejection, deterministic immutable plans, actual public-event realization for all ten scenarios, hidden-state counterfactual equality before reveal and divergence after reveal, and natural completion of one bounded battle under independent damage-policy actors. That fixture audit itself performed no production training, package installation, existing sealed-source edit, or final-data evaluation. The separate learning canary is reported below.

## Measured integration — 2026-10-15

The fixed-seed mixed canary executed all ten targeted prefixes, followed each with ordinary actor-controlled continuation, and included six official generated battles. It collected 1,859 actor decisions, six natural terminal outcomes and ten administrative turn-60 cutoffs. Two eight-game cohorts performed four finite Adam steps with changed parameters. Prefix actions remained excluded from learning rows. Aggregate collection throughput was 932.4 decisions/s and collection plus updates (including compilation) was 544.0 decisions/s.

This verifies that the curriculum integrates with complete-game collection and real updates. It does not establish that the curriculum improves learning; that requires a separate matched-budget comparison with independent evaluation. The 3,000-episode progressive schedule remains unexecuted. See `docs/F1_FULL_SHOWDOWN.md` and `artifacts/reports/f1-c1-canary.json` for the paired engineering records and preserved original failure.
