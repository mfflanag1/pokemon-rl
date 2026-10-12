"""Publish measured current-stage documentation before immutable sealing."""
import json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
r=json.loads((ROOT/'artifacts/runs/m3-switch-screen-v1/report.json').read_text())
exit_=json.loads((ROOT/'artifacts/reviews/m3-switch-screen/exit.json').read_text())
assert r['status']=='passed' and r['mode']=='development' and exit_['exit_code']==0
marker=ROOT/'artifacts/reviews/m3-switch-screen/documentation-updated.json'
assert not marker.exists(),'Completion documentation was already published; inspect it rather than appending twice'
an=r['analysis'];assert an['games']==31104 and len(an['cells'])==27
p=lambda x:f'{100*x:.2f}%'
result='passed' if an['all_seed_screen_passed'] else 'did not pass'
training=[t['checkpoints'][-1] for t in r['training']]
decisions=sum(c['decisions'] for c in training);terminal=sum(c['terminal'] for c in training);cutoffs=sum(c['truncated'] for c in training)
seconds=sum(t['elapsed_seconds_including_replay'] for t in r['training'])
wall=(datetime.fromisoformat(r['ended_at'])-datetime.fromisoformat(r['started_at'])).total_seconds()
primary='\n'.join(f"| {c['seed']} | {p(c['outcome_lower'])} | {p(c['outcome_lower'])}–{p(c['outcome_upper'])} | {p(c['baseline_lower'])}–{p(c['baseline_upper'])} | {100*c['secured_win_change']:+.2f} pp | {100*c['adjusted_improvement_lower']:+.2f} pp | {'Pass' if c['passes_screen'] else 'Fail'} |" for c in an['primary'])
endpoints='\n'.join(f"| {c['seed']} | {c['opponent']} | {c['wins']} | {c['losses']} | {c['cutoffs']} | {p(c['outcome_lower'])}–{p(c['outcome_upper'])} |" for c in an['cells'] if c['update']==8192)
handoff=f'''# M3 restricted categorical learning screen

The preregistered experiment **completed successfully; the conservative learning
screen {result}**. All three fresh training seeds and all27 evaluation cells
are retained. A failed learning threshold is distinct from an execution failure.
Full G2/G3/G4 and public ladder strength remain open; Prime is unnecessary.

## Measured results

The unchanged learner trained24,576 updates,786,432 capped games and{decisions:,}
player decisions across seeds1009/2027/3037. Training produced{terminal:,} terminal
and{cutoffs:,} explicitly truncated games. Training plus boundary replay took
{seconds:.3f} seconds; complete training/evaluation/action-audit wall time was
{wall:.3f} seconds. Timing concerns this local CPU implementation, including host
checks; it is not full-format or GPU throughput.

All **31,104 pinned Showdown evaluation games** completed the declared horizon:
**{an['terminal']:,} terminal**, **{an['truncated']:,} nonterminal cutoffs**, and
**{an['candidate_decisions']:,} candidate decisions**. Each seed contributes9 cells
and10,368 games. Nine saved checkpoints reproduce the original next learner,
rollout buffer, final environment and metric fingerprints exactly. Every stored
candidate action, behavior log probability and value is reconstructed from the
original receiving-player protocol prefix.

Primary endpoint versus random legal (1,152 games/576 complete seat pairs per seed):

| Training seed | Secured wins | Final outcome range | Initial outcome range | Observed secured-win change | Adjusted worst-case improvement lower bound | Gate |
|---|---:|---:|---:|---:|---:|---|
{primary}

The outcome range is [wins/games,(wins+cutoffs)/games], bounding arbitrary binary
assignments to unresolved games. It does not simulate a longer battle. The adjusted
improvement bound compares final wins with initial wins **plus all initial cutoffs**,
then applies the preregistered pair-level Hoeffding radius. All three seeds must
pass; there is no seed pooling or best-checkpoint selection. Improved secured wins
alone cannot pass this conservative gate. These are in-domain development results
on the constructed24-team profile, not evidence of full-format generalization.

All frozen opponent endpoints (each1,152 games):

| Seed | Opponent | Wins | Losses | Cutoffs | Outcome range |
|---|---|---:|---:|---:|---:|
{endpoints}

All initial and2,048-update cells remain in the
[complete report](../artifacts/runs/m3-switch-screen-v1/report.json). No outcome-based
rerun, budget extension, seed selection or parameter change was made. Critic values
and terminal-only value errors are descriptive; no calibration/convergence gate is
claimed, and no critic prediction decides a cutoff outcome.

## Frozen subject and protocol

The [plan](../contracts/switch-screen-v1/README.md) and its JSON contract were frozen
before training. Three fresh seeds each receive exactly8,192 updates and retain
checkpoints0/2,048/8,192. The opponents are equal-logit random legal, immediate
damage-greedy with forced switches only, and each seed's frozen initialization.
Each cell covers all24×24 ordered candidate/opponent team combinations in both
seats. The same team pair and mechanics seed are shared across seats/checkpoints;
diverging actions do not promise identical realized chance events. Candidate and
opponent randomness have separate role addresses and smoke uses a separate namespace.

This uses the unchanged sealed M3 scalar/JAX semantics,323-feature actor/value
model, categorical finite-word law, Showdown chance mapping and PPO kernels.
The [two-hour qualification](M3_SWITCH_SOAK.md) and historical evidence remain
immutable. Only the new screen modules and current registry/progress documents
were added or updated. Actors receive only their own compact player view and mask;
seed/team/pair metadata addresses RNG and never enters their features.

A twelve-turn cutoff remains nonterminal. The inferential unit is a two-seat pair;
one-sided bounds assume independent per-pair chance/policy streams conditional on
the trained model and fixed team grid. A union bound allocates .05/6 over the six
primary bounds. Deterministic PRNG namespaces are not a proof of independence.
The gate requires each final random-opponent secured-win fraction≥55%, adjusted
win lower bound>50%, and adjusted worst-case improvement lower bound>0.
Unexpected ties fail closed; simultaneous-faint effects and PP exhaustion are
outside this admitted profile. No cutoff becomes a draw or terminal result.

## Verification and artifacts

The [completion receipt](../artifacts/reports/m3-switch-screen.json) binds the
source, immutable predecessors, report, review and original evidence. Sealing
requires a fresh original-next-update replay for all nine checkpoints, complete
policy/value replay from every original player-view prefix, and **fresh replay of
all31,104 games** through unpatched Showdown with exact view hashes and complete
final-protocol equality. It then requires an identical reseal and live/archive hash
verification, retained outside the sealed review directory in the
[integrity record](../artifacts/reviews/m3-switch-screen-integrity.json).

The complete source-bound regression passes **476 tests (240 Node +236 Python)**
on347 unchanged implementation/configuration files. The pinned Showdown build
verifies. The final48-game smoke reproduces both checkpoint updates and all48 full
reference games. Nine copied-evidence negatives reject smoke promotion, source/
runtime drift, reserved-data use, missing cells, broken counters/checkpoint outputs,
changed action logs, false terminal outcomes and changed view prefixes. An earlier
smoke remains preserved before the final replay/freeze checks were added.

Independent work used the
[claude-review skill](/Users/maxf/.codex/skills/claude-review/SKILL.md).
The bounded statistics implementation auto-routed to Codex, received Fable QA, then
passed five local frozen tests plus6,561 exhaustive small outcome comparisons.
Two Fable harness snapshots and coordinator verification resolved the conditional
damage-helper, horizon and provenance concerns; see
[review dispositions](../artifacts/reviews/m3-switch-screen/review-dispositions.md).
Static reviewer opinions are not represented as executed tests.

Production directory: `artifacts/runs/m3-switch-screen-v1`. The launcher exits0;
`production.log`, `launch.json`, `exit.json`, the timestamped pre-training source/
runtime freeze and verification records remain under
`artifacts/reviews/m3-switch-screen`. The frozen timestamp is local provenance,
not external timestamp attestation. A quiet fifteen-minute follow-up finishes verification and sealing, then pauses.

```sh
# Always choose a fresh output directory; do not overwrite completed evidence.
PYTHONPATH=python .venv/bin/python -m pokemon_switch_screen.campaign artifacts/runs/NEW --smoke
.venv/bin/python scripts/seal-switch-screen.py artifacts/runs/NEW --audit-smoke
# Production additionally requires the exact pre-training review admission freeze.
PYTHONPATH=python .venv/bin/python -m pokemon_switch_screen.campaign artifacts/runs/NEW-development
.venv/bin/python scripts/seal-switch-screen.py artifacts/runs/NEW-development
```

## Next gate

Use the frozen panel and unresolved-outcome counts to diagnose learning and horizon
limitations before increasing compute. Broader claims require an expanded admitted
horizon/history/PP scope, fresh scalar/JAX/Showdown and information parity, then a
new preregistered evaluation. Do not tune on these outcomes and present a rerun as
confirmation. The restricted actor remains separate from the playable full-battle
client. Public deployment, competitive rank, stronger opponents, calibrated values,
beliefs/search and full G2/G3/G4 remain open. No final reserved teams, public server
or Prime resources were used.
'''
# Keep prose readable when interpolated counts abut words.
import re
handoff=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', handoff)
handoff=handoff.replace('G 2','G2').replace('G 3','G3').replace('G 4','G4').replace('M 3','M3')
# The replacement above must not alter path identifiers.
handoff=handoff.replace('m 3','m3')
(ROOT/'docs/M3_SWITCH_SCREEN.md').write_text(handoff)
summary=f'''The [M3 categorical learning screen](docs/M3_SWITCH_SCREEN.md) completed all
**31,104 pinned Showdown games** after fresh three-seed training (24,576 updates,
786,432 games,{decisions:,} decisions). The preregistered conservative learning gate
**{result}**. Evaluation has{an['terminal']:,} terminal games and{an['truncated']:,}
explicit unresolved cutoffs; no cutoff is scored as a draw. All476 tests pass.
See the [receipt](artifacts/reports/m3-switch-screen.json). Full G2/G3/G4 and public
ladder strength remain open; final data stay reserved and Prime is unnecessary.
'''
summary=summary.replace('games,','games, ').replace('has','has ').replace('and'+str(an['truncated']), 'and '+str(an['truncated'])).replace('All476','All 476')
pth=ROOT/'README.md';s=pth.read_text();s=s.replace('A separate learning-strength screen remains next.','The separate learning-strength screen is recorded below.')
s=s.replace('Next preregister the separate learning-strength screen.','The separate learning-strength screen is recorded below.')
s+='\n'+summary;pth.write_text(s)
for name,title in [('IMPLEMENTATION_PLAN.md','M3 restricted learning screen — measured development result'),('ACCEPTANCE_GATES.md','M3 restricted learning gate — '+result),('REVIEW_RECORD.md','M3 screen verification and evidence')]:
 pth=ROOT/'docs'/name;s=pth.read_text()
 if name=='IMPLEMENTATION_PLAN.md':s=s.replace('Next preregister the separate restricted learning screen across independent\ntraining seeds and frozen opponents, with explicit handling of nonterminal\ncutoffs.','The separate restricted learning screen across independent training seeds and\nfrozen opponents is now recorded below, with explicit handling of nonterminal\ncutoffs.')
 s+='\n\n## '+title+' (2026-10-11 Eastern)\n\n'+summary.replace('(docs/M3_SWITCH_SCREEN.md)','(M3_SWITCH_SCREEN.md)').replace('(artifacts/reports/','(../artifacts/reports/')
 s+='\nThree fixed seeds and all27 cells are retained. The actor, chance law and PPO kernels\nremain sealed. The primary gate uses conservative unresolved-outcome bounds and\ncomplete seat-pair uncertainty; it does not pool seeds or treat cutoffs as draws.\nThe source-bound476-test regression covers347 unchanged files; nine original-next-\nupdate checkpoints and all recorded action/value outputs are exact. The receipt\nrequires a fresh complete Showdown replay of every evaluation game. Review snapshots,\nverified dispositions, preflight and nine corruption probes are preserved. A completed\nexperiment may fail its learning gate. No public-ladder or full G2/G3/G4 promotion follows.\n'
 s=s.replace('all27','all 27').replace('source-bound476','source-bound 476').replace('covers347','covers 347')
 pth.write_text(s)
marker.write_text(json.dumps({'documents_updated':5,'learning_screen':result,'evaluation_games':an['games'],'report_content_hash':r['content_hash']},indent=2)+'\n')
print(marker.read_text())
