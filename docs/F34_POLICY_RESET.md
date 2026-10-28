# F34: structured policy and an external opponent benchmark

The accepted eight-hour goal began October 27, 2026 at 18:00:49 UTC. Its four-hour
checkpoint is 22:00:49 UTC and deadline is October 28 at 02:00:49 UTC (10:00:49 p.m.
EDT October 27). The objective is a qualified richer policy, a fixed three-seed
learning screen and a pinned local Foul Play comparison. A positive learning result
is not guaranteed. The rationale is in [the strategy review](STRATEGY_REVIEW_2026_10_27.md).

## Candidate and fixed screen

All three F32 nonlinear hosts (7103, 8209, 9311) initialize one candidate. Its
structured species, moves, items, abilities, public state and observable event history
feed a learned residual policy/value branch. Zero output projections preserve the
original predictions exactly at initialization. All parameters can train; Adam resets
to zero. The model has 261,559 parameters and a 4,224-wide packed observation. The
first 1,278 features preserve the predecessor representation. Opponent hidden state
is not exposed. The ordered public history is an input window, not recurrent state.

The fixed plan is `contracts/policy-reset-v1/plan.json`: 512 cohorts per seed, 1,024
selected rows each, two epochs and 256-row minibatches. Each cohort selects 256
self-play rows, 256 rows against its frozen same-seed F32 host, and 512 against the
unchanged tactical opponent. Complete trajectories and GAE precede row selection.
This totals 1,536 cohorts, 12,288 optimizer steps, 1,572,864 selected rows and
3,145,728 presentations. Fresh train/development namespaces and the F33 exposure
quarantine are enforced. No final reserved evaluation is used.

Initial and final local evaluation comprises 256 tactical, 128 base-power and 128
random games per seed and endpoint: 3,072 games total. All pairs swap seats and use
common fresh schedules. The primary gate requires each seed's tactical cutoff-aware
point improvement to reach five percentage points with an approximate adjusted
bootstrap interval above zero, no detected local-opponent regression, and final
cutoffs at most 5%. All outcomes, including unresolved cutoff bounds, remain reported.

## External assessment

`contracts/policy-reset-v1/external.json` fixes 32 paired seat-swaps per seed and
endpoint, 384 games total. Foul Play commit
`8f541f80f91f960603673796559f796b10ccf0fb` uses poke-engine 0.0.48 with Tera support
and pinned public Random Battles set priors. Its configuration is 100 ms, one search
thread and parallelism one; upstream samples two 100-ms or four 50-ms worlds, so
actual engine search is roughly 200 ms per decision plus overhead. Python seeds
are recorded, but Rust MCTS randomness and wall-time stopping are not reproducible.
Recorded actions can still be replayed through Showdown.

Foul Play receives its own request and player-visible protocol only, through an
offline adapter. It is never a training opponent or demonstration source in this
screen. This is a local, small-budget held-out-opponent development comparison,
not public ladder Elo. Upstream prior parsing has a known Ditto/Tera-token warning;
the pinned opponent is a useful external baseline, not a perfect oracle.

## Verification and interpretation

Focused checks cover initial prediction parity, public information, legal actions,
checkpoint state, complete-game collection and original-next-update recovery.
Verification samples 18 training games, 36 local evaluation games, 12 external
recorded-action games and six updates. Source snapshots and measured evidence are
hashed; this is sampled verification rather than an exhaustive predecessor replay.

This candidate changes representation, amount of training, Adam state and the
opponent mixture together. Results cannot identify an isolated architecture effect.
Tactical is trained on; recurring official generation does not establish held-out-team
or public strength. Full G2/G3/G4 remain open. No promotion, public server, deeper
search, Prime, cloud spending or GitHub publication is authorized here.

Metamon was assessed briefly but target-compatible Random Battles inference was
not established. The independent review rejected several overconfident compatibility
claims; the bounded disposition is retained in the F34 review directory. This is a
decision to keep today's single implementation path, not a claim that transfer is
impossible.

## Measured completion

The single workflow finished October 27 at 20:41:13 UTC and exited zero after
7,199.13 seconds (about two hours of scientific execution). Implementation,
qualification and review preceded launch at 18:41:14 UTC. The fixed comparison
completed within the eight-hour goal; a stronger policy has not been established.

All three local learning gates were **not passed**. Every planned comparison is
inconclusive under the prespecified approximate family-adjusted paired bootstrap.
No statistically detected regression is not proof that strength was preserved.

| Seed | Tactical wins, initial → final / 256 | Tactical gain | Adjusted interval | Foul Play wins, initial → final / 64 |
|---|---:|---:|---:|---:|
| 7103 | 92 → 112 | +7.81 pp | [−4.30, +20.31] pp | 6 → 5 |
| 8209 | 91 → 95 | +1.56 pp | [−8.98, +12.24] pp | 1 → 2 |
| 9311 | 92 → 104 | +4.69 pp | [−5.47, +14.84] pp | 4 → 6 |

Only seed 7103 clears the five-point tactical threshold; none clears the positive
adjusted interval requirement. Foul Play totals changed from 11/192 to 13/192
(5.73% to 6.77%), a descriptive pooled change, not an additional significance test.
The external per-seed gain intervals are [−14.06, +9.38], [−6.25, +9.38] and
[−4.69, +12.50] percentage points. Transfer improvement remains unestablished.
Base-power wins were 107→109, 105→110 and 110→115 per 128; random wins were
124→127, 127→124 and 123→125. The random decline for seed 8209 is retained.

Actual training included **73,416 complete games and 3,707,314 decisions**.
There were 2,184,173 eligible learner rows, 1,572,864 selected rows and 611,309
excluded by selection. Frozen and tactical opponent decisions never entered the
eligible learner pool. All 1,536 cohorts, 12,288 optimizer steps and 3,145,728 row
presentations completed. Local evaluation included 3,072 games and 151,335
decisions; external evaluation included 384 games and 18,277 decisions.
All these games ended naturally: zero cutoffs and zero ties. Full per-seed/stream
accounting is in the [measured results](../artifacts/reviews/f34-policy-reset/measured-results-v1/results.md).

Training also recorded 1,351 unavailable-choice responses (394/515/442 by seed),
across 903 games: 347 in self-play, 548 against frozen F32 and 456 against tactical.
Local evaluation recorded another 16 initially and 16 finally. This counter counts
submitted actions rejected by Showdown on either seat; it is not a count of options
removed by a mask. A rejected controlled-side attempt remains in its trajectory
with zero immediate reward, followed by a fresh decision on the refreshed request.
Opponent attempts count toward feedback but never enter learner rows. Terminal
reward is assigned once, GAE precedes selection, and no rejection penalty,
fallback substitution or replacement game is applied.

Only counts are retained for nonsampled training games. All 18 sampled training
games and the sampled local/external pairs had zero rejections, so none of the
66 sampled replays independently exercises this path.
We cannot classify every training rejection as trapping, attribute all to the
learner, or count how many rejected-attempt rows survived selection. The full
external traces support the more specific nine-attempt diagnosis below. These
limits and counts supplement the immutable generated report.

Terminal-only, equal-game value MSE was mixed. Tactical values were
.181318→.170172, .155283→.151617 and .149783→.171093; Foul Play values were
.130686→.079095, .091547→.084199 and .085412→.111767. Four of nine local cells
improved and five worsened. Trajectories changed across endpoints, so this is not
an isolated critic comparison or evidence of calibrated values in unseen play.

The training stage took 4,410.87 seconds: 4,253.06 seconds collecting training and
local evaluation, 93.32 seconds in optimizer calls including compilation, and
0.35 seconds in sampled update recovery. Recorded collection subcomponents include
2,729.14 seconds of encoding/inference/synchronization and 1,164.42 seconds of
worker-barrier time; collection is not pure simulator time. Useful selected-row
throughput was 1.284 million per training-stage wall hour. Peak parent RSS was
908.20 MiB, excluding children. These are CPU wall durations, not summed CPU time.
External initial/final panels took 1,404.78/1,351.00 seconds. Mean Foul Play
request latency was about 287 ms. Initial/final seed 9311 records contain six/three
rejected **learner** switch attempts: the advertised action mask permitted the
switch, then Showdown revealed that the active Pokémon was trapped. The refreshed
request cleared switch choices, with no fallback substitution or discarded game.
All nine occurred in episode 2's four initial/final seat games, all losses. Zero
illegal actions means zero advertised-mask violations, not universal simulator
acceptance. These rejection traces were inspected; episode 2 was not in the
preregistered external replay sample.

All optimizer metrics were finite. Last-epoch mean pre-update minibatch KL was
.000778/.000928/.000863 and clipping fraction .004143/.005585/.005365 across
the three seeds. These are descriptive minibatch statistics, not full-batch
post-update KL or additional learning gates.

## Verification and evidence

All 53 focused checks passed before launch. Independent sampled verification
completed in 25.94 seconds: 18 fresh training games, 36 local evaluation games,
12 external recorded-action replays, and six original-next-update recoveries
recomputing 48 optimizer steps. Nine checkpoint round trips were also checked.
External replay checks recorded Showdown actions/outcomes; it does not reproduce
native MCTS or independently recompute every external policy decision.

The [completion receipt](../artifacts/reports/f34-policy-reset.json) has SHA256
`e88ffc1eeacf74cfa1afd5c3b112335fe60bfbd115d70648104eb1cd255d076a` and canonical
content hash `69947424e14b96f651e6bc8de1022c5085bef42cb591c39bcc2d7b9d8884bb51`.
The [independent postcheck](../artifacts/reviews/f34-policy-reset-postcheck.json)
passed: byte-identical receipt copies, all 870 admitted files in three source
locations, all 4,883 evidence hashes, exact stage/seed/count inventories and
completion linkage. A separate read-only results review recomputed all 12 local
and external contrasts from raw endpoint records, checked all update ledgers and
all external record/action/value bindings, and inspected the rejected requests
([review record](../artifacts/reviews/f34-policy-reset-results-review.json)). Hash integrity
and sampled replay support execution fidelity; they do not establish positive learning.

The final Claude Opus 5 snapshot review found no material error in the headline
conclusion. Its reporting findings were independently verified and resolved in
this handoff; frozen reports remain unchanged. The [final review disposition](../artifacts/reviews/f34-policy-reset-final-claims-review.json)
records the added unavailable-choice counts, precise rejection attribution and
source checks for figures outside the review packet.

## Decision after this screen

Retain this implementation and external benchmark as a reproducible development
baseline; do not promote the endpoint policies or extend the completed run. The
positive tactical point gains are uncertain, and the external gap remains large.
These data do not demonstrate a representation ceiling, establish that history is
ineffective, or identify a uniquely optimal next algorithm.

My next research priority is one bounded teacher-assisted RL comparison using
trusted local trajectories from stronger play, with a fresh confirmation panel and
an opponent family excluded from training. This is a hypothesis to test, not a
remedy established by F34. Teacher-assisted RL has precedent in
[Kickstarting](https://arxiv.org/abs/1803.03835); it does not guarantee transfer here,
and earlier Maxcalibur imitation failures still apply. If Foul Play becomes a
teacher, it ceases to be a held-out opponent for that next experiment. Keep teacher
targets separate from on-policy PPO data and cap acquisition effort. No such
experiment was launched under this goal. The existing strategy review explains
the broader options and evidence limits.
