# F1 — full-Showdown learning vertical slice

Status: two reviewed 16-game engineering canaries completed on 2026-10-15, with real finite parameter updates. This establishes a full-reference collect/update path, not learning improvement or strength. The original failed attempt and the separately preregistered Recharge repair are preserved. Existing sealed engines and evidence remain unchanged.

F1 runs the pinned unmodified Node Showdown engine (`942cda71d14bf0618bd57991708a2e2734b6daba`) and trains a small Python/JAX actor/value model from its player projections. It admits all official Gen9 Random Battle mechanics through the reference engine; the observation encoder is deliberately incomplete and is not a faithful compressed simulator. Full-engine fidelity does not imply competitive policy quality.

## Interface and data boundary

`packages/full-showdown/worker.mjs` is a long-lived local JSON-lines process. Each command has an `id` and receives `{id,ok,data}` or `{id,ok:false,error}`. Supported operations are `hello`, `start` (`run_seed`, `episode_index`, optional `curriculum:true`), `step` (`actions`, `request_ids` keyed by exactly the pending sides), `cutoff`, and `close`. Any failed command destroys the current game; automatic process recovery is unsupported.

The coordinator gets provenance separately from `players.p1` and `players.p2`. An actor receives only its own `full-showdown-player-v1` object, validated by the additive JSON Schema. Node uses the existing `PlayerProjection` channel splitter; neither opponent private requests nor the simulator's raw state/end payload enters the actor. Exact own requests provide six own slots. Six opponent slots hold only identities/events publicly revealed to this player, with unknown slots represented by null. They are ordered by first reveal, not by hidden team position. Illusion replacement repairs the current appearance while preserving prior knowledge of a real disguised teammate.

The new 14-action order is four ordinary request moves, four Tera versions, then six request switch slots. This is unrelated to the sealed four-action contracts. Masks derive from the actual private request and cover waits, forced replacements, Revival Blessing targets and Struggle. Forced pass is coordinator control and never a training row. Both pending choices are validated before either is submitted. An advertised action later rejected as unavailable (for example hidden trapping information) remains a zero-reward actor decision whose next link sees the refreshed request; it is not discarded, marked terminal, or converted into a draw. Other errors abort the episode.

Default teams use the official generator in the explicit `full-showdown-f1/train` namespace. Every complete ordered team passes the frozen R7 identity partition and key gate. Rejected candidates never enter games, actor inputs or returned provenance. Optional C1 plans are readmitted through the same gate. Their fixed prefix runs inside Node before handoff, and realized public evidence is checked; prefix choices never enter PPO rows. C1 scenario IDs and evidence are coordinator metadata only.

## Collection, learning and budget

The CPU FP32 smoke accepts exactly 8 or 16 games, in fixed eight-game cohorts. All games in a cohort use one frozen collecting policy. A cohort is collected to natural outcomes or administrative cutoffs, then performs two full-batch PPO/Adam epochs. The second cohort, when requested, uses the updated policy. No gradient/GAE trace crosses a cohort.

Each player has a separate trajectory within each episode. Its next link skips the other player's decisions and forced waits but includes its own move/replacement/revival/unavailable-choice decisions. Natural outcomes pay win 1, loss 0, or true tie .5 exactly once on that player's final decision. Cutoff happens only at a fresh joint move request at the start of `max_turns`, before collecting or executing that turn's actions, with zero reward, nonterminal flag and the frozen collecting policy's value bootstrap. For example, `max_turns=2` permits turn 1 and cuts at the turn 2 request. The 10,000-decision safety limit raises an error that terminates the entire smoke run; the incomplete episode and cohort do not receive a learning update or successful receipt.

The model has 928 features, a 64-unit tanh trunk, 14 actor logits and a sigmoid value. Tokens use fixed SHA256 feature buckets, with intentional collisions; numeric features include HP fraction, own stats, levels, known boosts, current move PP/power and context. Masking stays separate. The actor uses a 16-leaf binary conditional tree with two padded masked leaves and the sealed binary rare-mass sampler. Stored path log probabilities use the same conditional law as selection. PPO uses clip .2, value coefficient .5, entropy .01, learning rate .001, norm clip 1, gamma 1 and lambda .95. Empty cohorts skip updates. Nonempty updates must be finite and change the parameter hash.

The report separates actor initialization, Node initialization, collection, and update time including compilation. It includes collection throughput, collection-plus-update throughput, total throughput including startup/shutdown, episode provenance, terminal versus cutoff outcomes, phase counts, unavailable-choice counts, optimizer steps, target range and before/after parameter hashes. It does not report Elo, win-rate improvement or a speed comparison to the JAX simulator. `contracts/full-showdown-v1/plan.json` states the engineering profile; it is not a preregistered production receipt.

```bash
node --test tests/full-showdown/*.test.mjs
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python .venv/bin/python -m unittest discover -s tests/full-showdown -p 'test_*.py' -v

# Exact commands of the completed, separately recorded engineering canaries:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python .venv/bin/python -m pokemon_full.smoke --games 16 --seed 20261015 --max-turns 60
# Add --curriculum for the separately labeled C1 mixture/prefix integration.
```

## Tests and limitations

Focused Node tests cover request masks, wait/pass/revival/Struggle, foreign-private/split rejection, six-slot public roster, Illusion correction, reserve memory, train-team admission and stale requests. A fixed six-member, train-admitted Arena Trap fixture verifies actual reference rejection of an advertised switch, request-ID refresh, removal of switch actions and successful continuation. Python replays those real replies through the collector with scripted action selection and checks that the rejected row survives, links to the same player's retry, and receives nonterminal cutoff bootstrap. Python tests also cover strict actor projection boundaries, public-information sensitivity, forced-wait/same-player GAE, terminal reward assignment, cutoff bootstrap, dyadic mask/rare-tail behavior and a real finite parameter update. Two short integration tests check a one-turn JSONL collection/cutoff and a C1 scripted handoff. These integration tests are engineering fixtures, not completed training canaries.

The public reducer is intentionally lossy: it omits many volatiles, ability/item reveal annotations embedded in other events, exact opponent PP, delayed attacks, durations, full history, and certain boost-copy/reset interactions. It retains only explicitly parsed reveals and approximate contextual summaries; missing information limits policy learning. Public identity inference can remain ambiguous, including repeated nicknames or unusual form/copy interactions; overfilling six slots fails loudly. Source engine legality remains authoritative.

Updates occur after whole-trajectory cohorts, not during games. There is no checkpoint resume, live-game persistence, vectorized Node collector, off-policy buffer, league, search, opponent model, public server connection, Prime job or resumable production model artifact. The model starts from random weights and self-plays both seats. It has no pretrained strategic knowledge. This slice proves that the full reference game can feed a real learning update under the stated boundary contract; strength and sustained reliability require later measured work.

## Measured engineering canaries — 2026-10-15

The original official attempt exited 1 at request 731 with missing move metadata; it has no completed result. A real Hyper Beam/Recharge fixture reproduced that error. The adapter now handles Recharge explicitly and rejects other nonexistent Dex moves. The exact original seed and 16-game budget then completed in a separately recorded replacement; no seed or outcome selection occurred. The raw failing request was not retained, so attribution of that particular request to Recharge remains inferred.

| Run | Games | Terminal / cutoff | Decisions | Collection decisions/s | Collect + update decisions/s | Startup-inclusive decisions/s |
|---|---:|---:|---:|---:|---:|---:|
| official | 16 | 9 / 7 | 1,675 | 928.0 | 509.0 | 354.4 |
| mixed | 16 | 6 / 10 | 1,859 | 932.4 | 544.0 | 384.5 |

Each run used two eight-game cohorts and four Adam steps; every update was finite and changed parameters. Targets remained in [0,1]. Timings include update compilation, with initialization/shutdown separated above; they are short single-process measurements on the local CPU, not steady-state scaling estimates. The mixed run used all ten targeted scenarios plus six official samples. Cutoffs at the start of turn 60 remain nonterminal bootstraps. No result here measures improvement against an opponent.

The final source-bound regression passed 567 tests (260 Node, 307 Python; 431 source/configuration files). The original and repair reviews, request fixtures, exact commands, exits, metrics and source/runtime hashes are preserved under `artifacts/reviews/f1-c1-full-showdown/` and `artifacts/reviews/f1-c1-repair-v2/`. Receipt: `artifacts/reports/f1-c1-canary.json`.
