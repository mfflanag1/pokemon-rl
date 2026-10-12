# CPU learner canaries v1

The implementation lives in `python/pokemon_learner`, outside the sealed R1–R7
reference consumers. NumPy 2.5.4 is a separate pinned CPU dependency. Parameters,
advantages, probabilities, gradients and Adam moments use FP32. The gradient-norm
reduction uses FP64 for a stable scalar diagnostic. No GPU or Prime service is used.

`canaries.json` version 2 freezes seeds 11/29/47, sample budgets, thresholds and
hyperparameters before the accepted run. Version 1's delayed task was two independent
bandits and is retained only as a preliminary diagnostic. Version 2 uses a shared
terminal comparison: the sole player choosing its designated action wins; equal
correctness is a draw. Own rewards are win=1, draw=.5, loss=0. Both roots learn
different optimal actions; delayed forced decisions are interleaved asymmetrically.
The one-step task has four legal actions. The delayed task has three legal root
actions and one legal action at each forced step. No exploration or auxiliary
penalty and no advantage normalization is applied.

## Mathematical boundary

GAE uses explicit forward links within a `(player, episode)` trajectory. Terminal
rows require next=-1 and zero bootstrap. A nonterminal cut has next=-1 and an
explicit same-player frozen value. Recurrence stops at cuts. Terminal reward is
not negated merely because the next tape row belongs to the other player.

The CPU model is a six-state categorical policy table and scalar value table.
PPO minimizes the mean negative clipped surrogate plus .5 times half-MSE value
loss, with an optional entropy term tested by finite differences (zero during
learning canaries). Single-legal-action rows retain their scalar surrogate/value
loss but have zero actor gradient and entropy. Invalid actions, empty masks,
nonfinite values and stale behavior versions reject.

Two shuffled epochs use minibatches of at most 256 rows. The delayed buffer has
384 rows, so its last minibatch is 128; each minibatch uses its own mean. Adam
updates parameters with norm clipping, epsilon 1e-8, betas .9/.999 and fixed LR .03.
These aggressive diagnostic settings are not the proposed Pokemon training recipe.

## Checkpoint boundary

Only `completed-update-whole-episodes` is supported. No live game, pending request,
partial buffer, bootstrap cut or minibatch cursor exists at this boundary. Actor
and learner weights coincide; behavior version equals completed updates. State
includes all parameters, Adam moments/step/configuration, sample and shuffle PCG64
states, task/plan, counters, and source/runtime provenance. Initialization RNG is
consumed only at construction; initialized arrays are saved exactly.

Checkpoints use explicit little-endian FP32 bytes encoded as hex, in a separate
versioned finite JSON envelope with a SHA-256 checksum. This is intentionally not
the integer-only battle-data contract. A sibling `.partial` is fsynced before
atomic replacement and parent-directory fsync. A stale partial file causes an
explicit error: preserve it under a different filename, restore the last complete
checkpoint, and then resume saves. Never promote partial writes to checkpoints.

Resume checks require exact next samples, log probabilities, returns, masks,
updates, parameters, optimizer state and counters on the same runtime/platform.
Crashes during collection and after one optimizer minibatch roll back to the last
completed checkpoint. Tests count discarded decisions/optimizer steps and reproduce
their work; they do not claim continuation at the crash point. Runtime, source,
configuration, checksum or boundary-counter drift rejects.

GAE's cut formula and perspective linkage are checked by hand fixtures. The current
collector deliberately collects whole episodes; end-to-end persistent-game cuts
and corresponding live-state checkpoint support remain future work. There is no
Pokemon policy architecture, online Pokemon training or learned-strength claim.

## Primary algorithm and runtime references

- [PPO paper](https://arxiv.org/abs/1707.06347): clipped surrogate objective.
- [GAE paper](https://arxiv.org/abs/1506.02438): discounted TD-residual estimator.
- [NumPy RNG compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html): exact stream compatibility depends on matching build, calls and environment.
