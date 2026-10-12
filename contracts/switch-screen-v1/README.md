# Preregistered restricted categorical learning screen

Freeze this plan and implementation before development outcomes are collected.
Use the unchanged sealed switching learner, chance law, twelve-turn cap and
24 constructed teams. Fresh independent seeds 1009/2027/3037 each train exactly
8,192 updates (262,144 games); retain updates 0/2,048/8,192. No seed selection,
hyperparameter tuning, outcome-dependent extension or selective cell reruns.
The old engineering soak checkpoint is not used as a candidate or opponent.

All evaluation games run on unpatched pinned Showdown. Each of 576 pairs covers
one ordered candidate/opponent team combination from the full 24 by 24 grid.
Both seats use the same mechanics seed with the same candidate and opponent teams
swapped between seats. Checkpoints share pair IDs and RNG addresses; different
actions do not promise identical realized chance events. Training, evaluation and
smoke namespaces are separate. Each seed has its own evaluation stream. Actor
inputs are solely the receiving player's protocol-derived compact view and mask.
Pair/seat/seed/opponent metadata addresses randomness only; hidden state never
enters candidate or baseline features. Waits generate no action or value sample.

Frozen opponents: random legal (equal logits under the sealed finite-word
categorical law); damage-greedy (highest expected immediate damage against the
public active species using only own exact HP and stats; average all 16 rolls and
the 1/24 critical law, lowest-index move on ties, forced switches only); and the
same seed's frozen update-zero policy. A game is stopped only at a terminal or the
sealed turn-13 joint request. Every terminal win/loss and every nonterminal cutoff
is retained. A cutoff is never called a draw, loss or terminal outcome.
Ties are outside the admitted mechanics: no recoil, residual damage or simultaneous
faint effect exists, PP cannot exhaust by turn twelve, and Showdown's long-battle
tie limit is unreachable. An unexpected tie fails the run and preserves evidence.

For each cell report W/L/C and the unresolved outcome interval [W/N,(W+C)/N].
This interval bounds any eventual binary outcome assignment; it does not simulate
or validate a longer battle. The pair is the unit for uncertainty: average seats
first. Conditional on each frozen policy and the fixed team grid, model independent
per-pair pseudo-random streams as independent chance/policy draws. Hoeffding's
one-sided bounds allow nonidentical team strata. For the three final random-opponent
cells, use delta=.05/6 for each of two bounds: secured-win mean minus
sqrt(log(1/delta)/(2*pairs)); and worst-case paired improvement mean minus
2*sqrt(log(1/delta)/(2*pairs)). The latter is final win indicator minus initial
(win OR cutoff) indicator, averaged within pair. Its range is [-1,1]. Thus this
gate cannot pass by treating initial cutoffs as losses. The intervals are
conservative under the independent-stream assumption; deterministic PRNG seeds
are not a statistical proof of independence. Other cells are descriptive only.

Every seed must have final secured-win fraction >=.55, its adjusted lower bound
>.5, and adjusted worst-case improvement lower bound >0. All seeds must pass;
do not pool seeds or select a winner. A completed experiment may fail this screen.
Heavy censoring can make the result inconclusive even if observed wins improve.
Report both observed secured-win change and the conservative improvement bound.
No terminal-only win-rate promotion, critic-based cutoff adjudication or invented
draws. No claim of calibrated value, full-format strength, public rank or G2/G3/G4.

32-game inference batches; 27 cells and 31,104 reference games. Smoke uses one seed,
updates 0/2, four pairs and all opponents (48 games), disjoint mechanics/action keys,
and cannot satisfy the development gate. Persist full final player protocols,
every action/log likelihood/value, frozen checkpoint and original-next-update
fingerprints. Audit all player-view prefixes, actions and outcomes, and replay every
game from its original seed on unpatched Showdown with exact view/protocol equality, then bind the
complete experiment to its preregistration, unchanged source/runtime/reservation,
passing regression and independent review. Never overwrite a prior run.
The timestamped source/runtime freeze is required and hash-bound before production
training starts; it is a local provenance record, not external timestamp attestation.

Local CPU only. Final reserved teams remain unused. Prime is unnecessary.
