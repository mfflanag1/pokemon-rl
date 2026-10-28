# M1: JAX autodiff learner and observation-only collector

**Implemented on local CPU:** PPO/GAE and Adam parity against the sealed NumPy
learner, the original six learning canaries, and an observation-only collector for
the unchanged fixed Rattata duel. Exact checkpoint continuation is supported at
completed whole-episode updates. **190 regression tests pass: 63 Node and 127 Python.**
**Full G2 and G3 remain open.** No Prime resources,
GPU or reserved final-evaluation outcomes were used.

## Numerical result

The new `python/pokemon_jax_learner` package is additive. Prior NumPy learner,
JAX simulator, reference modules, dependency pins and evidence remain unchanged.
The frozen experiment plan is `contracts/jax-learner-v1/plan.json`.

| Check | Fixed budget | Result |
|---|---:|---|
| Masked PPO loss, diagnostics and logits/value gradients | 96 cases, 3 seeds | Within atol 3e-6 + rtol 3e-5 |
| Linked GAE with interleaved players, terminals and own-value cuts | 96 cases | Within the same tolerance |
| Accumulated Adam parameters and both moment tensors | 384 updates | Within tolerance |
| Gather/scatter table gradients and accumulated learner updates | 96 updates | Within tolerance |
| G1 bandit and delayed shared-outcome canaries | Both tasks × all 3 original seeds; unchanged full budgets | Every run exceeds 99% optimal-action probability |

Observed maximum absolute differences were 8.94e-8 for logit gradients, 9.54e-7
for GAE advantages, 1.19e-7 for Adam parameters and 9.69e-8 for accumulated table
updates. Every comparison met the predeclared combined absolute/relative tolerance.
Final canary optimal-action probabilities range from 99.60% to 99.77%.

The core uses `jax.value_and_grad`; GAE uses a reverse JAX scan. CPU NumPy is
used for input validation and independent expected values, not as a numerical
fallback for the new learner. Synthetic canaries deliberately retain the original
CPU toy collector while using JAX autodiff updates. The real duel collector uses
the previously sealed JAX environment and the new JAX policy/value model.

PPO uses an explicit inclusive branch at the clipping boundary to match the sealed
reference's chosen subgradient. Naive `minimum`/`clip` autodiff can average gradients
at a tie. Tests cover exact boundary derivatives, both advantage signs, stopped
behavior/advantage/return gradients, forced actions, illegal logits, finite differences
and a separately computed NumPy chain rule for the linear model's weight gradients.
There is no advantage normalization, value clipping or added exploration penalty.

Adam retains reference beta/epsilon/bias-correction semantics. Its global norm is
reduced in FP32, whereas the NumPy oracle uses FP64 for that norm. The declared
numeric tolerance applies; cross-backend updates are not claimed byte-identical.
Nonfinite outputs are rejected before optimizer mutation. The raw compiled kernels
assume admitted FP32 arrays; trainer and campaign admission require CPU and x64 off.

## Collector and information boundary

The correctness model is a linear 59-feature policy/value pair with two actions.
Its features are precisely the G2 semantic player view: eleven header integers and
eight six-integer public-history rows, divided by 105. This is not a proposed final
Pokemon architecture. `ActorInput` carries only features and legal-action masks.
It rejects simulator state and target-bearing carriers. Rows are independent: one
player's input cannot affect another player's logits through batch attention.

Both player views are constructed before either action is sampled. One frozen
parameter snapshot serves the entire collection. Environment chance, action sampling
and minibatch shuffling use distinct named key domains. The root is the seed's
Threefry key; domain tags are `0x454e56` for environment, `0x414354` for action
sampling and `0x534846` for shuffling. Chance uses G2's `sample_kernel` with that
folded environment master. G2's public `chance.sample(seed, ...)` uses an unfolded
master and therefore does not reproduce this collector from the same integer seed. Both submitted decisions
are retained when a KO cancels the second move; the environment correctly withholds
its PP cost and public move reveal. Terminal lanes have action -1 and zero probability,
and are removed from the training tape.

Actor and target records are separate:

- `ActorTape`: features, mask, submitted action, behavior log-probability/value,
  game ID, side and turn. No rewards, future winners, chance draws or hidden state.
- `Targets`: own terminal reward, terminal flag, next index, trajectory identity,
  bootstrap, advantage and return. The winner receives 1; the loser receives 0.

Links always advance to the same player's next decision in the same game. Each
player's last row is terminal with bootstrap zero. There is no alternating-player
sign rule. GAE is independently compared against the NumPy reference. Buffer arrays
are read-only, hash-bound to the frozen policy and version; stale, reused, foreign,
modified and parameter-drifted buffers are rejected. Targets enter the loss only,
with stopped gradients.

The first buffer for each seed is exported as separate `actor.npz` and `targets.npz`
files with a checksum manifest. In-memory training uses admitted `Collection`
objects, not arbitrary NPZ loading. The export is an audit artifact, not a general
untrusted-data training API.

Information tests change the environment seed while holding the initial observations,
parameters and action stream fixed, check latent-chance collisions with identical
public histories, reject privileged carriers, verify zero winner/terminal indicators
on exported actor rows, and check row independence. This is a code/interface boundary,
not a sandbox against deliberately malicious actor code. The fixed public-set profile
still cannot satisfy full G2's hidden-set mutation strata.

## Whole-episode continuation and timing

The campaign performs 8 updates × 64 games for each of seeds 11, 29 and 47:
**1,536 training games and 9,614 submitted decisions**. Each seed is freshly restored from disk before
each of its next seven updates; collections, update diagnostics, parameters, optimizer
moments and counters match exactly. That is **21 disk resumes with compared updates**.
This is deterministic repetition of the same training work, not extra independent games.

Checkpoints include FP32 arrays as exact bytes, both Adam moments and step, policy
version, collection ledger, game counter, plan, source hashes, runtime/compiler settings
and the seed/key addressing contract. Writes use a temporary file, fsync and atomic
replace. Tests cover checksum/source-runtime-plan rejection, invalid moments/counters,
interrupted publication, collection failure before buffer publication, and partial
optimizer updates. Recovery discards the uncommitted collection/update and replays it
from the last valid boundary. Saving during collection or update is forbidden.

This collector finishes every game before updating. It does not support persistent
games, arbitrary rollout cuts, mid-turn resume or partial-minibatch continuation.
The core's cut formula is tested, but that is not an online cut collector.

Each report times actual collection plus optimization, including view projection,
inference, sampling, simulator steps, host transfers, linkage, GAE and shuffling.
It excludes audit replay, buffer serialization and checkpoint writes. These CPU wall
times include compilation/cache effects. Compact GAE arrays and final minibatches
have variable shapes; the implementation does not yet meet G3's fixed-shape/no-new-
compilation soak requirement. Kernel-only G2 throughput must not be substituted for
these complete-cycle measurements or used to justify paid hardware.

The accepted local CPU campaign measured 3.060 s, 1.391 s and 0.840 s of primary
collect/update work for the three seeds respectively: 5.291 s total, approximately
1,817 submitted decisions/s including the recorded compilation/cache effects.
Later seeds reuse compiled shapes, so these are not independent hardware-speed
replicates or a steady-state performance guarantee.

## Reproduction and evidence

```sh
# Existing pinned CPU environment; no new dependencies are needed.
npm test
PYTHONPATH=python .venv/bin/python -m pokemon_jax_learner.campaign artifacts/runs/NEW-learner-run
```

Use a fresh output directory. Runs bind the exact G2 predecessor receipt and verify
that earlier core source and the reserved final dataset remain unchanged. The accepted
campaign, independent reviews, complete regression log and source/evidence archive
are linked by [the completion receipt](../artifacts/reports/m1-jax-learner.json).
The accepted campaign is `artifacts/runs/m1-jax-learner-v2`, result hash
`0fddf9966fbccf92a842d92f9a6292054994fd59a2f191963ccac490c45bc64e`.
The earlier passing v1 is retained; numerical results, canary traces and final
policy hashes are unchanged after the evidence/provenance tightening. To verify
and seal this scope, run `.venv/bin/python scripts/seal-jax-learner.py artifacts/runs/m1-jax-learner-v2`.

Cross-model work used the claude-review skill:
Fable implemented only the numerical core; Codex QA accepted it; independent executed
checks confirmed parity before integration was recorded as accepted. A separate Fable
review covers the actor/collector, training targets and checkpoint boundary. Original
packets and verified dispositions are retained under `artifacts/reviews/m1-jax-learner`.

## Next work

The next engineering step is a fixed-shape complete collect/update loop, followed by
longer resume/soak and memory/recompilation checks on CPU. Separately close the declared
G2 coverage gaps before any policy promotion: expanded player-view reference coverage,
appropriate hidden-information tests for a less trivial admitted profile, and an
explicit decision on exact finite-word Showdown chance mapping. The nominal-law
sampler's tiny known distinction from Showdown remains unchanged.

Then preregister a restricted-policy evaluation against frozen random and damage-
greedy opponents, with multiple training seeds and seat-swapped matches. The current
self-play and toy canaries make no competitive-strength or G3 claim. Prime remains
unnecessary for the next correctness work; consider paid hardware only after a complete
measured workload makes the choice worthwhile. Keep final evaluation data sealed.
