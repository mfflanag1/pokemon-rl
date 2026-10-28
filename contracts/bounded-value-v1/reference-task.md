# Independent NumPy reference ownership

Own only `python/pokemon_bounded/reference.py`. No other file edits. You are not
alone in this codebase; the coordinator owns policy, learner, checkpoints,
campaign and integration tests. Do not revert their work or delegate recursively.

Implement a dependency-free NumPy reference (no JAX imports) with:

- `sigmoid(raw)` -> stable FP32 bounded array including raw +-1000, no overflow.
- `forward(parameters, features)` -> `(logits, values)` for tanh MLP:
  h=tanh(x@w+b); logits=h@pw+pb; value=sigmoid(h@vw+vb).
  Parameters shapes w=(323,32), b=(32,), pw=(32,4), pb=(4,), vw=(32,), vb=().
- `loss_grad(parameters, features, mask, actions, old_logp, advantages, returns,
  valid, *, clip=.2, value_coef=.5, entropy_coef=.01)` -> `(metrics, gradients)`.
  Reuse supplied NumPy `pokemon_learner.core.ppo_loss_grad` on valid rows only,
  then implement analytic chain rule through sigmoid, value/actor heads and tanh.
  Gradients are a dict matching every parameter. Empty valid rows yield exactly
  zero loss/metrics/gradients; invalid rows may contain NaN payloads and action -1,
  and must be excluded before any arithmetic. Inputs/parameters are never mutated.
  Return the existing PPO reference metric keys. Check compatible shapes/dtypes,
  legal actions on valid rows, finite parameters and valid inputs, [0,1] returns.
- This is an oracle only in the smooth categorical region. Explicitly reject
  valid rows whose legal-logit spread exceeds 8 (conservative sufficient bound
  to stay away from the original binary conditional floor). Check this before
  calling the ordinary softmax PPO helper; do not claim floor/clipping parity.
  For invalid padded rows the mask/feature/logit payload must not trigger rejection.
- Add readable docs explaining reference domain and independent chain rule.

Checks selected by coordinator: stable sigmoid endpoints; initial .5 output;
analytic network gradient against finite differences for all six parameter tables;
nonzero value-head and shared-trunk gradients; masked illegal, padding and empty
batch controls; input immutability. Do not train or run battles.
