"""Independent NumPy oracle for the bounded-value policy network.

This module deliberately covers only the smooth categorical PPO region.  It
does not reproduce the production actor's rare-mass floor: ``loss_grad``
rejects examples whose legal logits span more than eight before using the
ordinary softmax PPO oracle.

The network derivatives below are an explicit chain rule through the sigmoid
value head, the actor and value affine heads, and the shared tanh trunk.  No
autodiff or finite-difference gradients are used by the implementation.
"""

import numpy as np

from pokemon_learner.core import ppo_loss_grad


F32 = np.float32
_SHAPES = {
    "w": (323, 32),
    "b": (32,),
    "pw": (32, 4),
    "pb": (4,),
    "vw": (32,),
    "vb": (),
}
_METRICS = (
    "loss",
    "actor_loss",
    "value_loss",
    "entropy",
    "max_ratio_error",
    "clip_fraction",
    "approx_kl",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _parameters(parameters):
    _require(isinstance(parameters, dict) and set(parameters) == set(_SHAPES),
             "Parameter keys")
    for name, shape in _SHAPES.items():
        value = parameters[name]
        _require(isinstance(value, np.ndarray) and
                 value.dtype == F32 and value.shape == shape,
                 "Parameter FP32 shapes")
        _require(np.all(np.isfinite(value)), "Nonfinite parameters")


def sigmoid(raw):
    """Return a numerically stable FP32 logistic sigmoid."""
    raw = np.asarray(raw, dtype=F32)
    result = np.empty_like(raw, dtype=F32)
    positive = raw >= F32(0)
    result[positive] = F32(1) / (F32(1) + np.exp(-raw[positive]))
    exponential = np.exp(raw[~positive])
    result[~positive] = exponential / (F32(1) + exponential)
    return result


def forward(parameters, features):
    """Evaluate the tanh policy trunk and bounded sigmoid value head."""
    _parameters(parameters)
    features = np.asarray(features)
    _require(features.dtype == F32 and features.ndim == 2 and
             features.shape[1] == 323, "Feature FP32 shape")
    _require(np.all(np.isfinite(features)), "Nonfinite features")

    hidden = np.tanh(features @ parameters["w"] + parameters["b"])
    logits = hidden @ parameters["pw"] + parameters["pb"]
    values = sigmoid(hidden @ parameters["vw"] + parameters["vb"])
    _require(np.all(np.isfinite(logits)) and np.all(np.isfinite(values)),
             "Nonfinite network output")
    return logits, values


def loss_grad(parameters, features, mask, actions, old_logp, advantages,
              returns, valid, *, clip=.2, value_coef=.5, entropy_coef=.01):
    """Compute PPO metrics and analytic gradients on unpadded rows only.

    Invalid rows are removed before their payload is inspected or used in any
    arithmetic, so NaN padding and the conventional action ``-1`` are inert.
    The returned gradient dictionary has the same six keys and shapes as the
    parameter dictionary.
    """
    _parameters(parameters)
    features, mask, actions = (np.asarray(features), np.asarray(mask),
                               np.asarray(actions))
    old_logp, advantages, returns = (np.asarray(old_logp),
                                     np.asarray(advantages),
                                     np.asarray(returns))
    valid = np.asarray(valid)

    _require(features.dtype == F32 and features.ndim == 2 and
             features.shape[1] == 323, "Feature FP32 shape")
    n = features.shape[0]
    _require(mask.shape == (n, 4) and mask.dtype == np.bool_,
             "Policy mask shape/dtype")
    _require(actions.shape == (n,) and actions.dtype.kind in "iu",
             "Action shape/dtype")
    _require(all(a.shape == (n,) and a.dtype == F32
                 for a in (old_logp, advantages, returns)),
             "PPO FP32 vector shapes")
    _require(valid.shape == (n,) and valid.dtype == np.bool_,
             "Valid shape/dtype")
    _require(0 < clip < 1 and value_coef >= 0 and entropy_coef >= 0,
             "PPO coefficients")

    indices = np.flatnonzero(valid)
    gradients = {name: np.zeros_like(value)
                 for name, value in parameters.items()}
    if indices.size == 0:
        return {name: 0.0 for name in _METRICS}, gradients

    x = features[indices]
    legal = mask[indices]
    selected = actions[indices]
    old = old_logp[indices]
    advantage = advantages[indices]
    target = returns[indices]
    _require(np.all(np.isfinite(x)) and np.all(np.isfinite(old)) and
             np.all(np.isfinite(advantage)) and np.all(np.isfinite(target)),
             "Nonfinite valid learner data")
    _require(np.all(legal.any(axis=1)), "Empty legal-action mask")
    _require(np.all((selected >= 0) & (selected < 4)),
             "Action index on valid row")
    _require(np.all(legal[np.arange(indices.size), selected]),
             "Known-forbidden action")
    _require(np.all((target >= F32(0)) & (target <= F32(1))),
             "Bounded value targets")

    hidden = np.tanh(x @ parameters["w"] + parameters["b"])
    logits = hidden @ parameters["pw"] + parameters["pb"]
    values = sigmoid(hidden @ parameters["vw"] + parameters["vb"])
    _require(np.all(np.isfinite(logits)) and np.all(np.isfinite(values)),
             "Nonfinite network output")
    legal_max = np.max(np.where(legal, logits, F32(-np.inf)), axis=1)
    legal_min = np.min(np.where(legal, logits, F32(np.inf)), axis=1)
    _require(np.all(legal_max - legal_min <= F32(8)),
             "Actor logits outside smooth reference domain")

    metrics, grad_logits, grad_values = ppo_loss_grad(
        logits, values, legal, selected, old, advantage, target,
        clip=clip, value_coef=value_coef, entropy_coef=entropy_coef)

    grad_raw_value = grad_values * values * (F32(1) - values)
    gradients["pw"] = hidden.T @ grad_logits
    gradients["pb"] = grad_logits.sum(axis=0, dtype=F32)
    gradients["vw"] = hidden.T @ grad_raw_value
    gradients["vb"] = np.asarray(grad_raw_value.sum(dtype=F32), dtype=F32)

    grad_hidden = (grad_logits @ parameters["pw"].T +
                   grad_raw_value[:, None] * parameters["vw"][None, :])
    grad_pre_tanh = grad_hidden * (F32(1) - hidden * hidden)
    gradients["w"] = x.T @ grad_pre_tanh
    gradients["b"] = grad_pre_tanh.sum(axis=0, dtype=F32)
    _require(all(g.dtype == F32 and np.all(np.isfinite(g))
                 for g in gradients.values()), "Nonfinite gradients")
    return metrics, gradients
