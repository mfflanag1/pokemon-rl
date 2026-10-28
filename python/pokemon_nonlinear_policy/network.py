"""Function-preserving width-32 action-local nonlinear policy residual (F29).

The host is the sealed F28 matchup network on the public 1278 channels. The new branch
reads each of the 14 action rows (25 public channels) together with the shared trunk
context and adds a tanh-hidden residual to the policy logits only. No new value branch,
feature, critic head or optimizer. `residual_out` starts at zero, so both arms reproduce
the predecessor's predictions exactly; `control` never evaluates the branch at all.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_matchup.network import SHAPES as MATCHUP_SHAPES
from pokemon_matchup.features import CHANNELS

WIDTH = 32
ARMS = ('control', 'nonlinear')
NEW_SHAPES = {'local_w': (CHANNELS, WIDTH), 'context_w': (64, WIDTH),
              'residual_b': (WIDTH,), 'residual_out': (WIDTH,)}
SHAPES = {**MATCHUP_SHAPES, **NEW_SHAPES}


def check_arm(arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    return arm


def _check_group(g, shapes, what):
    if not isinstance(g, dict) or set(g) != set(shapes):
        raise ValueError(what + ' inventory')
    for k, shape in shapes.items():
        a = np.asarray(g[k])
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid ' + what.lower())


def extend(p, seed):
    """Expand an exact matchup predecessor; identical initialization for both arms."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    _check_group(p, MATCHUP_SHAPES, 'Predecessor parameter')
    rng = np.random.default_rng(int(seed))
    local_w = rng.normal(0, .025, NEW_SHAPES['local_w']).astype(np.float32)
    context_w = rng.normal(0, .025, NEW_SHAPES['context_w']).astype(np.float32)
    q = {k: jnp.asarray(p[k]) for k in MATCHUP_SHAPES}
    q.update(local_w=jnp.asarray(local_w), context_w=jnp.asarray(context_w),
             residual_b=jnp.zeros(NEW_SHAPES['residual_b'], jnp.float32),
             residual_out=jnp.zeros(NEW_SHAPES['residual_out'], jnp.float32))
    z = jax.tree.map(jnp.zeros_like, q)
    opt = (z, z, jnp.int32(0))
    check_state(q, opt)
    return q, opt


def check_state(p, opt):
    if not isinstance(opt, (tuple, list)) or len(opt) != 3:
        raise ValueError('Invalid optimizer state')
    _check_group(p, SHAPES, 'Parameter')
    _check_group(opt[0], SHAPES, 'Parameter/moment')
    _check_group(opt[1], SHAPES, 'Parameter/moment')
    step = np.asarray(opt[2])
    if step.shape != () or step.dtype != np.int32 or not 0 <= int(step) < 2**31:
        raise ValueError('Invalid counter')
    if any((np.asarray(v) < 0).any() for v in opt[1].values()):
        raise ValueError('Negative second moment')


def residual(p, h, rows):
    """Action-local tanh residual: rows (n,14,25) with trunk context h (n,64) -> (n,14)."""
    pre = rows @ p['local_w'] + (h @ p['context_w'])[:, None, :] + p['residual_b']
    return jnp.tanh(pre) @ p['residual_out']


def forward(p, x, arm):
    check_arm(arm)
    h = jnp.tanh(x[:, :928] @ p['w'] + p['b'])
    extra = x[:, 928:]
    rows = extra.reshape(-1, 14, CHANNELS)
    logits = h @ p['pw'] + p['pb'] + rows @ p['matchup_policy']
    value = jax.nn.sigmoid(h @ p['vw'] + p['vb'] + extra @ p['matchup_value'])
    if arm == 'nonlinear':
        logits = logits + residual(p, h, rows)
    return logits, value
