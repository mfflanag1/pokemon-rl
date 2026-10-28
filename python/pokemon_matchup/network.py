"""Function-preserving action-local public-feature residuals on the F22 flat host."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_shared.learning import FLAT_SHAPES
from .features import CHANNELS, EXTRA, FEATURES

SHAPES = {**FLAT_SHAPES, 'matchup_policy': (CHANNELS,), 'matchup_value': (EXTRA,)}

def extend(p):
    if set(p) != set(FLAT_SHAPES):
        raise ValueError('Expected flat predecessor')
    q = {**p, 'matchup_policy': jnp.zeros(CHANNELS, jnp.float32),
         'matchup_value': jnp.zeros(EXTRA, jnp.float32)}
    z = jax.tree.map(jnp.zeros_like, q)
    opt = (z, z, jnp.int32(0))
    check_state(q, opt)
    return q, opt

def check_state(p, opt):
    if not isinstance(opt, (tuple,list)) or len(opt) != 3:
        raise ValueError('Invalid optimizer state')
    for g in [p,opt[0],opt[1]]:
        if not isinstance(g,dict) or set(g) != set(SHAPES):
            raise ValueError('Parameter inventory')
        for k,shape in SHAPES.items():
            a=np.asarray(g[k])
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('Invalid parameter/moment')
    step=np.asarray(opt[2])
    if step.shape != () or step.dtype != np.int32 or not 0 <= int(step) < 2**31:
        raise ValueError('Invalid counter')
    if any((np.asarray(v)<0).any() for v in opt[1].values()):
        raise ValueError('Negative second moment')

def forward(p,x):
    h=jnp.tanh(x[:,:928]@p['w']+p['b'])
    extra=x[:,928:]
    logits=h@p['pw']+p['pb']+extra.reshape(-1,14,CHANNELS)@p['matchup_policy']
    value=jax.nn.sigmoid(h@p['vw']+p['vb']+extra@p['matchup_value'])
    return logits,value
