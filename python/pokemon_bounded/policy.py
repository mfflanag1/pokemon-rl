"""M3 actor with a sigmoid value head; identical action law at fixed parameters."""
import jax
import jax.numpy as jnp
from pokemon_switch_engine import policy as original

ActorInput = original.ActorInput
encode = original.encode
initialize = original.initialize  # Zero raw value head means value .5, not zero.


def forward(parameters, obs):
    logits, raw_value = original.forward(parameters, obs)
    return logits, jax.nn.sigmoid(raw_value)


@jax.jit
def act(parameters, obs, keys):
    actions, logp, raw_value = original.act(parameters, obs, keys)
    value = jnp.where(obs.mask.any(axis=1), jax.nn.sigmoid(raw_value), 0)
    return actions, logp, value


def objective(parameters, buffer):
    # Sanitize features BEFORE matmul to prevent NaN padding's 0*NaN gradients.
    obs = ActorInput(jnp.where(buffer.valid[:, None], buffer.features, 0), buffer.mask)
    logits, values = forward(parameters, obs)
    return original.loss(logits, values, buffer.mask, buffer.actions, buffer.old_logp,
                         buffer.advantages, buffer.returns, buffer.valid,
                         clip=.2, value_coef=.5, entropy_coef=.01)
