"""Compact typed entities and ordered public events added to the frozen F32 host.

Both output projections start at zero. The old actor and value-prelogit arithmetic
is retained verbatim, while every host and new parameter remains trainable on fresh
on-policy trajectories. This is a new model with a reset optimizer, not a resumed F32
optimizer or a claim isolating representation from additional training.
"""
import jax
import jax.numpy as jnp
import numpy as np

from pokemon_nonlinear_policy.network import SHAPES as HOST_SHAPES
from .features import CATEGORICAL_SIZES, FIELD_SHAPES, FIELD_SLICES as SLICES, MON_NUMERIC_FIELDS

EMBEDDINGS = {'species': 16, 'ability': 8, 'item': 8, 'type': 4, 'move': 16,
              'status': 4, 'event_kind': 8, 'event_side': 3, 'event_aux': 8}
ENTITY_WIDTH, EVENT_WIDTH, HISTORY_WIDTH, CONTEXT_WIDTH, ACTION_WIDTH = 48, 32, 64, 128, 64
EVENT_FIELDS = ('event_kind', 'event_side', 'event_side', 'move', 'species',
                'ability', 'item', 'type', 'status', 'event_aux')
MON_INPUT = (EMBEDDINGS['species'] + EMBEDDINGS['ability'] + EMBEDDINGS['item']
             + 5 * EMBEDDINGS['type'] + 4 * EMBEDDINGS['move'] + EMBEDDINGS['status'] + 24)
EVENT_INPUT = sum(EMBEDDINGS[k] for k in EVENT_FIELDS) + 4
CONTEXT_INPUT = 6 * ENTITY_WIDTH + HISTORY_WIDTH + 2 * EMBEDDINGS['move'] + 64
ACTION_INPUT = EMBEDDINGS['move'] + EMBEDDINGS['type'] + 8 + 2 * ENTITY_WIDTH
NEW_SHAPES = {**{'sp_' + k: (CATEGORICAL_SIZES[k], d) for k, d in EMBEDDINGS.items()},
    'sp_mon_w': (MON_INPUT, ENTITY_WIDTH), 'sp_mon_b': (ENTITY_WIDTH,),
    'sp_event_w': (EVENT_INPUT, EVENT_WIDTH), 'sp_event_b': (EVENT_WIDTH,),
    'sp_history_w': (FIELD_SHAPES['event_mask'][0] * EVENT_WIDTH, HISTORY_WIDTH),
    'sp_history_b': (HISTORY_WIDTH,),
    'sp_context_w': (CONTEXT_INPUT, CONTEXT_WIDTH), 'sp_context_b': (CONTEXT_WIDTH,),
    'sp_action_w': (ACTION_INPUT, ACTION_WIDTH),
    'sp_action_context_w': (CONTEXT_WIDTH, ACTION_WIDTH), 'sp_action_b': (ACTION_WIDTH,),
    'sp_policy_out': (ACTION_WIDTH,), 'sp_value_out': (CONTEXT_WIDTH,)}
SHAPES = {**HOST_SHAPES, **NEW_SHAPES}
PARAMETER_COUNT = sum(int(np.prod(s)) for s in SHAPES.values())


def _group(group, shapes, label):
    if not isinstance(group, dict) or set(group) != set(shapes):
        raise ValueError(label + ' inventory')
    for name, shape in shapes.items():
        value = np.asarray(group[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(label + ' shape/dtype/finite: ' + name)


def check_state(parameters, optimizer):
    _group(parameters, SHAPES, 'Parameter')
    if not isinstance(optimizer, (tuple, list)) or len(optimizer) != 3:
        raise ValueError('Optimizer inventory')
    _group(optimizer[0], SHAPES, 'First moment')
    _group(optimizer[1], SHAPES, 'Second moment')
    if any(np.any(np.asarray(a) < 0) for a in optimizer[1].values()):
        raise ValueError('Negative second moment')
    step = np.asarray(optimizer[2])
    if step.shape != () or step.dtype != np.int32 or not 0 <= int(step) < 2**31:
        raise ValueError('Optimizer counter')


def extend(host, seed):
    """Copy one admitted F32 parameter state; reset every Adam moment and step."""
    _group(host, HOST_SHAPES, 'F32 host')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise ValueError('Unsigned initialization seed required')
    rng = np.random.default_rng(int(seed))
    parameters = {k: jnp.asarray(v) for k, v in host.items()}
    for name, shape in NEW_SHAPES.items():
        if name.endswith('_b') or name.endswith('_out'):
            value = np.zeros(shape, np.float32)
        else:
            scale = .08 if name in {'sp_' + k for k in EMBEDDINGS} else 1 / np.sqrt(shape[0])
            value = rng.normal(0, scale, shape).astype(np.float32)
        parameters[name] = jnp.asarray(value)
    zero = jax.tree.map(jnp.zeros_like, parameters)
    optimizer = (zero, zero, jnp.int32(0))
    check_state(parameters, optimizer)
    return parameters, optimizer


def _field(x, name):
    return x[:, SLICES[name]].reshape((x.shape[0], *FIELD_SHAPES[name]))


def _embedding(parameters, domain, ids):
    return parameters['sp_' + domain][ids.astype(jnp.int32)]


def _pool(h, present):
    count = jnp.maximum(present.sum(1, keepdims=True), 1)
    average = (h * present[..., None]).sum(1) / count
    maximum = jnp.max(jnp.where(present[..., None] > 0, h, -1e9), 1)
    maximum = jnp.where((present.sum(1) > 0)[:, None], maximum, 0.)
    return average, maximum


def structured_context(parameters, x, host_hidden):
    """Only tensorized player observations enter the actor and critic branches."""
    batch = x.shape[0]
    numeric = _field(x, 'mon_numeric')
    mon_inputs = [_embedding(parameters, name, _field(x, field)) for name, field in
                  [('species', 'species'), ('ability', 'ability'), ('item', 'item')]]
    mon_inputs.extend([
        _embedding(parameters, 'type', _field(x, 'types')).reshape(batch, 12, -1),
        _embedding(parameters, 'move', _field(x, 'moves')).reshape(batch, 12, -1),
        _embedding(parameters, 'status', _field(x, 'status')), numeric])
    present = numeric[:, :, MON_NUMERIC_FIELDS.index('present')]
    active = numeric[:, :, MON_NUMERIC_FIELDS.index('active')] * present
    entities = jnp.tanh(jnp.concatenate(mon_inputs, -1) @ parameters['sp_mon_w'] + parameters['sp_mon_b'])
    entities = entities * present[..., None]
    own, foe = entities[:, :6], entities[:, 6:]
    own_active = (own * active[:, :6, None]).sum(1)
    foe_active = (foe * active[:, 6:, None]).sum(1)
    pools = [*_pool(own, present[:, :6]), *_pool(foe, present[:, 6:]), own_active, foe_active]

    event_ids = _field(x, 'event_ids')
    event_inputs = [_embedding(parameters, domain, event_ids[:, :, i]) for i, domain in enumerate(EVENT_FIELDS)]
    event_inputs.append(_field(x, 'event_numeric'))
    events = jnp.tanh(jnp.concatenate(event_inputs, -1) @ parameters['sp_event_w'] + parameters['sp_event_b'])
    events = events * _field(x, 'event_mask')[..., None]
    history = jnp.tanh(events.reshape(batch, -1) @ parameters['sp_history_w'] + parameters['sp_history_b'])
    counts = _field(x, 'public_move_counts')
    count_context = counts @ parameters['sp_move'] / jnp.maximum(
        counts.sum(-1, keepdims=True), jnp.finfo(jnp.float32).tiny)
    context = jnp.tanh(jnp.concatenate([*pools, history, count_context.reshape(batch, -1), host_hidden], -1)
                       @ parameters['sp_context_w'] + parameters['sp_context_b'])
    subjects = jnp.concatenate([jnp.broadcast_to(own_active[:, None, :], (batch, 8, ENTITY_WIDTH)), own], 1)
    action_inputs = jnp.concatenate([
        _embedding(parameters, 'move', _field(x, 'action_move')),
        _embedding(parameters, 'type', _field(x, 'action_type')),
        _field(x, 'action_numeric'), subjects,
        jnp.broadcast_to(foe_active[:, None, :], (batch, 14, ENTITY_WIDTH))], -1)
    action_hidden = jnp.tanh(action_inputs @ parameters['sp_action_w']
                            + (context @ parameters['sp_action_context_w'])[:, None, :]
                            + parameters['sp_action_b'])
    return action_hidden, context


def forward(parameters, x):
    """Masked action selection is shared separately; value remains in [0, 1]."""
    h = jnp.tanh(x[:, :928] @ parameters['w'] + parameters['b'])
    extra = x[:, 928:1278]
    rows = extra.reshape(-1, 14, 25)
    logits = h @ parameters['pw'] + parameters['pb'] + rows @ parameters['matchup_policy']
    pre = rows @ parameters['local_w'] + (h @ parameters['context_w'])[:, None, :] + parameters['residual_b']
    logits = logits + jnp.tanh(pre) @ parameters['residual_out']
    value_prelogit = h @ parameters['vw'] + parameters['vb'] + extra @ parameters['matchup_value']
    action_hidden, context = structured_context(parameters, x, h)
    return (logits + action_hidden @ parameters['sp_policy_out'],
            jax.nn.sigmoid(value_prelogit + context @ parameters['sp_value_out']))
