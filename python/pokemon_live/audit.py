"""Replay every collected action through the independent scalar simulator/view."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_continuation import scalar, engine, chance
from pokemon_switch_engine.scalar import require
from . import learner as l


def collection(before, ids, seed, boundary, buffer, final):
    states = [engine.unpack(before, i) for i in range(l.BATCH)]
    counts = {'scalar_frames': 0, 'scalar_player_views': 0, 'post12_decisions': 0,
              'history_over32_decisions': 0, 'forced_waits': 0}
    for group in range(l.GROUPS):
        c = chance.sample(jax.random.key(seed), ids, jnp.array([s['group'] for s in states], jnp.uint32))
        crit, roll, tie = map(np.asarray, c)
        for i, state in enumerate(states):
            stopped = state['winner'] != -1 or state['phase'] == 0 and state['turn'] >= boundary
            actions = []
            for side in (0, 1):
                view = scalar.view(state, side); row = group * l.LANES + i * 2 + side
                mask = np.array(view['mask'], bool) & ~np.bool_(stopped)
                features = np.concatenate([np.asarray(view[k]).ravel() for k in ('own', 'pp', 'opponent', 'context', 'history')]).astype(np.float32) / 128
                if not mask.any(): features[:] = 0
                require(np.array_equal(features, np.asarray(buffer.features[row])), 'Scalar actor feature mismatch')
                require(np.array_equal(mask, np.asarray(buffer.mask[row])), 'Scalar action mask mismatch')
                action = int(buffer.actions[row]); actions.append(action)
                counts['scalar_player_views'] += 1
                counts['post12_decisions'] += int(mask.any() and state['turn'] > 12)
                counts['history_over32_decisions'] += int(mask.any() and len(state['history']) > 32)
                counts['forced_waits'] += int(not stopped and state['phase'] == 1 and not mask.any())
            if not stopped:
                states[i], _ = scalar.step(state, actions, {'crit': crit[i].tolist(), 'roll': roll[i].tolist(), 'tie': int(tie[i])})
            counts['scalar_frames'] += 1
    for i, state in enumerate(states):
        require(state == engine.unpack(final, i), 'Scalar final-state mismatch')
    return counts
