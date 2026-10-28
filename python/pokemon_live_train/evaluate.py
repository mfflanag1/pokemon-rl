"""Frozen, player-view-only opponents; paired complete-game evaluation."""
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_continuation import engine, carry, chance, scalar
from pokemon_live import policy
from pokemon_switch_engine import policy as categorical
from pokemon_switch_engine.scalar import require
from pokemon_switch_screen.statistics import summarize
from . import learner as l


def damage_actions(view):
    """Immediate mean-roll damage from public species and own exact HP only.

    The restricted roster's public base stats determine defense. Hidden swift
    affects speed only. No hidden opponent HP, PP, member or swift enters here.
    """
    require(type(view) is engine.PlayerView, 'Opponent requires PlayerView')
    n = len(view.context); ix = jnp.arange(n)
    own = view.own[ix, view.context[:, 2]]
    foe = view.opponent[ix, view.context[:, 3]]
    source = own[:, 0] - 1; target = jnp.maximum(foe[:, 0] - 1, 0)
    scores = jnp.stack([engine.damage(source, target, own[:, 1], jnp.full(n, a),
                                      jnp.zeros(n, bool), jnp.full(n, 7)) for a in (0, 1)], 1)
    scores = jnp.concatenate([scores.astype(jnp.float32), jnp.zeros((n, 2))], 1)
    return jnp.where(view.mask.any(1), jnp.argmax(jnp.where(view.mask, scores, -1), 1), -1).astype(jnp.int32)


@jax.jit
def game_batch(parameters, frozen, team_ids, game_ids, master_seed, candidate_seat, opponent_kind):
    state = engine.initial_kernel(team_ids); master = jax.random.key(master_seed)
    actor_master = jax.random.fold_in(master, 0x45564137)

    def one(state, _):
        views = [engine.view_kernel(state, side) for side in (0, 1)]
        obs = [policy.encode(view) for view in views]
        live = ~carry.stopped(state, 61)
        obs = [policy.ActorInput(o.features, o.mask & live[:, None]) for o in obs]
        actions = []; values = []; logs = []
        for side in (0, 1):
            keys = jax.vmap(lambda gid, group: jax.random.fold_in(jax.random.fold_in(
                jax.random.fold_in(actor_master, gid), group), side))(game_ids, state.group.astype(jnp.uint32))
            ca, cl, cv = policy.act(parameters, obs[side], keys)
            fa, fl, fv = policy.act(frozen, obs[side], keys)
            words = jax.vmap(lambda k: jax.random.bits(k, (3, 2), dtype=jnp.uint32))(keys)
            ra = categorical.select(jnp.zeros((len(game_ids), 4)), obs[side].mask, words)
            da = damage_actions(views[side]); da = jnp.where(live, da, -1)
            oa = jnp.where(opponent_kind == 0, ra, jnp.where(opponent_kind == 1, fa, da))
            actions.append(jnp.where(candidate_seat == side, ca, oa))
            values.append(cv); logs.append(cl)
        actions = jnp.stack(actions, 1)
        after, _ = carry.step_kernel(state, actions, chance.sample(master, game_ids, state.group.astype(jnp.uint32)), 61)
        return after, (actions, jnp.stack(values, 1), jnp.stack(logs, 1),
                       jnp.stack([o.mask.any(1) for o in obs], 1), state.turn)
    final, tape = jax.lax.scan(one, state, None, length=70)
    return final, tape


def schedule(seed, opponent, start, seat):
    # Restricted team identities recur; this is fresh-game, in-distribution
    # evaluation, not a held-out-team generalization test. Addresses are disjoint.
    ids = jnp.arange(start, start + l.BATCH, dtype=jnp.uint32) + jnp.uint32(0x40000000 + opponent * 65536)
    pair = jnp.stack([(ids + seed) % 24, (ids // 24 + seed // 24) % 24], 1).astype(jnp.int32)
    return (pair if seat == 0 else pair[:, ::-1]), ids


def audit_tape(final, tape, teams, ids, master_seed):
    """Independent scalar transition replay for every evaluation action/outcome."""
    states = [engine.unpack(engine.initial_kernel(teams), i) for i in range(l.BATCH)]
    actions, values, logps, valid, turns = map(np.asarray, tape)
    require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), 'Evaluation values')
    require(np.isfinite(logps).all(), 'Evaluation log probabilities')
    for group in range(70):
        crit, roll, tie = map(np.asarray, chance.sample(jax.random.key(master_seed), ids,
            jnp.array([s['group'] for s in states], jnp.uint32)))
        for i, state in enumerate(states):
            stopped = state['winner'] != -1 or state['phase'] == 0 and state['turn'] >= 61
            for side in (0, 1):
                mask = np.asarray(scalar.view(state, side)['mask'], bool) & ~np.bool_(stopped)
                require(bool(mask.any()) == bool(valid[group, i, side]), 'Evaluation request mask')
                action = int(actions[group, i, side])
                require(mask[action] if 0 <= action < 4 else action == -1 and not mask.any(), 'Evaluation illegal action')
            require(int(turns[group, i]) == state['turn'], 'Evaluation turn linkage')
            if not stopped:
                states[i], _ = scalar.step(state, actions[group, i].tolist(),
                    {'crit': crit[i].tolist(), 'roll': roll[i].tolist(), 'tie': int(tie[i])})
    for i, state in enumerate(states): require(state == engine.unpack(final, i), 'Evaluation scalar outcome mismatch')


def diagnostics(final, tape, seat):
    actions, values, _, valid, turns = map(np.asarray, tape)
    active = valid[:, :, seat]; v = values[:, :, seat]; terminal = np.asarray(final.winner) != -1
    target = (np.asarray(final.winner) == seat).astype(float)
    selected = active & terminal[None, :]
    calibration = []
    for lo in np.arange(0, 1, .2):
        hit = selected & (v >= lo) & (v <= lo + .2 if lo > .79 else v < lo + .2)
        calibration.append({'lower':float(lo),'count':int(hit.sum()),
            'mean_value':float(v[hit].mean()) if hit.any() else None,
            'mean_outcome':float(np.broadcast_to(target, v.shape)[hit].mean()) if hit.any() else None})
    return {'decisions':int(active.sum()),'actions':np.bincount(actions[:, :, seat][active], minlength=4).tolist(),
        'value_min':float(v[active].min()),'value_max':float(v[active].max()),
        'terminal_value_rows':int(selected.sum()),
        'terminal_brier_sum':float(((v - target[None, :])**2)[selected].sum()),
        'calibration':calibration,'post12_decisions':int((active & (turns > 12)).sum()),
        'terminal_after12':int((terminal & (np.asarray(final.turn) > 12)).sum())}


def comparison(final, initial):
    plan = l.PLAN['screen']
    return summarize(np.asarray(final, np.int32), np.asarray(initial, np.int32),
                     alpha=plan['alpha'], family_size=plan['family_size'])


def gate(cells):
    require(set(cells) == set(l.PLAN['screen']['opponents']), 'Gate opponent roster')
    return (cells['random']['adjusted_improvement_lower'] > 0 and
            all(cells[name]['worst_case_improvement'] >= l.PLAN['screen']['other_opponent_regression_floor']
                for name in ('initialization', 'damage')))


def interpretation(cells):
    if gate(cells): return 'passed-preregistered-diagnostic-gate'
    if any(cells[name]['worst_case_improvement'] < l.PLAN['screen']['other_opponent_regression_floor']
           for name in ('initialization','damage')): return 'failed-noisy-point-regression-guard'
    return 'inconclusive-at-preregistered-budget'
