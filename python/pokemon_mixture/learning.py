"""Exact per-stratum PPO exposure; full-trajectory GAE precedes subsampling."""
import hashlib
import jax.numpy as jnp
import numpy as np
from pokemon_full.policy import finite_parameters
from pokemon_full_train.learning import prepare_batch, update, derived_seed, PPO, BATCH
from pokemon_full_train.storage import canonical
from pokemon_throughput.collector import merge_rows


def array_hash(batch):
    h = hashlib.sha256()
    for key, value in sorted(batch.items()):
        a = np.asarray(value)
        h.update(canonical([key, str(a.dtype), list(a.shape)])); h.update(a.tobytes())
    return h.hexdigest()


def select_batch(collection, quotas, seed, arm, update_index):
    """Collection maps stratum to complete games under one frozen policy."""
    if set(collection) != set(quotas):
        raise ValueError('Wrong selection strata')
    selected, ledger = [], {}
    for stratum, quota in quotas.items():
        rows = merge_rows(collection[stratum])
        if type(quota) is not int or not 0 < quota <= len(rows):
            raise ValueError('Insufficient learner rows for fixed quota')
        batch = prepare_batch(rows)
        rng = np.random.default_rng(derived_seed('f5-select', seed, arm, update_index, stratum))
        ids = rng.choice(len(rows), size=quota, replace=False)
        part = {k: a[ids] for k, a in batch.items()}
        selected.append(part)
        ledger[stratum] = {'collected_rows': len(rows), 'selected_rows': quota,
            'excluded_rows': len(rows)-quota, 'selected_indices': ids.tolist(),
            'complete_batch_sha256': array_hash(batch), 'selected_batch_sha256': array_hash(part)}
    batch = {k: np.concatenate([part[k] for part in selected]) for k in selected[0]}
    if len(batch['actions']) % BATCH:
        raise ValueError('Selected batch must fill complete minibatches')
    return batch, ledger


def optimize_batch(parameters, optimizer, batch, seed):
    n = len(batch['actions'])
    if n == 0 or n % BATCH:
        raise ValueError('Invalid exact training exposure')
    rng = np.random.default_rng(seed)
    m, v, step = optimizer
    metrics = []
    for _ in range(PPO['epochs']):
        order = rng.permutation(n)
        for first in range(0, n, BATCH):
            ids = order[first:first+BATCH]
            part = {k: jnp.asarray(a[ids]) for k, a in batch.items()}
            part['valid'] = jnp.ones(BATCH, dtype=bool)
            parameters, m, v, step, result = update(parameters, m, v, step, part)
            record = {k: float(x) for k, x in result.items()}
            if not all(np.isfinite(x) for x in record.values()):
                raise ValueError('Nonfinite update')
            metrics.append(record)
    for state in [parameters, m, v]:
        finite_parameters(state)
    return parameters, (m, v, step), metrics


def quotas_for(plan, arm):
    # Canonical JSON sorts object keys; batch row order must survive that roundtrip.
    weights = plan['arm_weights'][arm]
    return {name: int(weights[name]*plan['selected_rows_per_update'])
            for name in ['self_play', 'base_power', 'tactical'] if name in weights}


def training_specs(plan, seed, arm, stratum, first, count=8):
    run_seed = derived_seed('f5-training', seed, arm, stratum)
    return [dict(seed=run_seed, episode=i, max_turns=plan['max_turns'], split='train',
                 opponent=None if stratum == 'self_play' else stratum,
                 learner_side='p1' if i % 2 == 0 else 'p2') for i in range(first, first+count)]


def check_training(games, specs):
    if len(games) != len(specs):
        raise ValueError('Wrong training game count')
    for game, spec in zip(games, specs):
        for key in ['episode', 'split', 'opponent', 'learner_side']:
            if game[key] != spec[key]:
                raise ValueError('Wrong training coordinates')
        if game['swap'] is not False or any(t['partition'] != 'train' for t in game['provenance']['teams']):
            raise ValueError('Wrong training admission')
        controlled = [t for t in game['trace'] if spec['opponent'] is None or t[0] == spec['learner_side']]
        if game['learner_rows'] != game['learner_decisions'] or len(controlled) != game['learner_rows']:
            raise ValueError('Opponent actions entered learner rows or learner rows lost')
        if spec['opponent'] is not None and any(t[3] is not None for t in game['trace'] if t[0] != spec['learner_side']):
            raise ValueError('Opponent trace carries learner values')
        if len(game['trace']) != game['decisions'] or sum(game['phases'].values()) != game['decisions']:
            raise ValueError('Decision counter mismatch')
        if game['outcome']['kind'] not in ['terminal', 'administrative_cutoff']:
            raise ValueError('Incomplete training game')


def collect_update(pool, parameters, plan, seed, arm, update_index, cursors):
    quotas = quotas_for(plan, arm)
    collection, strata, next_cursors = {}, {}, dict(cursors)
    for stratum, quota in quotas.items():
        games, profiles, all_specs = [], [], []
        first = cursors[stratum]
        while sum(len(rows) for rows, _ in games) < quota:
            if len(games) >= plan['max_games_per_stratum_update']:
                raise RuntimeError('Fixed stratum collection ceiling reached')
            specs = training_specs(plan, seed, arm, stratum, first+len(games))
            new, profile = pool.collect(parameters, specs)
            check_training([g for _, g in new], specs)
            if any(len(rows) != game['learner_rows'] for rows, game in new):
                raise ValueError('Collected row ledger mismatch')
            games.extend(new); profiles.append(profile); all_specs.extend(specs)
        collection[stratum] = games
        next_cursors[stratum] = first+len(games)
        strata[stratum] = {'first_episode': first, 'specs': all_specs,
                           'episodes': [g for _, g in games], 'profiles': profiles}
    batch, selection = select_batch(collection, quotas, seed, arm, update_index)
    return batch, strata, selection, next_cursors
