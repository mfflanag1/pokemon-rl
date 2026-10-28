"""Physical-game-clustered actor-gradient agreement on original selected rows."""
from functools import partial
import hashlib
import jax
import numpy as np
from pokemon_matchup.learning import validate
from .core import component, vector

actor_gradient = jax.jit(jax.grad(partial(component, name='actor')))


def row_metadata(ledger, batch):
    selected = []
    for stratum in ledger['strata']:
        rows = []
        for gid, game in enumerate(ledger['games']):
            if stratum['first_episode'] <= game['episode'] < stratum['next_episode']:
                if game['opponent'] != stratum['opponent'] or game['split'] != 'train':
                    raise ValueError('Stratum/game identity mismatch')
                for actor in game['actor_trace']:
                    reward = game['outcome']['rewards'][actor['side']] if game['outcome']['kind'] == 'terminal' else None
                    rows.append({'game': gid, 'side': actor['side'], 'stream': stratum['opponent'] or 'self_play',
                                 'terminal_reward': reward, 'actor': actor})
        choice = stratum['selection']
        ids = choice['selected_indices']
        if len(rows) != choice['collected_rows'] or len(ids) != choice['selected_rows'] or len(set(ids)) != len(ids):
            raise ValueError('Original selection inventory mismatch')
        if any(type(i) is not int or not 0 <= i < len(rows) for i in ids):
            raise ValueError('Original selection index mismatch')
        selected.extend(rows[i] for i in ids)
    if len(selected) != len(batch['actions']):
        raise ValueError('Selected row count mismatch')
    for i, m in enumerate(selected):
        actor = m.pop('actor')
        if (hashlib.sha256(batch['features'][i].tobytes()).hexdigest() != actor['feature_sha256']
                or batch['mask'][i].tolist() != actor['mask'] or int(batch['actions'][i]) != actor['action']
                or float(batch['old_logp'][i]) != actor['log_probability']):
            raise ValueError('Selected row/actor identity mismatch')
        if abs(float(batch['returns'][i]-batch['advantages'][i]) - actor['value']) > 2e-6:
            raise ValueError('Selected row/value identity mismatch')
    return selected


def between_share(values, labels):
    values = np.asarray(values, np.float64)
    total = float(np.sum((values-values.mean())**2))
    between = sum(int(np.sum(labels == label)) * float(values[labels == label].mean()-values.mean())**2
                  for label in np.unique(labels))
    return between / total if total > 1e-30 else None


def game_halves(metadata, seed, repeats=4):
    rng = np.random.default_rng(seed)
    gids = np.asarray([m['game'] for m in metadata])
    streams = np.asarray([m['stream'] for m in metadata])
    results = []
    for _ in range(repeats):
        a = np.zeros(len(metadata), bool)
        for stream in sorted(set(streams)):
            games = np.unique(gids[streams == stream])
            if len(games) < 2:
                raise ValueError('Too few selected games for disjoint halves')
            order = rng.permutation(games)
            a |= (streams == stream) & np.isin(gids, order[:len(order)//2])
        if not a.any() or a.all() or set(gids[a]) & set(gids[~a]):
            raise ValueError('Game cluster split failed')
        results.append((a, ~a))
    return results


def audit_signal(p, batch, ledger, seed):
    metadata = row_metadata(ledger, batch)
    gids = np.asarray([m['game'] for m in metadata])
    sides = np.asarray([m['side'] for m in metadata])
    streams = np.asarray([m['stream'] for m in metadata])
    advantage = np.asarray(batch['advantages'], np.float64)
    summary = {}
    for stream in sorted(set(streams)):
        ids = streams == stream
        games = [g for g in ledger['games'] if (g['opponent'] or 'self_play') == stream]
        reward = np.asarray([m['terminal_reward'] if m['terminal_reward'] is not None else np.nan for m in metadata])
        denom = float(np.abs(advantage[ids]).sum())
        won = [i for i,g in enumerate(ledger['games']) if (g['opponent'] or 'self_play') == stream
               and g['outcome']['kind'] == 'terminal' and g['outcome']['rewards'][g['learner_side']] == 1]
        summary[stream] = {'collected_physical_games':len(games), 'selected_physical_games':len(np.unique(gids[ids])),
            'selected_rows':int(ids.sum()), 'selected_game_seat_trajectories':len(set(zip(gids[ids], sides[ids]))),
            'won_tactical_games':len(won) if stream == 'tactical' else None,
            'selected_won_tactical_games':len(set(won)&set(gids[ids])) if stream == 'tactical' else None,
            'positive_advantage_fraction':float(np.mean(advantage[ids]>0)),
            'mean_advantage':float(advantage[ids].mean()), 'std_advantage':float(advantage[ids].std()),
            'winning_seat_abs_advantage_mass_fraction':float(np.abs(advantage[ids & (reward == 1)]).sum())/denom if denom else None,
            'physical_game_between_advantage_variance_share':between_share(advantage[ids],gids[ids]),
            'game_seat_between_advantage_variance_share':between_share(advantage[ids],2*gids[ids]+(sides[ids]=='p2'))}
    records = []
    for a,b in game_halves(metadata, seed):
        vecs = []
        for mask in [a,b]:
            data = validate({**batch, 'valid':mask})
            grad = actor_gradient(p, data)
            vecs.append(vector(grad, sorted(p)))
        norms = [float(np.linalg.norm(v)) for v in vecs]
        denom = norms[0]*norms[1]
        records.append({'half_rows':[int(a.sum()),int(b.sum())],
                        'half_games':[len(np.unique(gids[a])),len(np.unique(gids[b]))],
                        'gradient_norms':norms,
                        'actor_cosine':float(vecs[0]@vecs[1])/denom if denom>1e-30 else None,
                        'left_game_ids':np.unique(gids[a]).tolist(), 'right_game_ids':np.unique(gids[b]).tolist()})
    return {'streams':summary,'game_disjoint_halves':records,
            'interpretation':'Four overlapping diagnostic splits of each selected cohort, not independent samples or a formal gradient-SNR estimate. Both seats of a physical game remain in one half; stream composition is stratified by game count, not matched rows. Differences include game/phase mix and sparse outcomes. No off-policy pooling or winning-game selection.'}
