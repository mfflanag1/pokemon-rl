"""Unchanged teacher collection with player-only coverage annotations."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import numpy as np
from pokemon_battle_audit.transport import Worker
from pokemon_imitation.teacher import collect_game
from pokemon_imitation.data import validate
from pokemon_full.features import encode
from pokemon_credit.campaign import record
from pokemon_entropy.campaign import array_record


class ObservedWorker:
    def __init__(self, worker):
        self.worker = worker
        self.frame = None
        self.metadata = []
        self.identities = []

    def request(self, operation, **command):
        before = self.frame
        after = self.worker.request(operation, **command)
        if operation == 'start':
            self.metadata, self.identities = [], []
            self.episode = command['episode_index']
        if operation == 'step':
            for submitted in after['submitted']:
                if not submitted['accepted']:
                    continue
                side = submitted['side']
                p = before['players'][side]
                target = command['actions'][side]
                if submitted['action'] != target or target != before['tactical_actions'][side]:
                    raise ValueError('Accepted teacher target mismatch')
                x, mask = encode(p)
                own = next(q for q in p['own'] if q and q['active'])
                foe = next(q for q in p['opponent'] if q and q['active'])
                self.metadata.append({'game':self.episode, 'phase':p['phase'], 'target':target,
                    'own_species':own['species'], 'foe_species':foe['species'],
                    'destination_species':p['own'][target-8]['species'] if target >= 8 else '',
                    'feature_hash':hashlib.sha256(x.tobytes()+mask.tobytes()).hexdigest()})
                self.identities.append({'episode':self.episode, 'side':side,
                    'request_id':p['request_id'], 'action':target, 'phase':p['phase']})
        self.frame = after
        return after


def validate_metadata(data, rows):
    validate(data)
    if len(rows) != len(data['targets']):
        raise ValueError('Coverage row count mismatch')
    for i, row in enumerate(rows):
        h = hashlib.sha256(data['features'][i].tobytes()+data['mask'][i].tobytes()).hexdigest()
        if row['game'] != int(data['game_index'][i]) or row['target'] != int(data['targets'][i]) or row['feature_hash'] != h:
            raise ValueError('Coverage row linkage mismatch')
        phase = {'move':1, 'replace':2, 'revival':3}.get(row['phase'])
        if phase is None or data['features'][i, 865+phase] != 1:
            raise ValueError('Coverage phase linkage mismatch')


def generate(seed, count, split, plan, evidence, original=None):
    evidence.mkdir(parents=True, exist_ok=False)
    def work(slot):
        outputs = []
        with Worker(evidence/f'worker-{slot}') as worker:
            observer = ObservedWorker(worker)
            for episode in range(slot, count, plan['workers']):
                data, game = collect_game(observer, seed, episode, split, plan['max_turns'])
                if observer.identities != game['row_identity']:
                    raise ValueError('Coverage identity mismatch')
                rows = observer.metadata
                validate_metadata(data, rows)
                record(evidence/f'game-{episode:04d}.json', game, original/f'game-{episode:04d}.json' if original else None)
                record(evidence/f'coverage-{episode:04d}.json', rows, original/f'coverage-{episode:04d}.json' if original else None)
                outputs.append((episode, data, game, rows))
        return outputs
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        outputs = sorted([item for batch in pool.map(work, range(min(count, plan['workers']))) for item in batch], key=lambda x:x[0])
    if [x[0] for x in outputs] != list(range(count)):
        raise ValueError('Missing or duplicate teacher game')
    data = {k:np.concatenate([o[1][k] for o in outputs]) for k in outputs[0][1]}
    games = [o[2] for o in outputs]
    rows = [r for o in outputs for r in o[3]]
    h = array_record(evidence/'dataset.npz', data, original/'dataset.npz' if original else None)
    inventory = {'split':split, 'seed':seed, 'games':count, 'rows':len(rows), 'fingerprint':h,
        'terminals':sum(g['outcome']['kind']=='terminal' for g in games),
        'cutoffs':sum(g['outcome']['kind']=='administrative_cutoff' for g in games),
        'decisions':sum(g['decisions'] for g in games),
        'rejected_choices':sum(g['rejected_choices'] for g in games),
        'phase_counts':dict(Counter(r['phase'] for r in rows)),
        'action_counts':dict(Counter(str(r['target']) for r in rows))}
    if inventory['terminals']+inventory['cutoffs'] != count or inventory['rows']+inventory['rejected_choices'] != inventory['decisions']:
        raise ValueError('Outcome/decision count mismatch')
    record(evidence/'inventory.json', inventory, original/'inventory.json' if original else None)
    return data, games, rows, inventory


def combine(original, additional, original_rows, additional_rows, offset=512):
    if set(np.unique(original['game_index'])) != set(range(offset)) or set(np.unique(additional['game_index'])) != set(range(1536)):
        raise ValueError('Corpus game inventory mismatch')
    combined = {k:np.concatenate([original[k], additional[k]+offset if k=='game_index' else additional[k]]) for k in original}
    rows = [dict(r) for r in original_rows]+[{**r, 'game':r['game']+offset} for r in additional_rows]
    validate_metadata(combined, rows)
    return combined, rows
