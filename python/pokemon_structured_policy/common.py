"""Small F34 scheduling and evidence helpers; no predecessor admission recursion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from pokemon_full_train.learning import derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest, runtime
from pokemon_full_learning.statistics import compare

SEEDS = (7103, 8209, 9311)
DEADLINE = datetime(2026, 10, 28, 2, 0, 49, tzinfo=timezone.utc).timestamp()


def read(path):
    return json.loads(Path(path).read_text())


def before_deadline():
    if time.time() >= DEADLINE:
        raise TimeoutError('F34 eight-hour deadline reached; no new work admitted')


def evaluation_specs(plan, seed, opponent, first, count):
    return [dict(seed=derived_seed('f34-evaluation-teams-v1', seed, opponent),
        episode=episode, max_turns=plan['max_turns'], split='development', swap=swap,
        opponent=opponent, learner_side='p2' if swap else 'p1',
        actor_seed=derived_seed('f34-evaluation-actor-v1', seed, opponent))
        for episode in range(first, first + count) for swap in (False, True)]


def training_specs(plan, seed, first, opponent):
    return [dict(seed=derived_seed('f34-training-teams-v1', seed, opponent), episode=i,
        max_turns=plan['max_turns'], split='train', swap=False, opponent=opponent,
        learner_side='p2' if i % 2 else 'p1',
        actor_seed=derived_seed('f34-training-actor-v1', seed, opponent))
        for i in range(first, first + plan['training_chunk_games'])]


def state_hash(parameters, optimizer):
    import numpy as np
    h = hashlib.sha256()
    for prefix, group in [('p', parameters), ('m', optimizer[0]), ('v', optimizer[1])]:
        for key, value in sorted(group.items()):
            a = np.asarray(value)
            h.update((prefix + '/' + key + '/' + str(a.shape) + '/' + str(a.dtype)).encode())
            h.update(a.tobytes())
    h.update(np.asarray(optimizer[2]).tobytes())
    return h.hexdigest()


def bind_inputs(destination, plan, source_paths):
    """Freeze selected source/config bytes once, separate from mutable progress prose."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for rel in sorted(set(source_paths)):
        path = ROOT / rel
        data = path.read_bytes()
        target = destination / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()
    execution = runtime()
    if execution['backend'] != 'cpu' or execution['x64']:
        raise ValueError('F34 requires the qualified CPU FP32 runtime')
    body = {'schema_version': 'f34-inputs-v1', 'plan': plan, 'source_files': files,
            'runtime': execution}
    atomic_json(destination / 'manifest.json', body)
    return digest(destination / 'manifest.json')


def verify_inputs(destination):
    destination = Path(destination)
    body = read(destination / 'manifest.json')
    if runtime() != body['runtime']:
        raise ValueError('F34 qualified runtime drift')
    for rel, expected in body['source_files'].items():
        if digest(ROOT / rel) != expected or digest(destination / 'source' / rel) != expected:
            raise ValueError('F34 input drift: ' + rel)
    return body


def admit_games(games, split, quarantine):
    for game in games:
        if game['split'] != split or game['outcome']['kind'] not in ('terminal', 'administrative_cutoff'):
            raise ValueError('Unfinished or wrong-role game')
        for team in game['provenance']['teams']:
            if team['partition'] != split or team['team_key'] in quarantine:
                raise ValueError('Team role/quarantine violation')
        if len(game['trace']) != game['decisions'] or sum(game['phases'].values()) != game['decisions']:
            raise ValueError('Decision accounting mismatch')
        # Rejected requests are legal protocol feedback, retained as decisions;
        # they cannot be silently omitted from trajectories or outcome accounting.


def learning_analysis(initial, final, settings):
    contrasts = {opponent: compare(initial[opponent], final[opponent], settings)
                 for opponent in initial}
    primary = contrasts['tactical']
    checks = {
        'tactical_cutoff_aware_gain_at_least_five_points': primary['cutoff_aware_point_interval'][0] >= .05,
        'tactical_adjusted_bootstrap_positive': primary['approximate_familywise_bootstrap_interval'][0] > 0,
        'no_detected_regression': all(c['gate'] != 'regression' for c in contrasts.values()),
        'final_cutoffs_at_most_five_percent': all(c['final']['cutoffs'] <= .05 * c['final']['games']
                                               for c in contrasts.values()),
    }
    return {'contrasts': contrasts, 'checks': checks, 'passed': all(checks.values()),
        'scope': 'Development adaptation; tactical is trained on. Approximate paired-cluster bootstrap, not public strength.'}
