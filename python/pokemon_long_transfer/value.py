"""Descriptive value diagnostics for frozen-opponent evaluation traces.

Targets are final terminal rewards, repeated for each learner decision. The
constant-half comparator uses those exact same decisions and targets. No
independence, confidence, or policy improvement claims are made.
"""
import math
from numbers import Integral, Real


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('Expected a finite real scalar')
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError('Expected a finite real scalar') from exc
    if not math.isfinite(result):
        raise ValueError('Expected a finite real scalar')
    return result


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def summarize_value(games):
    """Return JSON-safe MSE, population spread, and ten calibration bins.

    ``game_mean_mse`` weights terminal games equally; ``decision_mean_mse``
    weights their learner decisions equally. Baselines use prediction 0.5;
    improvement is baseline minus model MSE. Empty cohorts are permitted.
    Cutoff rewards must be finite numbers but are never used as targets.
    Trace request IDs/actions are nonnegative integers and hashes nonempty
    strings; this function does not authenticate hashes or replay actions.
    """
    if isinstance(games, (str, bytes, dict)):
        raise ValueError('Expected an iterable of games')
    try:
        iterator = iter(games)
    except TypeError as exc:
        raise ValueError('Expected an iterable of games') from exc
    predictions, targets, errors, baselines = [], [], [], []
    game_errors, game_baselines = [], []
    cutoff_games = 0
    buckets = [[] for _ in range(10)]
    for game in iterator:
        if not isinstance(game, dict) or game.get('learner_side') not in ('p1', 'p2'):
            raise ValueError('Invalid learner side/game')
        side = game['learner_side']
        outcome = game.get('outcome')
        if not isinstance(outcome, dict) or outcome.get('kind') not in ('terminal', 'administrative_cutoff'):
            raise ValueError('Invalid outcome kind')
        rewards = outcome.get('rewards')
        if not isinstance(rewards, dict) or set(rewards) != {'p1', 'p2'}:
            raise ValueError('Invalid rewards')
        reward = {s: _number(rewards[s]) for s in ('p1', 'p2')}
        terminal = outcome['kind'] == 'terminal'
        if terminal and (any(v not in (0., .5, 1.) for v in reward.values()) or
                         reward['p1'] + reward['p2'] != 1.):
            raise ValueError('Invalid terminal rewards')
        trace = game.get('trace')
        if not isinstance(trace, (list, tuple)):
            raise ValueError('Invalid trace')
        values = []
        for entry in trace:
            if not isinstance(entry, (list, tuple)) or len(entry) != 5:
                raise ValueError('Invalid trace entry')
            actor, request_id, action, value, feature_hash = entry
            if actor not in ('p1', 'p2'):
                raise ValueError('Invalid trace side')
            if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0
                   for v in (request_id, action)):
                raise ValueError('Invalid request ID/action')
            if not isinstance(feature_hash, str) or not feature_hash:
                raise ValueError('Invalid feature hash')
            if actor == side:
                value = _number(value)
                if not 0. <= value <= 1.:
                    raise ValueError('Prediction outside [0, 1]')
                values.append(value)
            elif value is not None:
                raise ValueError('Opponent prediction must be None')
        if not values:
            raise ValueError('Game has no controlled decisions')
        if not terminal:
            cutoff_games += 1
            continue
        target = reward[side]
        mse = [(v - target)**2 for v in values]
        baseline = (.5 - target)**2
        game_errors.append(_mean(mse))
        game_baselines.append(baseline)
        predictions.extend(values)
        targets.extend([target] * len(values))
        errors.extend(mse)
        baselines.extend([baseline] * len(values))
        for value, error in zip(values, mse):
            buckets[min(int(value * 10), 9)].append((value, target, error))
    result = {'terminal_games': len(game_errors), 'cutoff_games': cutoff_games,
              'terminal_decisions': len(predictions),
              'mean_prediction': _mean(predictions), 'mean_target': _mean(targets),
              'prediction_std': None}
    if predictions:
        result['prediction_std'] = math.sqrt(_mean([
            (v - result['mean_prediction'])**2 for v in predictions]))
    for weighting, model, baseline in [('game', game_errors, game_baselines),
                                        ('decision', errors, baselines)]:
        result[f'{weighting}_mean_mse'] = _mean(model)
        result[f'{weighting}_mean_baseline_mse'] = _mean(baseline)
        result[f'{weighting}_mean_improvement'] = (
            _mean(baseline) - _mean(model) if model else None)
    result['bins'] = [
        {'lo': i / 10, 'hi': (i + 1) / 10, 'count': len(bucket),
         'mean_prediction': _mean([x[0] for x in bucket]),
         'empirical_target': _mean([x[1] for x in bucket]),
         'mean_mse': _mean([x[2] for x in bucket])}
        for i, bucket in enumerate(buckets)]
    return result
