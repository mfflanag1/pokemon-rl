"""F35 PPO plus one fixed offline action-label loss; no demonstration returns."""
import jax
import jax.numpy as jnp
import numpy as np

from pokemon_fixed.math import adam_step
from pokemon_full.policy import distribution
from pokemon_full_train.learning import PPO
from pokemon_structured_policy import learning as original
from pokemon_structured_policy.network import check_state, forward
from .data import validate_demo

COEFFICIENT = .1
DEMO_ROWS = 256


def _coefficient(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)):
        raise ValueError('F35 assistance coefficient')
    if not np.isfinite(value) or float(value) not in (0., COEFFICIENT):
        raise ValueError('F35 admits only frozen assistance or zero-assistance control')
    return float(value)


def demo_objective(parameters, batch):
    """Pure hard-action NLL. Critic-only parameters have exactly zero gradient."""
    logits, _ = forward(parameters, batch['features'])
    probs, logs = distribution(logits, batch['mask'])
    selected = logs[jnp.arange(len(logits)), batch['actions']]
    loss = -jnp.mean(selected)
    return loss, {'demo_nll': loss,
                  'demo_accuracy': jnp.mean((jnp.argmax(probs, 1) == batch['actions']).astype(jnp.float32)),
                  'demo_target_mass': jnp.mean(probs[jnp.arange(len(logits)), batch['actions']])}


def objective(parameters, batch, demonstrations, coefficient):
    ppo_loss, metrics = original.objective(parameters, batch)
    demo_loss, demo_metrics = demo_objective(parameters, demonstrations)
    loss = ppo_loss + coefficient * demo_loss
    return loss, {**metrics, 'loss': loss, 'ppo_loss': ppo_loss, **demo_metrics,
                  'weighted_demo_loss': coefficient * demo_loss}


@jax.jit
def kernel(parameters, m, v, step, batch, demonstrations, coefficient):
    (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(
        parameters, batch, demonstrations, coefficient)
    parameters, m, v, step, norm = adam_step(parameters, gradients, m, v, step,
                                           lr=PPO['learning_rate'], max_norm=1.)
    return parameters, (m, v, step), {**metrics, 'gradient_norm': norm}


def update(parameters, optimizer, batch, demonstrations=None, *, coefficient=COEFFICIENT):
    coefficient = _coefficient(coefficient)
    if coefficient == 0.:
        return original.update(parameters, optimizer, batch)
    check_state(parameters, optimizer)
    b = original.validate(batch)
    demo = validate_demo(demonstrations)
    if len(demo['actions']) != DEMO_ROWS or len(batch['valid']) != original.MINIBATCH or not np.asarray(batch['valid']).all():
        raise ValueError('Assisted update requires 256 live PPO and demonstration rows; no padding')
    if int(optimizer[2]) >= 2**31 - 1:
        raise ValueError('Optimizer counter overflow')
    demo = {k: jnp.asarray(v) for k, v in demo.items()}
    q, qo, metrics = kernel(parameters, *optimizer, b, demo, jnp.float32(coefficient))
    check_state(q, qo)
    if int(qo[2]) != int(optimizer[2]) + 1 or not all(np.isfinite(float(v)) for v in metrics.values()):
        raise ValueError('F35 update linkage/metrics')
    return q, qo, {k: float(v) for k, v in metrics.items()}


def optimize_batch(parameters, optimizer, batch, seed, *, dataset=None, demo_seed=None,
                   cohort=0, coefficient=COEFFICIENT):
    """Same PPO permutation and eight Adam steps; demo RNG is separate and explicit."""
    coefficient = _coefficient(coefficient)
    if coefficient == 0.:
        return original.optimize_batch(parameters, optimizer, batch, seed)
    if not isinstance(batch, dict) or set(batch) != original.FIELDS or len(batch['actions']) != original.ROWS:
        raise ValueError('Fixed F35 PPO cohort inventory')
    for name, value in [('shuffle seed', seed), ('demo seed', demo_seed), ('cohort', cohort)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**32:
            raise ValueError('Unsigned F35 ' + name + ' required')
    if dataset is None or dataset.split != 'train':
        raise ValueError('Assistance requires the train-only demonstration corpus')
    b = original.validate({**batch, 'valid': np.ones(original.ROWS, bool)})
    check_state(parameters, optimizer)
    if int(optimizer[2]) > 2**31 - 1 - original.EPOCHS * (original.ROWS // original.MINIBATCH):
        raise ValueError('Optimizer counter overflow')
    rng = np.random.default_rng(int(seed))
    records = []
    for _ in range(original.EPOCHS):
        order = rng.permutation(original.ROWS)
        for first in range(0, original.ROWS, original.MINIBATCH):
            ids = order[first:first + original.MINIBATCH]
            demo = dataset.sample(seed=int(demo_seed), cohort=int(cohort), update=len(records), rows=DEMO_ROWS)
            parameters, optimizer, metrics = update(parameters, optimizer, {k: v[ids] for k, v in b.items()}, demo)
            records.append(metrics)
    return parameters, optimizer, records


@jax.jit
def _probe_predictions(parameters, features, mask):
    logits, values = forward(parameters, features)
    probabilities, logs = distribution(logits, mask)
    return probabilities, logs, values


def probe_metrics(parameters, dataset, *, reference_parameters=None, batch_rows=256):
    """Whole-corpus diagnostics, equally weighted by games and by decisions.

    Observed terminal rewards support diagnostics only, never a value loss.
    No score here selects a checkpoint or changes the frozen assistance schedule.
    """
    if type(batch_rows) is not int or batch_rows < 1:
        raise ValueError('Positive diagnostic batch size required')
    nll, correct, mass, values, drift = [], [], [], [], []
    for first in range(0, dataset.rows, batch_rows):
        count = min(batch_rows, dataset.rows - first)
        features = np.zeros((batch_rows, dataset.arrays['features'].shape[1]), np.float32)
        mask = np.ones((batch_rows, 14), bool)
        features[:count] = dataset.arrays['features'][first:first + count]
        mask[:count] = dataset.arrays['mask'][first:first + count]
        probabilities, logs, value = _probe_predictions(parameters, jnp.asarray(features), jnp.asarray(mask))
        probabilities, logs, value = (np.asarray(v[:count]) for v in (probabilities, logs, value))
        actions = dataset.arrays['actions'][first:first + count]
        ids = np.arange(count)
        nll.extend(np.asarray(-logs[ids, actions], dtype=np.float64))
        correct.extend(np.asarray(np.argmax(probabilities, 1) == actions, dtype=np.float64))
        mass.extend(np.asarray(probabilities[ids, actions], dtype=np.float64))
        values.extend(np.asarray(value, dtype=np.float64))
        if reference_parameters is not None:
            _, _, reference = _probe_predictions(reference_parameters, jnp.asarray(features), jnp.asarray(mask))
            drift.extend(np.asarray(value - np.asarray(reference[:count]), dtype=np.float64))
    series = {'nll': np.asarray(nll), 'accuracy': np.asarray(correct), 'target_mass': np.asarray(mass),
              'value_mean': np.asarray(values)}
    if drift:
        series.update(value_signed_drift=np.asarray(drift), value_absolute_drift=np.abs(drift),
                      value_squared_drift=np.square(drift))
    result = {'rows': dataset.rows, 'games': len(dataset.game_ids), 'split': dataset.split,
              'action_counts': np.bincount(dataset.arrays['actions'], minlength=14).tolist()}
    for key, array in series.items():
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite demonstration diagnostic')
        result['decision_' + key] = float(array.mean())
        result['game_' + key] = float(np.mean([array[a:b].mean() for a, b in zip(dataset.offsets[:-1], dataset.offsets[1:])]))
    terminal_errors = [(np.asarray(values[a:b]) - target)**2
                       for a, b, target in zip(dataset.offsets[:-1], dataset.offsets[1:], dataset.terminal_returns)
                       if target is not None]
    result.update(terminal_games=len(terminal_errors), terminal_rows=sum(len(v) for v in terminal_errors),
                  decision_terminal_value_mse=float(np.concatenate(terminal_errors).mean()) if terminal_errors else None,
                  game_terminal_value_mse=float(np.mean([v.mean() for v in terminal_errors])) if terminal_errors else None)
    return result
