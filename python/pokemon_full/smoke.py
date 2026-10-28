"""Explicit fixed-budget engineering smoke. No strength metric or resumable checkpoint."""
import argparse
import json
import time
import jax
import jax.numpy as jnp
from .client import NodeWorker
from .features import FEATURES
from .learner import train_cohort, parameter_hash
from .policy import initialize, act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, choices=[8, 16], default=8)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--max-turns', type=int, default=60)
    parser.add_argument('--curriculum', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or not 2 <= args.max_turns <= 120:
        parser.error('seed must be uint32; max-turns must be in [2,120]')
    if jax.default_backend() != 'cpu' or jax.config.x64_enabled:
        raise RuntimeError('F1 smoke is admitted only on CPU FP32')
    total_started = started = time.perf_counter()
    parameters = initialize(args.seed)
    initial_hash = parameter_hash(parameters)
    # Explicit actor warmup outside measured collection, including both wait/live.
    act(parameters, jnp.zeros((2, FEATURES), jnp.float32), jnp.ones((2, 14), bool),
        jax.random.split(jax.random.key(args.seed), 2))[0].block_until_ready()
    actor_initialization_seconds = time.perf_counter()-started
    zeros = {k: jnp.zeros_like(v) for k, v in parameters.items()}
    optimizer = (zeros, {k: jnp.zeros_like(v) for k, v in parameters.items()}, jnp.int32(0))
    cohorts = []
    with NodeWorker() as worker:
        for first in range(0, args.games, 8):
            parameters, optimizer, report = train_cohort(worker, parameters, optimizer, run_seed=args.seed,
                first_episode=first, max_turns=args.max_turns, curriculum=args.curriculum)
            cohorts.append(report)
        receipt = {'schema_version': 'full-showdown-smoke-v1', 'scope': 'engineering_smoke_no_strength_claim',
            'arguments': vars(args), 'reference_commit': worker.hello['reference_commit'],
            'backend': jax.default_backend(), 'jax_version': jax.__version__,
            'actor_initialization_seconds': actor_initialization_seconds,
            'node_initialization_seconds': worker.initialization_seconds,
            'parameter_count': sum(v.size for v in parameters.values()),
            'initial_parameter_hash': initial_hash, 'final_parameter_hash': parameter_hash(parameters),
            'optimizer_steps': int(optimizer[2]), 'cohorts': cohorts}
    receipt['total_seconds_including_initialization_and_shutdown'] = time.perf_counter()-total_started
    receipt['total_decisions'] = sum(cohort['decisions'] for cohort in cohorts)
    receipt['decisions_per_total_second'] = receipt['total_decisions']/receipt['total_seconds_including_initialization_and_shutdown']
    print(json.dumps(receipt, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
