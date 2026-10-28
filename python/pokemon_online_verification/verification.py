"""F40 verifier-only repair; preserve the original F39 run and its failure."""
import argparse
import json
from pathlib import Path

from pokemon_full_train.storage import atomic_json
from pokemon_online_continuation import verification as original
from .guard import snapshot


def probe():
    inventory = snapshot()
    if inventory['forbidden']:
        raise ValueError('F40 forbidden teacher/search module: ' + repr(inventory['forbidden']))
    return inventory


def run(training_run, output):
    """Replace only the guard callable for this invocation; all replay logic is original."""
    history = []
    previous = original.teacher_search_modules

    def corrected_guard():
        inventory = snapshot()
        history.append(inventory)
        return inventory['forbidden']

    original.teacher_search_modules = corrected_guard
    try:
        result = original.run(training_run, output)
        if len(history) != 2 or any(item['forbidden'] for item in history):
            raise ValueError('F40 expected exactly two clean module snapshots')
        result['f40_runtime_policy'] = {'schema_version': 'f40-runtime-policy-v1',
            'replacement': 'Only pokemon_online_continuation.verification.teacher_search_modules for this invocation.',
            'snapshots': history, 'production_repeated': False,
            'original_failure_preserved': True}
        atomic_json(Path(output) / 'report.json', result)
        return result
    finally:
        original.teacher_search_modules = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', action='store_true', help='Import/guard check only; no scientific run.')
    args = parser.parse_args()
    if args.probe:
        if args.training_run is not None or args.output is not None:
            parser.error('--probe cannot be combined with run paths')
        print(json.dumps({'status': 'passed', 'inventory': probe()}), flush=True)
    else:
        if args.training_run is None or args.output is None:
            parser.error('--training-run and --output are required')
        result = run(args.training_run, args.output)
        print(json.dumps({'status': result['status'], 'counts': result['counts']}), flush=True)
