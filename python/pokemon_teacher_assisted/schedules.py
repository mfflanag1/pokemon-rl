"""Inspect fixed admitted team identities without running any battle or changing schedules."""
import argparse
from pathlib import Path
import subprocess
import time

from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest
from .common import (SEEDS, DEADLINE, before_deadline, evaluation_specs, teacher_specs,
                     read, verify_inputs)


def jobs_for(plan):
    jobs = []
    for seed in SEEDS:
        for role in ('train', 'probe'):
            for spec in teacher_specs(plan, seed, role)[::2]:
                jobs.append({'group': 'teacher_' + role, 'host_seed': seed,
                    'seed': spec['run_seed'], 'episode': spec['episode_index'],
                    'split': spec['split'], 'swap': False})
        for opponent, pairs in {**plan['evaluation_pairs'], 'foul_play': plan['external_pairs_per_seed_state']}.items():
            for spec in evaluation_specs(plan, seed, opponent, 0, pairs)[::2]:
                jobs.append({'group': 'evaluation', 'host_seed': seed, 'opponent': opponent,
                    'seed': spec['seed'], 'episode': spec['episode'], 'split': spec['split'], 'swap': False})
    return jobs


def check_records(jobs, records, quarantine):
    if len(jobs) != len(records):
        raise ValueError('Fixed schedule inventory mismatch')
    keys = {name: set() for name in ('teacher_train', 'teacher_probe', 'evaluation')}
    for spec, record in zip(jobs, records):
        if any(record.get(k) != v for k, v in spec.items()) or len(record['teams']) != 2:
            raise ValueError('Schedule coordinates changed')
        pair = []
        for team in record['teams']:
            key = team['team_key']
            if team['partition'] != spec['split'] or key in quarantine:
                raise ValueError('Schedule split/quarantine violation')
            pair.append(key)
            keys[record['group']].add(key)
        if len(set(pair)) != 2:
            raise ValueError('Schedule self-team matchup')
    intersections = {}
    groups = list(keys)
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            intersections[left + '/' + right] = sorted(keys[left] & keys[right])
    if any(intersections.values()):
        raise ValueError('Fixed teacher/probe/evaluation team intersection; no schedule replacement permitted')
    return {'unique_team_counts': {k: len(v) for k, v in keys.items()},
            'intersections': intersections, 'boundary': 'exact ordered whole-team identity only'}


def run(output, admission):
    before_deadline()
    output, admission = Path(output).resolve(), Path(admission).resolve()
    admitted = verify_inputs(admission)
    jobs = jobs_for(admitted['plan'])
    output.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    try:
        with (output / 'team-identities.jsonl').open('xb') as out, (output / 'stderr.log').open('xb') as err:
            process = subprocess.run(['node', 'packages/teacher-assisted/schedules.mjs'],
                input=canonical(jobs), cwd=ROOT, stdout=out, stderr=err,
                timeout=max(1., min(600., DEADLINE - time.time())), check=False)
        if process.returncode:
            raise RuntimeError('Fixed team identity generation exited ' + str(process.returncode))
        import json
        records = [json.loads(line) for line in (output / 'team-identities.jsonl').read_text().splitlines()]
        result = check_records(jobs, records, set(admitted['plan']['quarantined_team_keys']))
        report = {'status': 'passed', 'input_hash': digest(admission / 'manifest.json'),
            'pairs': len(jobs), 'team_identities_sha256': digest(output / 'team-identities.jsonl'),
            'elapsed_seconds': time.perf_counter() - begun, **result}
        atomic_json(output / 'report.json', report)
        return report
    except BaseException as exc:
        atomic_json(output / 'failure.json', {'status': 'failed', 'error': repr(exc),
            'elapsed_seconds': time.perf_counter() - begun, 'schedule_replaced': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output, args.admission), flush=True)
