"""Fixed three-seed, two-cohort CPU experiment with complete scalar/trace audits."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from pokemon_continuation import engine
from pokemon_learner.checkpoint import digest, wire, file_hash
from pokemon_switch_engine.scalar import require
from . import learner as l, checkpoint as cp, audit as scalar_audit

CHECKPOINT_WINDOWS = (0, 1, 2, 3, 4, 5, 10)
SCOPE = {'strength': 'unqualified', 'sustained_qualification': 'not run', 'full_g2': 'open', 'full_g3': 'open',
         'full_g4': 'open', 'final_reserved_data_used': False, 'public_server_used': False, 'prime_used': False}


def write(path, data):
    with Path(path).open('xb') as f: f.write(wire(data) + b'\n')


def summary(previous, result, *, scalar=True):
    s, buffer, final, metrics = result; valid = np.asarray(buffer.valid)
    first = int(previous.windows) % 5 == 0; last = int(s.windows) % 5 == 0
    before = engine.initial_kernel(l.teams(s.ids, s.seed)) if first else previous.environment
    boundary = l.PLAN['boundaries'][(int(s.windows) - 1) % 5]
    counts = scalar_audit.collection(before, s.ids, int(s.seed), boundary, buffer, final) if scalar else {}
    return {'window': int(s.windows), 'updates': int(s.updates), 'optimizer_step': int(s.step), 'next_game': int(s.next_game),
            'games_started': l.BATCH if first else 0, 'continued_games': 0 if first else int((np.asarray(before.winner) == -1).sum()),
            'decisions': int(valid.sum()), 'terminal_rows': int(buffer.terminal.sum()),
            'new_terminal_games': int(buffer.terminal.sum()) // 2,
            'cutoff_games': int((np.asarray(final.winner) == -1).sum()) if last else 0,
            'reward_sum': float(buffer.rewards.sum()),
            'value_min': float(np.asarray(buffer.values)[valid].min()) if valid.any() else None,
            'value_max': float(np.asarray(buffer.values)[valid].max()) if valid.any() else None,
            'return_min': float(np.asarray(buffer.returns)[valid].min()) if valid.any() else None,
            'return_max': float(np.asarray(buffer.returns)[valid].max()) if valid.any() else None,
            'metrics': {k: np.asarray(v).tolist() for k, v in metrics.items()}, 'scalar_audit': counts,
            'fingerprints': {k: cp.fingerprint(v) for k, v in zip(('session', 'buffer', 'final_state', 'metrics'), result)}}


def audit_checkpoint(out, s, admitted):
    name = f'checkpoint-{int(s.seed)}-{int(s.windows)}.json'; path = out / name
    cp.save(path, s, admitted)
    original = summary(s, l.transaction(s, int(s.seed), audit_only=True), scalar=False)
    loaded = cp.load(path, int(s.seed), admitted)
    resumed = summary(loaded, l.transaction(loaded, int(s.seed), audit_only=True), scalar=False)
    require(original == resumed, 'Original-next-update replay mismatch')
    return {'file': name, 'sha256': file_hash(path), 'state': cp.fingerprint(s), 'original_next': original}


def totals(records):
    trace = [x for r in records for x in r['trace']]
    result = {'collection_windows': len(trace), 'optimizer_updates': sum(r['trace'][-1]['updates'] for r in records),
              'checkpoint_pairs': len(records) * len(CHECKPOINT_WINDOWS),
              'generation_checkpoint_audit_transactions': 2 * len(records) * len(CHECKPOINT_WINDOWS)}
    for total, field in [('games', 'games_started'), ('decisions', 'decisions'), ('terminal_games', 'new_terminal_games'), ('cutoff_games', 'cutoff_games')]:
        result[total] = sum(x[field] for x in trace)
    result['generation_audit_games_started'] = 2 * sum(e['original_next']['games_started'] for r in records for e in r['checkpoints'])
    result['generation_audit_decisions'] = 2 * sum(e['original_next']['decisions'] for r in records for e in r['checkpoints'])
    for key in ('scalar_frames', 'scalar_player_views', 'post12_decisions', 'history_over32_decisions', 'forced_waits'):
        result[key] = sum(x['scalar_audit'][key] for x in trace)
    require(result['games'] == result['terminal_games'] + result['cutoff_games'], 'Closed cohort accounting')
    require(result['post12_decisions'] > 0 and result['history_over32_decisions'] > 0 and result['forced_waits'] > 0, 'Nonvacuous continuation')
    return result


def run(directory):
    out = Path(directory).resolve(); require(out.is_relative_to(l.ROOT / 'artifacts/runs'), 'Run path')
    out.mkdir(exist_ok=False); admitted = cp.admission(); write(out / 'admission.json', admitted)
    records = []; started = time.perf_counter()
    try:
        for seed in l.PLAN['seeds']:
            s = l.start(seed); trace = []; checkpoints = []
            for window in range(l.PLAN['windows_per_seed'] + 1):
                if window in CHECKPOINT_WINDOWS: checkpoints.append(audit_checkpoint(out, s, admitted))
                if window == l.PLAN['windows_per_seed']: break
                result = l.transaction(s, seed); trace.append(summary(s, result)); s = result[0]
            records.append({'seed': seed, 'trace': trace, 'checkpoints': checkpoints})
        require(cp.admission() == admitted, 'Source/runtime drift')
        report = {'format': 'live-learner-canary-1', 'status': 'passed-engineering-canary',
                  'admission_hash': digest(admitted), 'records': records, **totals(records),
                  'elapsed_seconds': time.perf_counter() - started, **SCOPE}
        report['content_hash'] = digest(report); write(out / 'report.json', report)
    except BaseException as error:
        write(out / 'failure.json', {'error': repr(error)}); raise
    print(json.dumps({k: report[k] for k in ('status', 'collection_windows', 'optimizer_updates', 'games', 'decisions', 'terminal_games', 'cutoff_games')}))
    return report


def audit(directory):
    out = Path(directory); report = json.loads((out / 'report.json').read_bytes()); admitted = json.loads((out / 'admission.json').read_bytes())
    require(not (out / 'failure.json').exists(), 'Failed run evidence')
    require(report['format'] == 'live-learner-canary-1' and report['status'] == 'passed-engineering-canary', 'Report identity')
    require(digest({k: v for k, v in report.items() if k != 'content_hash'}) == report['content_hash'], 'Report digest')
    require(admitted == cp.admission() and digest(admitted) == report['admission_hash'], 'Source/runtime binding')
    require([r['seed'] for r in report['records']] == l.PLAN['seeds'], 'Seed roster')
    for record in report['records']:
        seed = record['seed']; s = l.start(seed)
        require(len(record['trace']) == l.PLAN['windows_per_seed'], 'Window budget')
        require([e['file'] for e in record['checkpoints']] == [f'checkpoint-{seed}-{w}.json' for w in CHECKPOINT_WINDOWS], 'Checkpoint roster')
        checkpoints = dict(zip(CHECKPOINT_WINDOWS, record['checkpoints']))
        for window in range(l.PLAN['windows_per_seed'] + 1):
            if window in checkpoints:
                e = checkpoints[window]; path = out / e['file']; loaded = cp.load(path, seed, admitted)
                require(file_hash(path) == e['sha256'] and cp.fingerprint(loaded) == e['state'] == cp.fingerprint(s), 'Checkpoint versus original full trajectory')
                require(summary(loaded, l.transaction(loaded, seed, audit_only=True), scalar=False) == e['original_next'], 'Original-next-update fingerprint/telemetry')
            if window == l.PLAN['windows_per_seed']: break
            result = l.transaction(s, seed)
            require(summary(s, result) == record['trace'][window], 'Full scalar/learner trace replay'); s = result[0]
    total = totals(report['records'])
    require(all(report[k] == v for k, v in {**total, **SCOPE}.items()), 'Totals/scope')
    require(cp.admission() == admitted, 'Audit source/runtime drift')
    return {'status': 'passed', 'replayed_windows': total['collection_windows'],
            'acceptance_replay_transactions': total['collection_windows'] + total['checkpoint_pairs'],
            'acceptance_checkpoint_replay_transactions': total['checkpoint_pairs'], **total}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory'); parser.add_argument('--audit', action='store_true'); args = parser.parse_args()
    if args.audit: print(json.dumps(audit(args.directory)))
    else: run(args.directory)
