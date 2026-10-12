"""Exercise production-evidence rejection without permitting receipt writes."""
import json
import runpy
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
run = ROOT / 'artifacts/runs/showdown-soak-v1'
out = ROOT / 'artifacts/reviews/showdown-soak'
module = runpy.run_path(str(ROOT / 'scripts/seal-showdown-soak.py'))
seal = module['seal']
writes = []
def deny_write(*args):
    writes.append(str(args[0]))
    raise AssertionError('Unexpected receipt/archive write')
seal.__globals__['write_once'] = deny_write
original_check_output = subprocess.check_output
def capture_output(*args, **kwargs):
    return original_check_output(*args, **{**kwargs, 'stderr': subprocess.STDOUT})
subprocess.check_output = capture_output

def update(directory, name, edit):
    path = directory / name
    data = json.loads(path.read_text()); edit(data)
    path.write_text(json.dumps(data))

cases = [
 ('failure', lambda d: (d / 'failure.json').write_text('{}'), 'Failed run cannot be sealed'),
 ('progress', lambda d: update(d, 'progress.json', lambda x: x.update(completed=999)), 'Complete progress'),
 ('source', lambda d: update(d, 'preregistration.json', lambda x: x['admission']['source_files'].update({'package.json':'0'*64})), 'Frozen source/plan'),
 ('game', lambda d: (d / 'game-0000.json.gz').write_bytes(b'corrupt'), 'Game evidence binding'),
 ('report', lambda d: update(d, 'report.json', lambda x: x.update(elapsed_ms=x['elapsed_ms']+1)), 'Audit binding: report.json'),
 ('plugin', lambda d: (d / 'server/dist/server/chat-plugins/soak-team.js').write_text('corrupt'), 'Local receipt endpoint source'),
 ('format', lambda d: (d / 'server/dist/config/custom-formats.js').write_text('corrupt'), 'Exact admitted server format'),
 ('forged-audit', lambda d: update(d, 'audit.json', lambda x: x['decisions'].update(tactical=x['decisions']['tactical']+1)), 'Fresh wire audit differs from saved audit'),
]
results = []
for name, corrupt, expected in cases:
    directory = ROOT / ('artifacts/runs/showdown-soak-negative-' + name)
    directory.mkdir()
    for path in run.iterdir():
        if path.is_file(): shutil.copy2(path, directory / path.name)
    for name2 in ['server/server.log', 'server/config/config.js', 'server/dist/config/custom-formats.js', 'server/dist/server/chat-plugins/soak-team.js']:
        target = directory / name2; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(run / name2, target)
    corrupt(directory)
    try:
        seal(directory)
        raise AssertionError('Negative unexpectedly passed')
    except Exception as error:
        message = str(error)
        if name == 'format':
            assert getattr(error, 'returncode', None) == 1, message
            assert expected in error.output, error.output
            (out / 'sealer-format-negative.log').write_text(error.output)
        else: assert expected in message, message
        assert not writes
        results.append({'case': name, 'rejected': True, 'expected': expected, 'error': message})
    print(json.dumps(results[-1]), flush=True)
(out / 'sealer-negative-checks.json').write_text(json.dumps({'status':'pass','cases':results,'receipt_writes':writes}, indent=2)+'\n')
