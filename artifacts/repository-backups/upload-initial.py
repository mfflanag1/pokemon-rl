"""One-shot launcher for the authorized private-repository backup upload.

This only uploads the already verified archive; it never changes experiment files.
Output creation is exclusive so a continuation cannot silently duplicate a run.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
outputs = root / 'artifacts/repository-backups'
assets = Path(sys.argv[1]).resolve()
manifest = json.loads((outputs / 'runs-20261012.json').read_text())
archive = assets / manifest['archive']['name']
digest = hashlib.sha256()
with archive.open('rb') as stream:
    while data := stream.read(1024 * 1024):
        digest.update(data)
assert digest.hexdigest() == manifest['archive']['sha256']
assert archive.stat().st_size == manifest['archive']['bytes']
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
repo = 'mfflanag1/pokemon-rl'
remote = json.loads(subprocess.check_output(['gh', 'api', f'repos/{repo}/branches/main'], text=True))
assert remote['commit']['sha'] == head, 'Commit must be pushed before upload'
assert json.loads(subprocess.check_output(['gh', 'api', f'repos/{repo}'], text=True))['private']
command = ['gh', 'release', 'create', 'snapshot-2026-10-12', '--repo', repo,
           '--target', head, '--prerelease', '--latest=false', '--title',
           'Project and run-data snapshot — 2026-10-12', '--notes-file',
           str(assets / 'release-notes.md'), str(archive), str(assets / 'SHA256SUMS')]
start = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'pid': os.getpid(), 'commit': head, 'command': command,
         'archive_sha256': digest.hexdigest(), 'archive_bytes': archive.stat().st_size}
with (outputs / 'initial-upload-launch.json').open('x') as stream:
    json.dump(start, stream, indent=2); stream.write('\n')
with (outputs / 'initial-upload.log').open('x') as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
with (outputs / 'initial-upload-exit.json').open('x') as stream:
    json.dump({'exit_code': result.returncode,
               'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}, stream, indent=2)
    stream.write('\n')
raise SystemExit(result.returncode)
