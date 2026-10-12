"""Verify the completed private GitHub snapshot upload without changing run data."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
out = root / 'artifacts/repository-backups'
repo = 'mfflanag1/pokemon-rl'

def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', f'repos/{repo}/{path}'.rstrip('/')], text=True))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

exit_record = json.loads((out / 'initial-upload-exit.json').read_text())
assert exit_record['exit_code'] == 0, 'Upload has not exited successfully'
launch = json.loads((out / 'initial-upload-launch.json').read_text())
inventory = json.loads((out / 'runs-20261012.json').read_text())
archive = inventory['archive']
assert archive['sha256'] == launch['archive_sha256']
assert archive['bytes'] == launch['archive_bytes']
release = api('releases/tags/snapshot-2026-10-12')
assert not release['draft'] and release['prerelease']
assert release['target_commitish'] == launch['commit']
assert api('git/ref/tags/snapshot-2026-10-12')['object'] == {
    'sha': launch['commit'], 'type': 'commit',
    'url': f"https://api.github.com/repos/{repo}/git/commits/{launch['commit']}"}
assert api('')['private'] is True
checksums = f"{archive['sha256']}  {archive['name']}\n".encode()
expected = {archive['name']: (archive['bytes'], archive['sha256']),
            'SHA256SUMS': (len(checksums), hashlib.sha256(checksums).hexdigest())}
assets = {asset['name']: asset for asset in release['assets']}
assert set(assets) == set(expected), 'Missing or unexpected release assets'
for name, (size, digest) in expected.items():
    asset = assets[name]
    assert asset['state'] == 'uploaded' and asset['size'] == size, name
    assert asset.get('digest') == 'sha256:' + digest, (
        name, 'Digest absent or mismatched: independently download/hash before claiming success')
local_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
assert api('branches/main')['commit']['sha'] == local_head, 'Local HEAD not fully pushed'
result = {'status': 'passed', 'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'repository': f'https://github.com/{repo}', 'private': True,
          'remote_main_at_verification': local_head, 'snapshot_commit': launch['commit'],
          'release': release['html_url'], 'release_id': release['id'],
          'inventory_sha256': sha(out / 'runs-20261012.json'),
          'archive_roundtrip_checks_sha256': sha(out / 'pre-upload-checks.json'),
          'file_count': inventory['file_count'], 'uncompressed_bytes': inventory['uncompressed_bytes'],
          'assets': [{k: assets[name][k] for k in ('id', 'name', 'size', 'digest', 'state')}
                     for name in sorted(assets)]}
with (out / 'initial-transfer-verification.json').open('x') as stream:
    json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
print(json.dumps(result, indent=2))
