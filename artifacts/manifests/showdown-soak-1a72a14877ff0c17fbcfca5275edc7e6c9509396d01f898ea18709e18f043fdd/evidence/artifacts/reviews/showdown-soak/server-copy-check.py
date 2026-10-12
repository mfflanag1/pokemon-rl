"""Check the disposable server copy against the pinned upstream after shutdown."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
upstream = ROOT / 'vendor/pokemon-showdown'
server = ROOT / 'artifacts/runs/showdown-soak-v1/server'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
overrides = {'config/config.js', 'dist/config/custom-formats.js', 'dist/server/chat-plugins/soak-team.js'}
files = {}
for name in ['dist', 'config', 'data', 'translations', 'server/static']:
    originals = {str(p.relative_to(upstream)): p for p in (upstream/name).rglob('*') if p.is_file()}
    copies = {str(p.relative_to(server)): p for p in (server/name).rglob('*') if p.is_file()}
    assert set(originals) - overrides == set(copies) - overrides, name
    for relative, path in originals.items():
        if relative in overrides: continue
        assert sha(copies[relative]) == sha(path), relative
        files[relative] = sha(path)
assert (server/'node_modules').is_symlink()
assert (server/'node_modules').resolve() == (upstream/'node_modules').resolve()
report = {'status':'pass', 'unmodified_upstream_files':len(files), 'hashes':files,
          'explicit_overrides':{p:sha(server/p) for p in sorted(overrides)},
          'dependency_link':str((server/'node_modules').resolve())}
(ROOT/'artifacts/reviews/showdown-soak/server-copy-integrity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'hashes'}))
