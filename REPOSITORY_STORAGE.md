# Repository storage and restoration

The private project repository is
[mfflanag1/pokemon-rl](https://github.com/mfflanag1/pokemon-rl).
Git stores the source, contracts, dependency locks, documentation, review records,
completion receipts and their sealed source/evidence archives. Larger raw run data
are attached to a repository release; `.gitignore` remains unchanged because it is
part of the frozen experiment's source manifest.

## Initial snapshot

[Release `snapshot-2026-10-12`](https://github.com/mfflanag1/pokemon-rl/releases/tag/snapshot-2026-10-12)
contains `pokemon-runs-20261012.tar.zst` and `SHA256SUMS`. The
[file inventory](artifacts/repository-backups/runs-20261012.json) records every
relative path, byte count and SHA-256. The archive contains all 46,454 files then
present under `artifacts/runs`: 3,786,388,912 uncompressed bytes, stored in a
572,559,891-byte archive. Every archived file was independently decompressed and
hashed against that inventory before upload.

The snapshot was taken on October 12, 2026, from 02:13:10 to 02:13:25 UTC. The
M3 categorical screen had completed training and reference collection and was
still auditing. This snapshot does not certify that screen or replace any
experiment receipt. Subsequent reports and seals are preserved in later commits
and release supplements; the original snapshot is retained.

GitHub recommends release assets for large files outside Git history
([storage guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)).
The release is a data backup, not a software release or a strength claim.

## Restore on another machine

Use an account with access to the private repository. Install GitHub CLI and
Zstandard, then:

```sh
gh repo clone mfflanag1/pokemon-rl
cd pokemon-rl
mkdir -p /tmp/pokemon-run-backup
gh release download snapshot-2026-10-12 --repo mfflanag1/pokemon-rl \
  --dir /tmp/pokemon-run-backup
(cd /tmp/pokemon-run-backup && shasum -a 256 -c SHA256SUMS)
zstd -dc --long=29 /tmp/pokemon-run-backup/pokemon-runs-20261012.tar.zst | tar -xf -
```

Extract only into a fresh clone so an old snapshot cannot overwrite newer results.
Verify every restored run file using the committed inventory:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path('artifacts/repository-backups/runs-20261012.json').read_text())
for name, expected in m['files'].items():
    path = Path(name)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    assert path.stat().st_size == expected['bytes'], name
    assert digest.hexdigest() == expected['sha256'], name
print(f"Verified {len(m['files'])} restored files")
PY
```

Follow [R1 setup](docs/R1_REFERENCE.md) and [R2 setup](docs/R2_CONTRACT.md) to
rebuild the pinned Showdown checkout and local Node/Python environments. The
workspace's `.venv`, top-level `node_modules`, upstream vendor checkout and local
Git metadata are reproducible dependencies or machine state and are not backup
assets. Historical dependency copies already embedded in run evidence are retained
verbatim. Restoring bytes does not authorize using reserved final data or establish
that a different runtime can reproduce an old experiment.

## Current work

See [the screen handoff](docs/M3_SWITCH_SCREEN.md) for current execution status.
The original local audit remains authoritative until its production report,
fresh-reference sealing, complete hash verification and identical reseal finish.
Full G2/G3/G4 and public ladder strength remain open. Prime is unnecessary.
