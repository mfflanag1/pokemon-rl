# Repository storage and restoration

The project repository is
[mfflanag1/pokemon-rl](https://github.com/mfflanag1/pokemon-rl).
The October 28 source publication covers source, tests, contracts, documentation,
and root dependency configuration. It does not upload new experiment artifacts,
raw run logs, checkpoints, external teacher data, vendor checkouts, or environments.
Historical artifacts already committed on October 12 remain in Git history, and
the original release remains available. This update does not delete those records
or claim to be a complete current experiment backup.

The working checkout retains frozen scientific input bytes. Publication copies
of selected documentation replace machine-specific links and update the public
status; those edited copies are not the original hash-bound experiment inputs.
Newer handoffs may refer to evidence retained only in the local archive. Exact
replay requires that separately preserved evidence and the pinned runtime.

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

Install GitHub CLI and
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

See the [current status](README.md) and [F39 failure handoff](docs/F39_ONLINE_CONTINUATION.md).
F39 completed production training and evaluation but failed before independent
verification began; its three primary gates are provisionally false. F40 is a
verifier-only repair in preparation, with no retraining. No successful F39/F40
completion or policy promotion is claimed. Full G2/G3/G4 and public ladder
strength remain open.
