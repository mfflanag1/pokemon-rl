# Finish the GitHub preservation requested on October 11

The user requested the next project step and a GitHub repository storing all this
work. Repository creation and pushing project source, documentation and evidence
are authorized. Use the existing private `mfflanag1/pokemon-rl` repository and
`origin`; do not create another repository or make this one public.

## Initial backup

- Repository: https://github.com/mfflanag1/pokemon-rl
- Git contains all non-ignored project files, including sealed manifests/receipts.
- `REPOSITORY_STORAGE.md` explains restoration.
- `artifacts/repository-backups/runs-20261012.json` inventories all 46,454 run files
  present during the first snapshot, including the then-incomplete M3 screen.
- Release tag: `snapshot-2026-10-12`.
- Assets: `pokemon-runs-20261012.tar.zst`, `SHA256SUMS`.
- Archive SHA-256: a332e3d1d68515e88e0c7ab04c6d40486dd5583d06b61a6a2f0ec5b3a3e4cdad.
- Preserve this original release and inventory unchanged.
- Initial transfer verification is recorded beside this file after upload.

The initial archive upload uses `upload-initial.py` in this directory. Its
`initial-upload-launch.json`, `initial-upload.log` and `initial-upload-exit.json`
record the process, command and result. Never start another uploader while it is
alive. After exit 0, verify both GitHub release assets' sizes and SHA-256 digests
against the local archive inventory and checksum file; download/hash only if GitHub
omits its digest. Save `initial-transfer-verification.json` here and commit/push it.
Inspect the existing release if a transfer fails; preserve diagnostics and do not
claim a completed backup. Finish this initial verification before uploading the
later screen supplement. No duplicate release or overwrite of a published asset.

While the scientific audit continues, independently finish the initial transfer
as soon as its launcher exits. The prepared read-only verifier is:

```sh
python3 artifacts/repository-backups/verify-initial.py
```

It checks the exact release tag/commit, private visibility, completed upload,
both asset digests/sizes and the pushed main branch, then exclusively writes
`initial-transfer-verification.json`. If that record already exists, inspect it
instead of overwriting it. Commit/push the backup metadata and verification record
without changing any frozen scientific source. A missing server digest requires
the independent download/hash fallback described above, not relaxing the check.

## After the current M3 screen completes or fails

Finish the frozen experiment's existing continuation first. On success require
production exit 0, complete first sealing, all live/archive integrity checks and
identical resealing. On failure preserve the failure evidence and state that the
experiment is unqualified. This backup workflow never changes the experiment,
sealed files, source, reserved-data access rules or its statistical decision.

Preserve the final `artifacts/runs/m3-switch-screen-v1` directory in a NEW release
supplement. Use a new local temporary directory and a new release tag such as
`m3-switch-screen-result-2026-10-12`, checking that neither already exists before
creation. If continuation is interrupted, inspect existing release/assets first;
do not overwrite or duplicate a successful upload. Archive relative to the project
root, with a tar+zstd asset. Record every file's size and SHA-256 in a new JSON
inventory under `artifacts/repository-backups`, record the archive's SHA-256,
and independently decompress/hash every member against the inventory. Include a
SHA256SUMS asset. Keep raw archive files outside the project so Git cannot stage
large archives inadvertently. Restore supplements only over a verified initial
snapshot; newer run files intentionally supersede the older running snapshot.

Commit the final progress documents, receipt/archive (or failure evidence),
review outputs, new backup inventory and a short release/restore note under
`artifacts/repository-backups`. Review `git status` and preserve unrelated work.
Never rewrite historical commits or any sealed evidence. Source files remain frozen;
no test rerun is required for byte preservation or documentation alone. Existing
`.gitignore` remains unchanged. Do not put post-seal notes into the sealed screen
review directory or edit source/docs bound by a completed receipt.

Push the new commit to `origin/main`. The first streaming HTTPS push failed; this
command uses per-command buffering without changing global settings:

```sh
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  -c http.version=HTTP/1.1 -c http.postBuffer=524288000 push origin main
```

Create the supplemental release at the exact pushed commit, upload the archive and
checksums, then inspect GitHub's release-asset size and SHA-256 digest (or download
and hash if a digest is unavailable). Check the remote main SHA equals local HEAD
and repository visibility is private. Save a transfer-verification JSON outside
all sealed directories; commit/push that record and verify the final remote SHA.
Report the actual screen result, handoff, receipt (if valid), and GitHub repository
and release links, then pause the existing strength-screen heartbeat. If GitHub
transfer fails, preserve local evidence and report the specific blocker; never
claim the final backup is complete or discard files to force a successful upload.
