import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {openSync, closeSync, writeSync, readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {runEpisode, defaultSpec} from '../packages/reference-node/runner.mjs';
import {contractManifest, episodeArtifacts} from '../packages/reference-node/contract-v1.mjs';
import {canonical} from '../packages/battle-contract/canonical.mjs';
import {toProtocol} from '../packages/battle-contract/actions.mjs';

const started = Date.now();
const directory = `artifacts/runs/r2-${started}`;
mkdirSync(directory, {recursive: true});
const input = `${directory}/node.jsonl`, output = `${directory}/python.jsonl`;
const fd = openSync(input, 'wx');
const manifest = contractManifest();
const cases = [];
let artifactCount = 0;
try {
  for (let i = 0; i < 10; i++) {
    const label = `r1-smoke-${i}`;
    const source = await runEpisode(defaultSpec(label));
    const bundle = episodeArtifacts(source, manifest);
    for (const [j, view] of bundle.views.entries()) {
      assert.deepEqual(view.payload.admissible_actions.map(a => toProtocol(a, view.payload.request)), source.decisions[j].admissible_choices);
      assert.equal(toProtocol(bundle.decisions[j].payload.submitted_action, view.payload.request), source.decisions[j].choice);
    }
    const artifacts = [...(i === 0 ? [manifest] : []), ...bundle.teams, ...bundle.views, ...bundle.decisions, bundle.episode];
    for (const artifact of artifacts) writeSync(fd, canonical(artifact) + '\n');
    artifactCount += artifacts.length;
    cases.push({label, decisions: bundle.decisions.length, turns: source.outcome.turns,
      unavailable_rejections: bundle.decisions.filter(d => !d.payload.accepted).length,
      episode_id: bundle.episode.artifact_id, normalized_reference_hash: source.normalized_output_hash});
  }
} finally { closeSync(fd); }
const python = process.env.CONTRACT_PYTHON ?? '.venv/bin/python';
const outputFd = openSync(output, 'wx');
let result;
try {
  result = spawnSync(python, ['-m', 'pokemon_reference', input], {
    env: {...process.env, PYTHONPATH: 'python'}, stdio: ['ignore', outputFd, 'pipe'], encoding: 'utf8', timeout: 60000,
  });
} finally { closeSync(outputFd); }
assert.ifError(result.error);
assert.equal(result.status, 0, result.stderr);
const digest = path => createHash('sha256').update(readFileSync(path)).digest('hex');
assert.equal(digest(input), digest(output), 'Node/Python canonical JSONL bytes differ');
const pythonVersion = spawnSync(python, ['--version'], {encoding: 'utf8'});
assert.equal(pythonVersion.status, 0);
const report = {
  report_version: 'r2-contract-smoke-1', generated_at: new Date().toISOString(),
  manifest_id: manifest.manifest_id, node: process.version, python: pythonVersion.stdout.trim(),
  contract_source_hashes: manifest.payload.contract_hashes,
  consumer_source_hashes: Object.fromEntries(['python/pokemon_reference/contract.py', 'python/pokemon_reference/__main__.py', 'requirements-contract.txt'].map(p => [p, digest(p)])),
  cases, episodes: cases.length, artifacts: artifactCount,
  decisions: cases.reduce((n, c) => n + c.decisions, 0),
  turns: cases.reduce((n, c) => n + c.turns, 0),
  unavailable_rejections: cases.reduce((n, c) => n + c.unavailable_rejections, 0),
  canonical_jsonl_sha256: digest(input), canonical_jsonl_bytes: readFileSync(input).byteLength,
  python_canonical_roundtrip_identical: true, elapsed_ms: Date.now() - started,
  evidence_directory: directory,
};
writeFileSync(`${directory}/report.json`, JSON.stringify(report, null, 2) + '\n', {flag: 'wx'});
mkdirSync('artifacts/reports', {recursive: true});
writeFileSync('artifacts/reports/r2-contract.json', JSON.stringify(report, null, 2) + '\n');
writeFileSync(`${directory}/manifest.json`, canonical(manifest) + '\n', {flag: 'wx'});
console.log(JSON.stringify({episodes: report.episodes, decisions: report.decisions, artifacts: report.artifacts,
  canonical_bytes: report.canonical_jsonl_bytes, python_roundtrip: true, elapsed_ms: report.elapsed_ms, directory}));
