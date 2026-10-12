import assert from 'node:assert/strict';
import {mkdirSync, writeFileSync} from 'node:fs';
import path from 'node:path';
import {ROOT} from '../packages/reference-node/provenance.mjs';
import {describe} from '../packages/reference-node/engine.mjs';
import {defaultSpec, runEpisode, replayEpisode} from '../packages/reference-node/runner.mjs';

const manifest = describe();
const output = path.join(ROOT, 'artifacts/runs', `r1-${Date.now()}`);
mkdirSync(output, {recursive: true});
const start = performance.now();
const rows = [];
for (let index = 0; index < 10; index++) {
  const spec = defaultSpec(`r1-smoke-${index}`);
  const original = await runEpisode(spec);
  assert.ok(original.decisions.every(row => !row.choice.includes('terastallize')), 'R1 baseline must not choose Tera');
  const replay = await replayEpisode(original);
  assert.equal(original.content_hash, replay.content_hash);
  writeFileSync(path.join(output, `episode-${index}.json`), `${JSON.stringify(original)}\n`);
  rows.push({index, ...original.outcome, decisions: original.decisions.length,
    unavailable_rejections: original.decisions.filter(row => !row.accepted).length,
    content_hash: original.content_hash});
}
const report = {
  status: 'passed', scope: 'R1 reference smoke; not accelerated-engine fidelity',
  manifest_id: manifest.manifest_id, seed_label_pattern: 'r1-smoke-{0..9}',
  episodes: rows, replay_count: rows.length, elapsed_seconds: (performance.now() - start) / 1000,
  output_directory: path.relative(ROOT, output),
};
writeFileSync(path.join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
writeFileSync(path.join(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
mkdirSync(path.join(ROOT, 'artifacts/manifests'), {recursive: true});
writeFileSync(path.join(ROOT, 'artifacts/manifests', `${manifest.manifest_id}.json`), `${JSON.stringify(manifest, null, 2)}\n`);
mkdirSync(path.join(ROOT, 'artifacts/reports'), {recursive: true});
writeFileSync(path.join(ROOT, 'artifacts/reports/r1-smoke.json'), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
