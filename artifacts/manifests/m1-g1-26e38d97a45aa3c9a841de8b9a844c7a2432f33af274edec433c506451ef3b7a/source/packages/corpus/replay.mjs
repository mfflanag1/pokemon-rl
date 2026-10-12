import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {generateTeam} from '../reference-node/engine.mjs';
import {contractManifest, teamArtifact} from '../reference-node/contract-v1.mjs';
import {canonical, hashJSON, parseJSON} from '../battle-contract/canonical.mjs';
import {validateArtifact} from '../battle-contract/artifacts.mjs';
import {checkChunk} from './generate.mjs';
import {readJSON, validateIndex, seedAt, seal, atomicJSON, digest} from './common.mjs';

export function replayIndices(index, sampleCount) {
  const count = index.config.count;
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 1 || sampleCount > count) throw new TypeError('Invalid replay sample size');
  const selected = new Set();
  if (sampleCount >= 2) { selected.add(0); selected.add(count - 1); }
  // Include each shard if the sample budget allows, then sample without replacement.
  if (sampleCount >= index.chunks.length + 1) index.chunks.forEach(c => selected.add(c.start));
  for (let k = 0; selected.size < sampleCount; k++) {
    selected.add(Number(BigInt('0x' + hashJSON(['r3-replay-v1', index.corpus_id, k]).slice(0, 16)) % BigInt(count)));
  }
  return [...selected].sort((a, b) => a - b);
}

export function replayCorpus({out, samples = 1000}) {
  const directory = resolve(out), index = validateIndex(readJSON(`${directory}/index.json`));
  const manifest = contractManifest();
  if (manifest.manifest_id !== index.config.battle_manifest_id || canonical(readJSON(`${directory}/manifest.json`)) !== canonical(manifest)) throw new Error('Replay reference differs from corpus');
  const selected = replayIndices(index, Math.min(samples, index.config.count)), records = [];
  for (const chunk of index.chunks) {
    const needed = selected.filter(i => i >= chunk.start && i < chunk.start + chunk.count);
    const bytes = checkChunk(directory, chunk); // Check every shard, even with no sampled row.
    if (!needed.length) continue;
    const lines = bytes.toString('utf8').split('\n');
    if (lines.pop() !== '' || lines.length !== chunk.count) throw new Error('Shard row count mismatch');
    for (const i of needed) {
      const stored = validateArtifact(parseJSON(lines[i - chunk.start]), {kind: 'team', manifestId: manifest.manifest_id});
      const seed = seedAt(index.config.namespace, i);
      const regenerated = teamArtifact(generateTeam(seed), manifest.manifest_id);
      if (canonical(stored) !== canonical(regenerated)) throw new Error(`Team replay differs at index ${i}`);
      records.push({index: i, seed, artifact_id: stored.artifact_id, source_team_hash: stored.payload.source_team_hash});
    }
  }
  const report = seal({report_version: 'r3-replay-1', corpus_id: index.corpus_id, index_hash: index.content_hash,
    source_hash: digest(readFileSync(new URL('./replay.mjs', import.meta.url))),
    sample_version: 'r3-replay-v1', verified: true, sample_count: records.length, records});
  atomicJSON(`${directory}/replay.json`, report);
  return report;
}
