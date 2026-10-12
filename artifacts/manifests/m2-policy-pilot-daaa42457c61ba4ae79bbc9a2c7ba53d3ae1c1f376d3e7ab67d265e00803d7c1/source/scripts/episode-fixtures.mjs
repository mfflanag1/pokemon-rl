import {mkdirSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {gzipSync} from 'node:zlib';
import {defaultSpec} from '../packages/reference-node/runner.mjs';
import {fileHash} from '../packages/reference-node/provenance.mjs';
import {canonical} from '../packages/battle-contract/canonical.mjs';
import {seal, digest} from '../packages/corpus/common.mjs';
import {episodeManifest} from '../packages/episodes/manifest.mjs';
import {runPhaseEpisode, replayPhaseEpisode} from '../packages/episodes/session.mjs';
import {FIXTURES} from '../packages/episodes/fixtures.mjs';

const out = resolve(process.argv[2] ?? 'artifacts/runs/r4-fixtures-v1');
mkdirSync(out, {recursive: true});
const spec = defaultSpec('r1-smoke-8'), records = [];
for (const profile of ['official', 'constructed']) writeFileSync(`${out}/${profile}.manifest.json`, canonical(episodeManifest(profile)) + '\n', {flag: 'wx'});
async function save(name, options, kind) {
  const bundle = await runPhaseEpisode(spec, options), p = bundle.episode.payload;
  await replayPhaseEpisode(bundle);
  const bytes = gzipSync(Buffer.from(canonical(bundle) + '\n'), {level: 6});
  writeFileSync(`${out}/${name}.json.gz`, bytes, {flag: 'wx'});
  records.push({name, file: `${name}.json.gz`, evidence: kind, manifest: `${p.fixture ? 'constructed' : 'official'}.manifest.json`,
    bytes: bytes.length, sha256: digest(bytes), episode_hash: bundle.episode.content_hash,
    replay_verified: true, decisions: p.decisions.length, status: p.outcome.status,
    phases: [...new Set(p.frames.map(f => f.phase))], rejections: p.decisions.filter(d => !d.accepted).length});
  return bundle;
}
const natural = await save('natural-p1-first', {}, 'natural');
await save('natural-p2-first', {order: 'p2-first'}, 'natural');
await save('truncation-joint', {truncateAfter: 1}, 'natural-truncation');
const rejected = natural.episode.payload.decisions.find(d => !d.accepted);
if (!rejected) throw new Error('Required natural rejection fixture disappeared');
await save('truncation-at-rejection', {truncateAfter: rejected.sequence + 1}, 'natural-truncation');
for (const fixture of FIXTURES) await save(fixture, {fixture, truncateAfter: fixture === 'tie-after-first-choice' ? null : 1}, 'constructed');
const index = seal({version: 'r4-fixtures-1', producer_sha256: fileHash(new URL(import.meta.url)), records,
  scope: 'Constructed phase probes are not evidence of natural timing or reachability; natural bundles preserve actual reference mechanics.'});
writeFileSync(`${out}/index.json`, canonical(index) + '\n', {flag: 'wx'});
console.log(JSON.stringify({fixtures: records.length, out, content_hash: index.content_hash}));
