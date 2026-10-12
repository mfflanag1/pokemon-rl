import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, rmSync, readFileSync, writeFileSync, copyFileSync, readdirSync, unlinkSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir, hostname} from 'node:os';
import {join} from 'node:path';
import {generateCorpus, checkChunk} from '../../packages/corpus/generate.mjs';
import {replayCorpus, replayIndices} from '../../packages/corpus/replay.mjs';
import {generateTeam} from '../../packages/reference-node/engine.mjs';
import {seedAt, readJSON, validateIndex, seal} from '../../packages/corpus/common.mjs';
import {canonical, parseJSON} from '../../packages/battle-contract/canonical.mjs';
import {sourceCatalog} from '../../packages/corpus/catalog.mjs';
import {buildVocabulary} from '../../packages/reference-node/contract-v1.mjs';

const temporary = () => mkdtempSync(join(tmpdir(), 'pokemon-r3-test-'));
const config = out => ({out, namespace: 'r3-test', count: 5, shardSize: 2});

test('vocabulary includes lazily resolved cosmetic forms and templates retain distinct identities', () => {
  const catalog = sourceCatalog(), vocabulary = buildVocabulary();
  const species = new Set(vocabulary.domains.species);
  assert.equal(vocabulary.version, 'showdown-dex-ids-v2');
  assert.ok(species.has('florgesblue'));
  for (const resolved of Object.values(catalog.resolvers.species)) assert.ok(species.has(resolved.id), resolved.id);
  assert.equal(Object.keys(catalog.entries).length, 509);
  assert.equal(Object.values(catalog.entries).reduce((n, entry) => n + entry.sets.length, 0), 879);
  const sameRole = catalog.entries.grumpig.sets.filter(s => s.role === 'Bulky Setup');
  assert.equal(sameRole.length, 2);
  assert.notEqual(sameRole[0].index, sameRole[1].index);
  assert.notEqual(sameRole[0].template_hash, sameRole[1].template_hash);
});

test('whole teams are complete, ordered, deterministic across shards, and replay exactly', () => {
  const a = temporary(), b = temporary();
  try {
    const first = generateCorpus(config(a)), second = generateCorpus(config(b));
    assert.deepEqual(first, second);
    assert.deepEqual(first.chunks.map(c => c.count), [2, 2, 1]);
    let row = 0;
    for (const chunk of first.chunks) for (const line of checkChunk(a, chunk).toString().trimEnd().split('\n')) {
      const artifact = parseJSON(line);
      assert.equal(artifact.payload.generator_seed.hex, seedAt('r3-test', row++).slice(7));
      assert.equal(artifact.payload.pokemon.length, 6);
      assert.equal(artifact.payload.ids.length, 6);
    }
    assert.equal(row, 5);
    assert.deepEqual(generateCorpus(config(a)), first);
    const replay = replayCorpus({out: a, samples: 5});
    assert.equal(replay.sample_count, 5);
    assert.equal(replay.verified, true);
    assert.deepEqual(replayIndices(first, 5), [0, 1, 2, 3, 4]);
    assert.throws(() => replayIndices(first, 0));
  } finally { rmSync(a, {recursive: true}); rmSync(b, {recursive: true}); }
});

test('failed shard is never committed or skipped; resume records the failure and reproduces content', () => {
  for (const failAt of [0, 3]) {
    const directory = temporary();
    try {
      let calls = 0;
      assert.throws(() => generateCorpus(config(directory), seed => {
        if (calls++ === failAt) throw new Error('injected generator failure');
        return generateTeam(seed);
      }), /injected/);
      const progress = readJSON(`${directory}/progress.json`);
      assert.equal(progress.completed, failAt === 0 ? 0 : 2);
      const failure = JSON.parse(readFileSync(`${directory}/failures.jsonl`, 'utf8').trim());
      assert.equal(failure.attempted_index, failAt);
      assert.equal(failure.seed, seedAt('r3-test', failAt));
      if (failAt > 0) copyFileSync(`${directory}/teams-000000.jsonl.gz`, `${directory}/teams-000001.jsonl.gz`);
      const result = generateCorpus(config(directory));
      assert.equal(result.completed, 5);
      assert.equal(result.failure_events, 1);
      assert.equal(replayCorpus({out: directory, samples: 5}).sample_count, 5);
      if (failAt > 0) assert.equal(readdirSync(`${directory}/orphans`).length, 1);
    } finally { rmSync(directory, {recursive: true}); }
  }
});

test('wrong resume configuration and corrupt shards fail without rewriting a completed corpus', () => {
  const directory = temporary();
  try {
    const index = generateCorpus(config(directory));
    const before = readFileSync(`${directory}/index.json`);
    assert.throws(() => generateCorpus({...config(directory), namespace: 'different'}), /differs/);
    assert.deepEqual(readFileSync(`${directory}/index.json`), before);
    const file = `${directory}/${index.chunks[0].file}`, bytes = readFileSync(file);
    bytes[bytes.length - 1] ^= 1; writeFileSync(file, bytes);
    assert.throws(() => generateCorpus(config(directory)), /integrity/);
    assert.throws(() => replayCorpus({out: directory, samples: 1}), /integrity/);
    const missing = structuredClone(index); missing.chunks.splice(1, 1);
    const {content_hash, ...body} = missing;
    assert.throws(() => validateIndex(seal(body)), /shard/);
  } finally { rmSync(directory, {recursive: true}); }
});

test('active writer lock prevents a second writer', () => {
  const directory = temporary();
  try {
    writeFileSync(`${directory}/.writer-lock.json`, canonical({pid: process.pid, host: hostname()}));
    assert.throws(() => generateCorpus(config(directory)), /writer is active/);
  } finally { rmSync(directory, {recursive: true}); }
});

test('checkpoint failure reports the durable prefix and resumes through an orphan shard', () => {
  const directory = temporary(), checkpointTemp = `${directory}/progress.json.tmp-${process.pid}`;
  try {
    assert.throws(() => generateCorpus({...config(directory), onProgress: ({completed}) => {
      if (completed === 2) writeFileSync(checkpointTemp, 'injected interrupted write', {flag: 'wx'});
    }}), /EEXIST/);
    assert.equal(readJSON(`${directory}/progress.json`).completed, 2);
    const failure = readJSON(`${directory}/failures.jsonl`);
    assert.equal(failure.phase, 'checkpoint');
    assert.equal(failure.committed_records, 2);
    assert.equal(failure.attempted_index, null);
    assert.equal(failure.seed, null);
    unlinkSync(checkpointTemp);
    assert.equal(generateCorpus(config(directory)).completed, 5);
    assert.equal(readdirSync(`${directory}/orphans`).length, 1);
    assert.equal(replayCorpus({out: directory, samples: 5}).verified, true);
  } finally { rmSync(directory, {recursive: true}); }
});

test('dead local writer is recovered only with exclusive recovery ownership', () => {
  const directory = temporary();
  try {
    const child = spawnSync(process.execPath, ['-e', 'process.stdout.write(String(process.pid))'], {encoding: 'utf8'});
    assert.equal(child.status, 0);
    const dead = canonical({pid: Number(child.stdout), host: hostname()});
    writeFileSync(`${directory}/.writer-lock.json`, dead);
    writeFileSync(`${directory}/.writer-recovery.json`, canonical({pid: process.pid, host: hostname()}));
    assert.throws(() => generateCorpus(config(directory)), /EEXIST/);
    assert.equal(readFileSync(`${directory}/.writer-lock.json`, 'utf8'), dead);
    unlinkSync(`${directory}/.writer-recovery.json`);
    assert.equal(generateCorpus(config(directory)).completed, 5);
  } finally { rmSync(directory, {recursive: true}); }
});

test('leftover recovery guard does not prevent a fresh exclusive writer acquisition', () => {
  const directory = temporary();
  try {
    const guard = canonical({pid: process.pid, host: hostname()});
    writeFileSync(`${directory}/.writer-recovery.json`, guard);
    assert.equal(generateCorpus(config(directory)).completed, 5);
    assert.equal(readFileSync(`${directory}/.writer-recovery.json`, 'utf8'), guard);
  } finally { rmSync(directory, {recursive: true}); }
});
