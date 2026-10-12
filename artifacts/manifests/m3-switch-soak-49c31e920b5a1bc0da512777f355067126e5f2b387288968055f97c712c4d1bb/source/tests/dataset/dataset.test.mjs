import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {POLICY, teamKey, partition, partitionDigest, pairPartition} from '../../packages/dataset/identity.mjs';
import {checkJobSeeds} from '../../packages/dataset/collect.mjs';

const payload = JSON.parse(readFileSync('tests/dataset/fixtures/team.json')).payload;
test('Node and independent Python agree on ordered team identities and full-digest partitions', () => {
  const values = Array.from({length: 100}, (_, i) => { const p = structuredClone(payload); p.pokemon[0].level = i + 1; return p; });
  const result = spawnSync('.venv/bin/python', ['-c',
    'import json,sys;from pokemon_reference.dataset_identity import team_key,partition;p=json.load(sys.stdin);print(json.dumps([[team_key(x,p["policy"]),partition(x,p["policy"])] for x in p["teams"]]))'],
    {input: JSON.stringify({policy: POLICY, teams: values}), encoding: 'utf8', env: {...process.env, PYTHONPATH: 'python'}});
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(values.map(p => [teamKey(p), partition(p)]), JSON.parse(result.stdout));
  const one = values[0], two = values.find(p => partition(p) !== partition(one));
  assert.ok(two); assert.throws(() => pairPartition(one, two), /Cross-partition/);
});

test('partition edges use the entire 256-bit digest with exact rational thresholds', () => {
  const space = 1n << 256n, hex = n => n.toString(16).padStart(64, '0');
  const firstDev = (space * 9000n + 9999n) / 10000n, firstFinal = (space * 9500n + 9999n) / 10000n;
  for (const [n, expected] of [[0n, 'train'], [firstDev - 1n, 'train'], [firstDev, 'development'],
    [firstFinal - 1n, 'development'], [firstFinal, 'final'], [space - 1n, 'final']]) assert.equal(partitionDigest(hex(n)), expected);
  assert.throws(() => partitionDigest('0'), /full partition digest/);
});

test('new mechanical fields reject and exposed/cross-role battle seeds fail before collection', () => {
  const p = structuredClone(payload); p.pokemon[0].nature = 'Adamant';
  assert.throws(() => teamKey(p), /schema/);
  const job = {split: 'train', group: 'pair', spec: {mechanics_seed: 'm', policy_seeds: ['p', 'q']}};
  assert.throws(() => checkJobSeeds([job], {}, new Set(['m'])), /seed overlaps/);
  assert.doesNotThrow(() => checkJobSeeds([job, structuredClone(job)], {}, new Set()));
  const changed = structuredClone(job); changed.spec.policy_seeds[0] = 'm';
  assert.throws(() => checkJobSeeds([job, changed], {}, new Set()), /seed overlaps/);
});
