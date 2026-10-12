import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {featureFixtures} from '../../scripts/player-feature-fixtures.mjs';
import {canonical} from '../../packages/battle-contract/canonical.mjs';

test('R5 live hidden-state, HP, Transform, Illusion and Node/Python codec probes', async () => {
  const fixtures = await featureFixtures();
  const result = spawnSync(process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['-m', 'pokemon_reference.feature_checks'],
    {input: canonical(fixtures), encoding: 'utf8', env: {...process.env, PYTHONPATH: 'python'}, maxBuffer: 2 ** 24});
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.invariant_pairs, 18); assert.equal(report.positive_controls, 2);
  assert.equal(report.semantic_probes, 14); assert.equal(report.request_cases, 16);
  assert.ok(report.node_python_action_round_trips > 50);
});
