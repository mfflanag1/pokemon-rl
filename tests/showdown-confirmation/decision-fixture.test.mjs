import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {summarizeMatrix} from '../../packages/showdown-panel/statistics.mjs';
import {confirm} from '../../packages/showdown-confirmation/statistics.mjs';

const plan = JSON.parse(readFileSync(new URL('../../packages/showdown-confirmation/plan.json', import.meta.url)));
test('coordinator literal fixture checks all three metric orientations and shared pair indices', () => {
  // Edges in the sealed policy order. Quarter-valued pair scores are encoded as
  // two legal game outcomes; no bootstrap oracle computes the expected answer.
  const edgePairs = [[.5, .5, .5], [1, .5, 0], [1, .5, 0], [.25, .5, .75], [0, .5, 1], [1, .5, 0]];
  const mirrors = {0: [0, 0], 0.25: [0, .5], 0.5: [1, 0], 0.75: [.5, 1], 1: [1, 1]};
  const records = plan.matchups.flatMap(([a, b], e) => edgePairs[e].flatMap((s, pair) =>
    mirrors[s].map((score, mirror) => ({a, b, pair, mirror, score}))));
  const matrix = summarizeMatrix(records, {policyIds: plan.policy_ids, pairs: 3, bootstrapSamples: 5, seed: 0});
  const result = confirm(matrix, {...plan.confirmation, bootstrap_samples: 5, seed: 0});
  // Independently established seed-0 indices are [2,1,1, 0,0,2, 2,2,0, 0,2,0, 1,0,2].
  // Direct pairs [0,.5,1], random deltas [.5,0,-.5], damage deltas [-1,0,1].
  // B=5 gives rank 1. Expected minima are literal rational values below.
  assert.deepEqual(result.metrics.map(m => [m.name, m.mean, m.lower]),
    [['direct', .5, 1 / 3], ['random_delta', 0, -1 / 6], ['damage_delta', 0, -1 / 3]]);
  assert.equal(result.direct_point_passed, false); assert.equal(result.status, 'not-passed');
});
