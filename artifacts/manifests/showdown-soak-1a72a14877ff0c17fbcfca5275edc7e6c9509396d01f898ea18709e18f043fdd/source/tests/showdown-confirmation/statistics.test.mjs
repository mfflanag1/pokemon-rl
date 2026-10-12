import test from 'node:test';
import assert from 'node:assert/strict';
import {confirm} from '../../packages/showdown-confirmation/statistics.mjs';

const ids = ['baseline-v1', 'random-v1', 'damage-v1', 'tactical-v1'];
const options = {bootstrap_samples: 60, seed: 91, family_alpha: .05, comparisons: 3,
  minimum_direct_score: .52, direct_lower_threshold: .5, opponent_delta_lower_threshold: -.03};
function grid() {
  return ids.flatMap((a, i) => ids.slice(i + 1).map(b => ({a, b, pairs: 3,
    pair_scores: b === 'tactical-v1' && a === 'baseline-v1' ? [.25, .25, .25] : [.5, .5, .5]})));
}
test('constant-pair known answer passes all conditions without treating games as independent', () => {
  const result = confirm(grid(), options);
  assert.equal(result.status, 'pass'); assert.equal(result.lower_order_statistic, 1);
  assert.deepEqual(result.metrics.map(m => [m.name, m.mean, m.lower, m.passed]),
    [['direct', .75, .75, true], ['random_delta', 0, 0, true], ['damage_delta', 0, 0, true]]);
});
test('boundaries use strict lower comparisons and an inclusive direct point floor', () => {
  assert.equal(confirm(grid(), {...options, direct_lower_threshold: .75}).status, 'not-passed');
  assert.equal(confirm(grid(), {...options, opponent_delta_lower_threshold: 0}).status, 'not-passed');
  assert.equal(confirm(grid(), {...options, minimum_direct_score: .75}).status, 'pass');
  assert.equal(confirm(grid(), {...options, minimum_direct_score: .76}).direct_point_passed, false);
});
test('an opponent-specific regression cannot be hidden by direct strength', () => {
  const m = grid(); m.find(e => e.a === 'random-v1' && e.b === 'tactical-v1').pair_scores = [1, 1, 1];
  const result = confirm(m, options); assert.equal(result.metrics[0].passed, true);
  assert.equal(result.metrics[1].mean, -.5); assert.equal(result.status, 'not-passed');
});
test('input order does not change output and inputs are not mutated', () => {
  const m = grid(), before = JSON.stringify([m, options]);
  for (const e of m) { Object.freeze(e.pair_scores); Object.freeze(e); } Object.freeze(m);
  const one = confirm(m, options), two = confirm([...m].reverse(), options);
  assert.deepEqual(one, two); assert.equal(JSON.stringify([m, options]), before);
});
test('missing, duplicate, reversed and invalid pair grids fail before resampling', () => {
  const mutations = [m => m.pop(), m => { m[1] = m[0]; }, m => { [m[0].a, m[0].b] = [m[0].b, m[0].a]; },
    m => { m[0].a = 'foreign'; }, m => { m[0].pairs = 4; }, m => { m[0].pair_scores.pop(); },
    m => { m[0].pair_scores[1] = NaN; }, m => { m[0].pair_scores[1] = .1; },
    m => { delete m[0].pair_scores[1]; }, m => { m[0] = null; }];
  for (const mutate of mutations) { const m = grid(); mutate(m); assert.throws(() => confirm(m, options), /Invalid confirmation/); }
  for (const m of [null, {}, [], 'matrix']) assert.throws(() => confirm(m, options), /Invalid confirmation/);
});
test('invalid statistical options fail explicitly', () => {
  for (const o of [null, [], {}, {...options, bootstrap_samples: 0}, {...options, bootstrap_samples: 1.5},
    {...options, seed: -1}, {...options, seed: 2 ** 32}, {...options, seed: '9'},
    {...options, family_alpha: 0}, {...options, family_alpha: 1}, {...options, family_alpha: NaN},
    {...options, comparisons: 2}, {...options, minimum_direct_score: 1.1},
    {...options, direct_lower_threshold: -.1}, {...options, opponent_delta_lower_threshold: -1.1}]) {
    assert.throws(() => confirm(grid(), o), /Invalid confirmation/);
  }
});
