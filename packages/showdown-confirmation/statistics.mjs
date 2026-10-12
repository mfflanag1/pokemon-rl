import {mulberry32, uniformIndex} from '../showdown-panel/statistics.mjs';

const IDS = ['baseline-v1', 'random-v1', 'damage-v1', 'tactical-v1'];
const valid = (ok, why) => { if (!ok) throw new Error(`Invalid confirmation input: ${why}`); };
const finite = n => typeof n === 'number' && Number.isFinite(n);
const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;

/** Fixed three-comparison development screen. Whole-pair resampling preserves
 * paired differences and shared teams across edges. Nominal Bonferroni percentile
 * bounds are approximate, not finite-sample or final-evaluation guarantees. */
export function confirm(matrix, options) {
  valid(options && typeof options === 'object' && !Array.isArray(options), 'options');
  const {bootstrap_samples: samples, seed, family_alpha: alpha, comparisons,
    minimum_direct_score: minimum, direct_lower_threshold: direct,
    opponent_delta_lower_threshold: delta} = options;
  valid(Number.isSafeInteger(samples) && samples > 0, 'bootstrap_samples');
  valid(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff, 'seed');
  valid(finite(alpha) && alpha > 0 && alpha < 1 && comparisons === 3, 'family error allocation');
  const rank = Math.ceil(samples * alpha / comparisons);
  valid(rank >= 1 && rank <= samples, 'representable lower quantile');
  valid(finite(minimum) && minimum >= 0 && minimum <= 1, 'minimum_direct_score');
  valid(finite(direct) && direct >= 0 && direct <= 1, 'direct_lower_threshold');
  valid(finite(delta) && delta >= -1 && delta <= 1, 'opponent_delta_lower_threshold');
  valid(Array.isArray(matrix) && matrix.length === 6, 'six canonical edges');
  const edges = new Map(); let pairs;
  for (const edge of matrix) {
    valid(edge && typeof edge === 'object', 'edge');
    const i = IDS.indexOf(edge.a), j = IDS.indexOf(edge.b), key = `${i}:${j}`;
    valid(i >= 0 && j > i && !edges.has(key), 'foreign/reversed/duplicate edge');
    valid(Number.isSafeInteger(edge.pairs) && edge.pairs > 0, 'pairs');
    pairs ??= edge.pairs;
    valid(edge.pairs === pairs && Array.isArray(edge.pair_scores) && edge.pair_scores.length === pairs, 'aligned pair grid');
    valid(Array.from(edge.pair_scores).every(s => finite(s) && [0, .25, .5, .75, 1].includes(s)), 'pair score');
    edges.set(key, edge.pair_scores);
  }
  // Six unique canonical edges among four fixed policies imply the complete grid.
  const score = (i, j, p) => edges.get(`${i}:${j}`)[p];
  const values = [[], [], []];
  for (let p = 0; p < pairs; p++) {
    values[0].push(1 - score(0, 3, p));
    values[1].push(1 - score(1, 3, p) - score(0, 1, p));
    values[2].push(1 - score(2, 3, p) - score(0, 2, p));
  }
  const next = mulberry32(seed), draws = values.map(() => new Float64Array(samples));
  for (let b = 0; b < samples; b++) {
    const sums = [0, 0, 0];
    for (let p = 0; p < pairs; p++) {
      const index = uniformIndex(next, pairs);
      for (let m = 0; m < 3; m++) sums[m] += values[m][index];
    }
    for (let m = 0; m < 3; m++) draws[m][b] = sums[m] / pairs;
  }
  const metrics = ['direct', 'random_delta', 'damage_delta'].map((name, m) => {
    const threshold = m === 0 ? direct : delta, lower = draws[m].sort()[rank - 1];
    return {name, mean: mean(values[m]), lower, threshold, passed: lower > threshold};
  });
  const directPointPassed = metrics[0].mean >= minimum;
  return {status: directPointPassed && metrics.every(m => m.passed) ? 'pass' : 'not-passed', pairs,
    bootstrap_samples: samples, seed, family_alpha: alpha, comparisons, lower_order_statistic: rank,
    metrics, minimum_direct_score: minimum, direct_point_passed: directPointPassed,
    interpretation: 'Nominal approximate Bonferroni whole-pair bootstrap development screen; no finite-sample, final, ladder or G2/G3 guarantee.'};
}
