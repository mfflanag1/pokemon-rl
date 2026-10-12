/**
 * Showdown tactical-confirmation statistics.
 *
 * Pure, engine-free screen over the six-edge mirrored-pair matrix produced by
 * `summarizeMatrix` (../showdown-panel/statistics.mjs) for the fixed policy
 * order `baseline-v1, random-v1, damage-v1, tactical-v1`. It derives three
 * per-pair metrics for the candidate `tactical-v1`, resamples whole pairs with
 * ONE shared index stream across all metrics, and reports nominal one-sided
 * Bonferroni lower bounds per packages/showdown-confirmation/plan.json
 * (`confirmation` block). Nothing here is an alternative inferential method;
 * the plan's procedure is implemented literally.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Input contract: confirm(matrix, options)
 * ──────────────────────────────────────────────────────────────────────────
 * matrix   Array of exactly 6 edge summaries in the canonical lexicographic
 *          order of the policy list above, i.e. EXACTLY (and in this order):
 *            [0] baseline-v1 vs random-v1
 *            [1] baseline-v1 vs damage-v1
 *            [2] baseline-v1 vs tactical-v1
 *            [3] random-v1   vs damage-v1
 *            [4] random-v1   vs tactical-v1
 *            [5] damage-v1   vs tactical-v1
 *          Each entry must carry `a`, `b` (the canonical ids above — reversed
 *          orientation, duplicates, unknown ids and re-ordering are rejected,
 *          never re-interpreted), a positive safe integer `pairs`, and
 *          `pair_scores` (array or typed array) of length `pairs` whose
 *          entries are each one of 0, 0.25, 0.5, 0.75, 1 (mirror-averaged
 *          scores from `a`'s point of view, as summarizeMatrix emits). All six
 *          edges must share the same `pairs`. Extra properties (games, wins,
 *          interval95, …) are ignored. Non-finite or otherwise malformed
 *          values throw `ConfirmationValidationError`.
 *
 * options  The plan's `confirmation` object. ALL of the following are
 *          required (no defaults; an incomplete object is rejected). Extra
 *          keys such as the plan's `quantile` prose are ignored.
 *            bootstrap_samples               positive safe integer B
 *            seed                            integer in [0, 2^32 − 1]
 *            family_alpha                    finite number in (0, 1)
 *            comparisons                     exactly 3
 *            minimum_direct_score            finite number in [0, 1]
 *            direct_lower_threshold          finite number in [0, 1]
 *            opponent_delta_lower_threshold  finite number in [−1, 1]
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Metrics (per pair p; all exact in binary floating point)
 * ──────────────────────────────────────────────────────────────────────────
 *   direct[p]       = 1 − BT[p]                    tactical score vs baseline
 *   random_delta[p] = (1 − RT[p]) − BR[p]           tactical minus baseline
 *                                                   score against random
 *   damage_delta[p] = (1 − DT[p]) − BD[p]           tactical minus baseline
 *                                                   score against damage
 * where BT = matrix[2].pair_scores, BR = matrix[0], BD = matrix[1],
 * RT = matrix[4], DT = matrix[5] (all from the first-named policy's view).
 * `direct` lies in [0, 1]; the deltas lie in [−1, 1]. The random-vs-damage
 * edge (matrix[3]) is validated but enters no metric.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Bootstrap (deterministic)
 * ──────────────────────────────────────────────────────────────────────────
 * PRNG: `mulberry32(seed)` and `uniformIndex` imported from the sealed panel
 * module. For replicate s = 0..B−1, for p = 0..pairs−1, draw index[p] from
 * the single stream; then EVERY metric's replicate s is the mean of its
 * per-pair values at those same indices (whole-pair resampling with common
 * indices across all metrics, so differences are paired, not pooled).
 *
 * Lower bound: with the B replicates of a metric sorted ascending,
 *   rank k = ⌈B · family_alpha / comparisons⌉,   lower = k-th smallest.
 * k is computed from the double-precision product; if that product sits
 * within 1e−9 (relative) of an integer ≥ 1 it is taken as that integer so
 * that 1-ulp artefacts of binary decimals (e.g. 300·0.07/3) cannot bump the
 * rank. k always satisfies 1 ≤ k ≤ B because family_alpha ∈ (0, 1).
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Decision rules
 * ──────────────────────────────────────────────────────────────────────────
 *   direct        passed ⇔ lower >  direct_lower_threshold        (strict)
 *   random_delta  passed ⇔ lower >  opponent_delta_lower_threshold (strict)
 *   damage_delta  passed ⇔ lower >  opponent_delta_lower_threshold (strict)
 *   direct_point_passed ⇔ mean(direct) >= minimum_direct_score    (inclusive)
 *   status = 'pass' iff all three `passed` AND direct_point_passed,
 *            else 'not-passed'.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Result (exact interface)
 * ──────────────────────────────────────────────────────────────────────────
 *   {
 *     status,                'pass' | 'not-passed'
 *     pairs,                 shared number of pairs per edge
 *     bootstrap_samples,     echoed B
 *     seed,                  echoed seed
 *     family_alpha,          echoed
 *     comparisons,           echoed (3)
 *     rank,                  k used for every lower bound (see above)
 *     metrics: [             always in this order
 *       { name: 'direct',       mean, lower, threshold, passed, paired_scores },
 *       { name: 'random_delta', mean, lower, threshold, passed, paired_scores },
 *       { name: 'damage_delta', mean, lower, threshold, passed, paired_scores },
 *     ],                     paired_scores: fresh Array(pairs) of per-pair
 *                            metric values; threshold: the applied threshold
 *     minimum_direct_score,  echoed
 *     direct_point_passed,   mean(direct) >= minimum_direct_score
 *     interpretation,        INTERPRETATION string below
 *   }
 * Inputs are never mutated; every returned object/array is fresh.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Interpretation
 * ──────────────────────────────────────────────────────────────────────────
 * This is an approximate nominal development screen only. The bounds are
 * nominal one-sided Bonferroni bootstrap quantiles and carry no finite-sample
 * coverage guarantee; the status is not a final evaluation and says nothing
 * about G2, G3, ladder or public-server play.
 */

import { mulberry32, uniformIndex } from '../showdown-panel/statistics.mjs';

export const POLICY_IDS = Object.freeze(['baseline-v1', 'random-v1', 'damage-v1', 'tactical-v1']);

export const EDGES = Object.freeze([
  Object.freeze(['baseline-v1', 'random-v1']),
  Object.freeze(['baseline-v1', 'damage-v1']),
  Object.freeze(['baseline-v1', 'tactical-v1']),
  Object.freeze(['random-v1', 'damage-v1']),
  Object.freeze(['random-v1', 'tactical-v1']),
  Object.freeze(['damage-v1', 'tactical-v1']),
]);

const EDGE_BASELINE_RANDOM = 0;
const EDGE_BASELINE_DAMAGE = 1;
const EDGE_BASELINE_TACTICAL = 2;
const EDGE_RANDOM_TACTICAL = 4;
const EDGE_DAMAGE_TACTICAL = 5;

export const METRIC_NAMES = Object.freeze(['direct', 'random_delta', 'damage_delta']);
export const VALID_PAIR_SCORES = Object.freeze([0, 0.25, 0.5, 0.75, 1]);
export const REQUIRED_COMPARISONS = 3;

const RANK_TOLERANCE = 1e-9;

export const INTERPRETATION =
  'Approximate nominal development screen only. Each lower bound is the ' +
  'ceil(B * family_alpha / comparisons)-th smallest shared-pair bootstrap mean, ' +
  'a one-sided nominal Bonferroni bound over 3 comparisons with no finite-sample ' +
  'coverage guarantee. The status is a development-split screening outcome for ' +
  'this admitted pair set and format; it is not a final evaluation and implies ' +
  'no G2, G3, ladder or public-server performance guarantee.';

export class ConfirmationValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ConfirmationValidationError';
  }
}

function fail(message) {
  throw new ConfirmationValidationError(message);
}

function describe(value) {
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number' || typeof value === 'boolean' || value == null) return String(value);
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isArrayLike(value) {
  return Array.isArray(value) || ArrayBuffer.isView(value);
}

/**
 * Validate a plan `confirmation` object. Returns a fresh object holding only
 * the seven recognised fields.
 */
export function validateOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    fail(`options must be an object (plan.confirmation); got ${describe(options)}`);
  }
  const {
    bootstrap_samples,
    seed,
    family_alpha,
    comparisons,
    minimum_direct_score,
    direct_lower_threshold,
    opponent_delta_lower_threshold,
  } = options;

  if (!Number.isSafeInteger(bootstrap_samples) || bootstrap_samples < 1) {
    fail(`options.bootstrap_samples must be a positive safe integer; got ${describe(bootstrap_samples)}`);
  }
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    fail(`options.seed must be an integer in [0, 2^32 - 1]; got ${describe(seed)}`);
  }
  if (!isFiniteNumber(family_alpha) || !(family_alpha > 0 && family_alpha < 1)) {
    fail(`options.family_alpha must be a finite number in (0, 1); got ${describe(family_alpha)}`);
  }
  if (comparisons !== REQUIRED_COMPARISONS) {
    fail(`options.comparisons must be exactly ${REQUIRED_COMPARISONS}; got ${describe(comparisons)}`);
  }
  if (!isFiniteNumber(minimum_direct_score) || minimum_direct_score < 0 || minimum_direct_score > 1) {
    fail(`options.minimum_direct_score must be a finite number in [0, 1]; got ${describe(minimum_direct_score)}`);
  }
  if (!isFiniteNumber(direct_lower_threshold) || direct_lower_threshold < 0 || direct_lower_threshold > 1) {
    fail(`options.direct_lower_threshold must be a finite number in [0, 1]; got ${describe(direct_lower_threshold)}`);
  }
  if (
    !isFiniteNumber(opponent_delta_lower_threshold) ||
    opponent_delta_lower_threshold < -1 ||
    opponent_delta_lower_threshold > 1
  ) {
    fail(
      `options.opponent_delta_lower_threshold must be a finite number in [-1, 1]; got ${describe(opponent_delta_lower_threshold)}`,
    );
  }
  return {
    bootstrap_samples,
    seed,
    family_alpha,
    comparisons,
    minimum_direct_score,
    direct_lower_threshold,
    opponent_delta_lower_threshold,
  };
}

/**
 * Validate the six-edge matrix (see module header). Returns the shared number
 * of pairs. Does not mutate the input.
 */
export function validateMatrix(matrix) {
  if (!Array.isArray(matrix)) {
    fail(`matrix must be an array of ${EDGES.length} edge summaries (summarizeMatrix output); got ${describe(matrix)}`);
  }
  if (matrix.length !== EDGES.length) {
    fail(`matrix must have exactly ${EDGES.length} edges in canonical order; got ${matrix.length}`);
  }
  let pairs = null;
  matrix.forEach((edge, i) => {
    if (edge === null || typeof edge !== 'object' || Array.isArray(edge)) {
      fail(`matrix[${i}] must be an edge summary object; got ${describe(edge)}`);
    }
    const [a, b] = EDGES[i];
    if (edge.a !== a || edge.b !== b) {
      fail(
        `matrix[${i}] must be the canonical edge ${JSON.stringify(a)} vs ${JSON.stringify(b)}; got ${describe(edge.a)} vs ${describe(edge.b)}`,
      );
    }
    const scores = edge.pair_scores;
    if (!isArrayLike(scores)) {
      fail(`matrix[${i}].pair_scores must be a non-empty array; got ${describe(scores)}`);
    }
    if (scores.length < 1) {
      fail(`matrix[${i}].pair_scores must be a non-empty array; got an empty one`);
    }
    if (!Number.isSafeInteger(edge.pairs) || edge.pairs < 1 || edge.pairs !== scores.length) {
      fail(
        `matrix[${i}].pairs must be a positive safe integer equal to pair_scores.length (${scores.length}); got ${describe(edge.pairs)}`,
      );
    }
    if (pairs === null) {
      pairs = scores.length;
    } else if (scores.length !== pairs) {
      fail(`matrix[${i}] has ${scores.length} pairs but matrix[0] has ${pairs}; every edge must share the same pairs`);
    }
    for (let p = 0; p < scores.length; p += 1) {
      const v = scores[p];
      if (typeof v !== 'number' || !VALID_PAIR_SCORES.includes(v)) {
        fail(`matrix[${i}].pair_scores[${p}] must be one of 0, 0.25, 0.5, 0.75, 1; got ${describe(v)}`);
      }
    }
  });
  return pairs;
}

function computeMetricArrays(matrix, pairs) {
  const br = matrix[EDGE_BASELINE_RANDOM].pair_scores;
  const bd = matrix[EDGE_BASELINE_DAMAGE].pair_scores;
  const bt = matrix[EDGE_BASELINE_TACTICAL].pair_scores;
  const rt = matrix[EDGE_RANDOM_TACTICAL].pair_scores;
  const dt = matrix[EDGE_DAMAGE_TACTICAL].pair_scores;

  const direct = new Float64Array(pairs);
  const randomDelta = new Float64Array(pairs);
  const damageDelta = new Float64Array(pairs);
  for (let p = 0; p < pairs; p += 1) {
    direct[p] = 1 - bt[p];
    randomDelta[p] = (1 - rt[p]) - br[p];
    damageDelta[p] = (1 - dt[p]) - bd[p];
  }
  return [direct, randomDelta, damageDelta]; // METRIC_NAMES order
}

/**
 * Validate the matrix and return the three per-pair metric arrays as plain
 * arrays: `{ pairs, direct, random_delta, damage_delta }`.
 */
export function deriveMetrics(matrix) {
  const pairs = validateMatrix(matrix);
  const [direct, randomDelta, damageDelta] = computeMetricArrays(matrix, pairs);
  return {
    pairs,
    direct: Array.from(direct),
    random_delta: Array.from(randomDelta),
    damage_delta: Array.from(damageDelta),
  };
}

function meanOf(values) {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += values[i];
  return sum / values.length;
}

/**
 * Rank k = ⌈B · familyAlpha / comparisons⌉ of the one-sided lower bound, with
 * the 1e−9 integer snap documented in the module header. 1 ≤ k ≤ B.
 */
export function bonferroniRank(bootstrapSamples, familyAlpha, comparisons) {
  if (!Number.isSafeInteger(bootstrapSamples) || bootstrapSamples < 1) {
    fail(`bonferroniRank: bootstrapSamples must be a positive safe integer; got ${describe(bootstrapSamples)}`);
  }
  if (!isFiniteNumber(familyAlpha) || !(familyAlpha > 0 && familyAlpha < 1)) {
    fail(`bonferroniRank: familyAlpha must be a finite number in (0, 1); got ${describe(familyAlpha)}`);
  }
  if (!Number.isSafeInteger(comparisons) || comparisons < 1) {
    fail(`bonferroniRank: comparisons must be a positive safe integer; got ${describe(comparisons)}`);
  }
  const raw = (bootstrapSamples * familyAlpha) / comparisons;
  const nearest = Math.round(raw);
  let k;
  if (nearest >= 1 && Math.abs(raw - nearest) <= RANK_TOLERANCE * Math.max(1, raw)) {
    k = nearest;
  } else {
    k = Math.ceil(raw);
  }
  if (k < 1) k = 1;
  if (k > bootstrapSamples) k = bootstrapSamples;
  return k;
}

function bootstrapLowerBounds(metrics, pairs, bootstrapSamples, seed, rank) {
  const next = mulberry32(seed);
  const replicates = metrics.map(() => new Float64Array(bootstrapSamples));
  const indices = new Array(pairs);
  for (let s = 0; s < bootstrapSamples; s += 1) {
    for (let p = 0; p < pairs; p += 1) {
      indices[p] = uniformIndex(next, pairs);
    }
    for (let m = 0; m < metrics.length; m += 1) {
      const values = metrics[m];
      let sum = 0;
      for (let p = 0; p < pairs; p += 1) sum += values[indices[p]];
      replicates[m][s] = sum / pairs;
    }
  }
  return replicates.map((sample) => {
    sample.sort(); // TypedArray sort: numeric ascending
    return sample[rank - 1];
  });
}

/**
 * Run the confirmation screen. See the module header for the full contract.
 */
export function confirm(matrix, options) {
  const opts = validateOptions(options);
  const pairs = validateMatrix(matrix);
  const metrics = computeMetricArrays(matrix, pairs);
  const means = metrics.map(meanOf);
  const rank = bonferroniRank(opts.bootstrap_samples, opts.family_alpha, opts.comparisons);
  const lowers = bootstrapLowerBounds(metrics, pairs, opts.bootstrap_samples, opts.seed, rank);

  const thresholds = [
    opts.direct_lower_threshold,
    opts.opponent_delta_lower_threshold,
    opts.opponent_delta_lower_threshold,
  ];

  const metricResults = METRIC_NAMES.map((name, m) => ({
    name,
    mean: means[m],
    lower: lowers[m],
    threshold: thresholds[m],
    passed: lowers[m] > thresholds[m],
    paired_scores: Array.from(metrics[m]),
  }));

  const directPointPassed = means[0] >= opts.minimum_direct_score;
  const allLowerPassed = metricResults.every((r) => r.passed);

  return {
    status: allLowerPassed && directPointPassed ? 'pass' : 'not-passed',
    pairs,
    bootstrap_samples: opts.bootstrap_samples,
    seed: opts.seed,
    family_alpha: opts.family_alpha,
    comparisons: opts.comparisons,
    rank,
    metrics: metricResults,
    minimum_direct_score: opts.minimum_direct_score,
    direct_point_passed: directPointPassed,
    interpretation: INTERPRETATION,
  };
}
