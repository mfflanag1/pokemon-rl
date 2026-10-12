/**
 * Showdown opponent-panel statistics.
 *
 * Pure, dependency-free summary of a round-robin policy matrix that was played
 * as mirrored pairs: for every unordered policy matchup ("edge") and every
 * team/seed pair index, two games are played with the policy seats swapped
 * (`mirror` 0 and 1). This module turns that grid of per-game outcomes into
 * per-edge win/draw/loss accounting, per-pair scores, and a descriptive
 * paired-bootstrap interval of the mean score.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Input contract
 * ──────────────────────────────────────────────────────────────────────────
 * `summarizeMatrix(records, { policyIds, pairs, bootstrapSamples, seed })`
 *
 *   records           Array of `{ a, b, pair, mirror, score }` objects.
 *                       a, b    policy ids, both present in `policyIds`, a ≠ b.
 *                       pair    integer in [0, pairs).
 *                       mirror  0 or 1 (which seat assignment of the pair).
 *                       score   outcome for policy `a`: 0 (loss), 0.5 (draw),
 *                               or 1 (win). It is ALWAYS from `a`'s point of
 *                               view, also when mirror = 1.
 *                     Extra properties are ignored. Records may appear in any
 *                     order. A record may name the matchup in either
 *                     orientation; if `a` comes after `b` in `policyIds` the
 *                     record is normalised to the canonical orientation and its
 *                     score is complemented (1 − score), which is lossless
 *                     because the score is defined as `a`'s outcome.
 *   policyIds         ≥ 2 unique non-empty strings. Their order defines the
 *                     canonical orientation of every edge and the output order.
 *   pairs             positive integer: number of team/seed pairs per edge.
 *   bootstrapSamples  positive integer (default 10000).
 *   seed              integer in [0, 2^32 − 1] (default 73291).
 *
 * The grid must be complete and exact: for every edge (i < j in `policyIds`
 * index order), every pair 0..pairs−1 and both mirrors there must be exactly
 * one record. Missing, duplicate (including the same game given in the other
 * orientation), foreign-policy, self-play, out-of-range, non-finite, or
 * otherwise invalid records cause a `MatrixValidationError`. So do invalid
 * options. Inputs are never mutated.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Output
 * ──────────────────────────────────────────────────────────────────────────
 * An array with one entry per edge in lexicographic (i, j) index order, i.e.
 * (0,1), (0,2), …, (0,n−1), (1,2), …, (n−2,n−1):
 *
 *   {
 *     a, b,            canonical policy ids (a precedes b in policyIds)
 *     games,           2 × pairs
 *     pairs,           number of pairs
 *     wins, draws, losses,   game counts from a's point of view
 *     mean_score,      (wins + 0.5·draws) / games  (= mean of pair_scores)
 *     pair_scores,     Array(pairs); pair_scores[p] = (score(p,0)+score(p,1))/2
 *     interval95: [low, high]   descriptive 95 % bootstrap interval, see below
 *   }
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Bootstrap specification (deterministic)
 * ──────────────────────────────────────────────────────────────────────────
 * The resampling unit is the whole pair, never the individual game: the two
 * mirrored games of a pair share teams and mechanics seed and are therefore
 * not independent observations. Each bootstrap replicate resamples `pairs`
 * pair indices with replacement and averages the pair scores at those indices.
 *
 * The SAME resampled pair indices are applied to every edge within a
 * replicate, so cross-edge dependence that stems from reusing the same pairs
 * across matchups is preserved (differences between edges are comparable
 * replicate-by-replicate).
 *
 * PRNG: mulberry32 (Tommy Ettinger's 32-bit generator), stated explicitly in
 * `mulberry32` below, with its state kept as a uint32. Index draws use
 * unbiased rejection sampling (`uniformIndex`). Draw order: for replicate
 * s = 0..B−1, for p = 0..pairs−1, draw index[p] from the single PRNG stream.
 *
 * Quantiles: with B replicates sorted ascending and k = ⌈B/40⌉ (= ⌈0.025·B⌉
 * computed in integer arithmetic), `low` is the k-th smallest replicate and
 * `high` is the k-th largest replicate. This is the symmetric nearest-rank
 * rule for the 2.5 % and 97.5 % percentiles; it differs from the plain
 * nearest-rank 97.5 % rank ⌈0.975·B⌉ by at most one position and has the
 * property that complementing every score exactly mirrors the interval.
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Interpretation — read before quoting an interval
 * ──────────────────────────────────────────────────────────────────────────
 * • The intervals are DESCRIPTIVE. They summarise how much the observed mean
 *   score would move under resampling of the admitted pairs. They are not a
 *   test, carry no promotion / qualification decision, and make no claim about
 *   other team pools, seeds, formats, or servers.
 * • They are PER-EDGE intervals. No simultaneous-coverage or
 *   multiple-comparison adjustment is applied; quoting several edges together
 *   does not give joint 95 % coverage.
 * • DEGENERATE intervals (low === high) arise whenever all pair scores of an
 *   edge coincide — for example `pairs = 1`, every pair split 1/0 across its
 *   mirrors, or a sweep — and when B is tiny. A zero-width interval means the
 *   resampled statistic cannot vary, not that the true score is known
 *   exactly.
 * • The bootstrap does not model game-level randomness inside a fixed pair
 *   (both mirrors were played exactly once), nor the finite number of
 *   admitted teams.
 */

const UINT32_RANGE = 2 ** 32;
const VALID_SCORES = [0, 0.5, 1];

export class MatrixValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'MatrixValidationError';
  }
}

/**
 * mulberry32 — explicit, dependency-free uint32 PRNG.
 *
 * State: one uint32. Per call:
 *   state = (state + 0x6D2B79F5) mod 2^32
 *   t = state
 *   t = imul(t ^ (t >>> 15), t | 1)
 *   t = t ^ (t + imul(t ^ (t >>> 7), t | 61))
 *   return (t ^ (t >>> 14)) as uint32
 * where imul is 32-bit integer multiplication (Math.imul) and bitwise ops are
 * the usual JavaScript int32 operations. Returns integers in [0, 2^32 − 1].
 */
export function mulberry32(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new MatrixValidationError(`seed must be an integer in [0, 2^32 - 1]; got ${String(seed)}`);
  }
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}

/**
 * Unbiased integer in [0, n) from a uint32 source via rejection sampling:
 * draw u; accept iff u < 2^32 − (2^32 mod n); return u mod n.
 */
export function uniformIndex(next, n) {
  if (!Number.isInteger(n) || n < 1 || n > UINT32_RANGE) {
    throw new MatrixValidationError(`uniformIndex: n must be an integer in [1, 2^32]; got ${String(n)}`);
  }
  const limit = UINT32_RANGE - (UINT32_RANGE % n);
  let u;
  do {
    u = next();
  } while (u >= limit);
  return u % n;
}

/**
 * Symmetric nearest-rank 95 % interval of a sample of finite numbers.
 * With B values and k = ⌈B/40⌉: [k-th smallest, k-th largest].
 * The input is not mutated.
 */
export function percentileInterval95(values) {
  if (!Array.isArray(values) && !ArrayBuffer.isView(values)) {
    throw new MatrixValidationError('percentileInterval95: values must be an array');
  }
  const size = values.length;
  if (size < 1) {
    throw new MatrixValidationError('percentileInterval95: need at least one value');
  }
  const sorted = Array.from(values);
  for (const v of sorted) {
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new MatrixValidationError('percentileInterval95: values must be finite numbers');
    }
  }
  sorted.sort((x, y) => x - y);
  const k = Math.floor((size + 39) / 40); // ⌈size / 40⌉ without floating-point
  return [sorted[k - 1], sorted[size - k]];
}

function isPositiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function describe(value) {
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number' || typeof value === 'boolean' || value == null) return String(value);
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function validateOptions(options) {
  if (options === null || typeof options !== 'object') {
    throw new MatrixValidationError('options must be an object with policyIds and pairs');
  }
  const { policyIds, pairs, bootstrapSamples = 10000, seed = 73291 } = options;

  if (!Array.isArray(policyIds) || policyIds.length < 2) {
    throw new MatrixValidationError('policyIds must be an array of at least two policy ids');
  }
  const seen = new Set();
  for (const id of policyIds) {
    if (typeof id !== 'string' || id.length === 0) {
      throw new MatrixValidationError(`policyIds entries must be non-empty strings; got ${describe(id)}`);
    }
    if (seen.has(id)) {
      throw new MatrixValidationError(`policyIds contains duplicate policy id ${describe(id)}`);
    }
    seen.add(id);
  }
  if (!isPositiveInteger(pairs)) {
    throw new MatrixValidationError(`pairs must be a positive integer; got ${describe(pairs)}`);
  }
  if (!isPositiveInteger(bootstrapSamples)) {
    throw new MatrixValidationError(`bootstrapSamples must be a positive integer; got ${describe(bootstrapSamples)}`);
  }
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new MatrixValidationError(`seed must be an integer in [0, 2^32 - 1]; got ${describe(seed)}`);
  }
  return { policyIds: policyIds.slice(), pairs, bootstrapSamples, seed };
}

/**
 * Summarise a complete mirrored-pair round-robin grid. See the module header
 * for the full contract.
 */
export function summarizeMatrix(records, options) {
  const { policyIds, pairs, bootstrapSamples, seed } = validateOptions(options);

  if (!Array.isArray(records)) {
    throw new MatrixValidationError('records must be an array');
  }

  const indexOf = new Map(policyIds.map((id, i) => [id, i]));
  const count = policyIds.length;

  // Edges in lexicographic (i, j) index order.
  const edges = [];
  const edgeByKey = new Map();
  for (let i = 0; i < count; i += 1) {
    for (let j = i + 1; j < count; j += 1) {
      const cells = new Array(pairs);
      for (let p = 0; p < pairs; p += 1) cells[p] = [null, null];
      const edge = { a: policyIds[i], b: policyIds[j], ai: i, bi: j, cells };
      edges.push(edge);
      edgeByKey.set(`${i}:${j}`, edge);
    }
  }

  const expected = edges.length * pairs * 2;
  if (records.length !== expected) {
    throw new MatrixValidationError(
      `expected exactly ${expected} records (${edges.length} edges × ${pairs} pairs × 2 mirrors); got ${records.length}`,
    );
  }

  records.forEach((record, position) => {
    if (record === null || typeof record !== 'object' || Array.isArray(record)) {
      throw new MatrixValidationError(`record[${position}] must be an object; got ${describe(record)}`);
    }
    const { a, b, pair, mirror, score } = record;

    if (typeof a !== 'string' || !indexOf.has(a)) {
      throw new MatrixValidationError(`record[${position}].a ${describe(a)} is not a known policy id`);
    }
    if (typeof b !== 'string' || !indexOf.has(b)) {
      throw new MatrixValidationError(`record[${position}].b ${describe(b)} is not a known policy id`);
    }
    if (a === b) {
      throw new MatrixValidationError(`record[${position}] pairs policy ${describe(a)} against itself`);
    }
    if (!Number.isInteger(pair) || pair < 0 || pair >= pairs) {
      throw new MatrixValidationError(`record[${position}].pair must be an integer in [0, ${pairs}); got ${describe(pair)}`);
    }
    if (mirror !== 0 && mirror !== 1) {
      throw new MatrixValidationError(`record[${position}].mirror must be 0 or 1; got ${describe(mirror)}`);
    }
    if (typeof score !== 'number' || !Number.isFinite(score) || !VALID_SCORES.includes(score)) {
      throw new MatrixValidationError(`record[${position}].score must be 0, 0.5 or 1; got ${describe(score)}`);
    }

    const ai = indexOf.get(a);
    const bi = indexOf.get(b);
    const canonical = ai < bi;
    const edge = edgeByKey.get(canonical ? `${ai}:${bi}` : `${bi}:${ai}`);
    const canonicalScore = canonical ? score : 1 - score;

    if (edge.cells[pair][mirror] !== null) {
      throw new MatrixValidationError(
        `record[${position}] duplicates game ${describe(edge.a)} vs ${describe(edge.b)} pair ${pair} mirror ${mirror}`,
      );
    }
    edge.cells[pair][mirror] = canonicalScore;
  });

  // The count check plus duplicate rejection already implies completeness,
  // but verify explicitly so the invariant does not depend on that reasoning.
  for (const edge of edges) {
    for (let p = 0; p < pairs; p += 1) {
      for (let m = 0; m < 2; m += 1) {
        if (edge.cells[p][m] === null) {
          throw new MatrixValidationError(
            `missing game ${describe(edge.a)} vs ${describe(edge.b)} pair ${p} mirror ${m}`,
          );
        }
      }
    }
  }

  // Per-edge accounting.
  const summaries = edges.map((edge) => {
    let wins = 0;
    let draws = 0;
    let losses = 0;
    const pairScores = new Array(pairs);
    for (let p = 0; p < pairs; p += 1) {
      const [s0, s1] = edge.cells[p];
      for (const s of [s0, s1]) {
        if (s === 1) wins += 1;
        else if (s === 0.5) draws += 1;
        else losses += 1;
      }
      pairScores[p] = (s0 + s1) / 2;
    }
    const games = pairs * 2;
    return {
      a: edge.a,
      b: edge.b,
      games,
      pairs,
      wins,
      draws,
      losses,
      mean_score: (wins + 0.5 * draws) / games,
      pair_scores: pairScores,
      interval95: null,
    };
  });

  // Paired bootstrap with shared indices across edges.
  const next = mulberry32(seed);
  const replicates = summaries.map(() => new Float64Array(bootstrapSamples));
  const indices = new Array(pairs);
  for (let s = 0; s < bootstrapSamples; s += 1) {
    for (let p = 0; p < pairs; p += 1) {
      indices[p] = uniformIndex(next, pairs);
    }
    for (let e = 0; e < summaries.length; e += 1) {
      const scores = summaries[e].pair_scores;
      let sum = 0;
      for (let p = 0; p < pairs; p += 1) sum += scores[indices[p]];
      replicates[e][s] = sum / pairs;
    }
  }
  for (let e = 0; e < summaries.length; e += 1) {
    summaries[e].interval95 = percentileInterval95(replicates[e]);
  }

  return summaries;
}
