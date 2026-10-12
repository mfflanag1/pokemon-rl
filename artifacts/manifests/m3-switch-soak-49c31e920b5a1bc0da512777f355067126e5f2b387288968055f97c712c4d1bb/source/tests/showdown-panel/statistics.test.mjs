import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  summarizeMatrix,
  mulberry32,
  uniformIndex,
  percentileInterval95,
  MatrixValidationError,
} from '../../packages/showdown-panel/statistics.mjs';

// ───────────────────────────── helpers ─────────────────────────────

function edgeOrder(policyIds) {
  const edges = [];
  for (let i = 0; i < policyIds.length; i += 1) {
    for (let j = i + 1; j < policyIds.length; j += 1) edges.push([policyIds[i], policyIds[j]]);
  }
  return edges;
}

/** Build a complete grid; scoreOf(a, b, pair, mirror) gives a's outcome. */
function buildGrid(policyIds, pairs, scoreOf) {
  const records = [];
  for (const [a, b] of edgeOrder(policyIds)) {
    for (let pair = 0; pair < pairs; pair += 1) {
      for (const mirror of [0, 1]) {
        records.push({ a, b, pair, mirror, score: scoreOf(a, b, pair, mirror) });
      }
    }
  }
  return records;
}

function fromTable(table) {
  return (a, b, pair, mirror) => table[`${a}|${b}`][pair][mirror];
}

function deepFreeze(value) {
  if (value && typeof value === 'object') {
    Object.freeze(value);
    for (const key of Object.keys(value)) deepFreeze(value[key]);
  }
  return value;
}

function expectRejected(records, options, pattern) {
  assert.throws(
    () => summarizeMatrix(records, options),
    (error) => {
      assert.ok(error instanceof MatrixValidationError, `expected MatrixValidationError, got ${error && error.name}: ${error && error.message}`);
      assert.match(error.message, pattern);
      return true;
    },
  );
}

// ───────────────────────── exact accounting ─────────────────────────

test('exact 0 / 0.5 / 1 accounting, pair averaging, and edge order', () => {
  const policyIds = ['p', 'q', 'r'];
  const table = {
    'p|q': [[1, 0.5], [0, 0]],
    'p|r': [[1, 1], [1, 1]],
    'q|r': [[0.5, 0.5], [1, 0]],
  };
  // Reverse the record order to show the summary does not depend on it.
  const records = buildGrid(policyIds, 2, fromTable(table)).reverse();
  const result = summarizeMatrix(records, { policyIds, pairs: 2, bootstrapSamples: 200, seed: 1 });

  assert.equal(result.length, 3);
  assert.deepEqual(result.map((e) => [e.a, e.b]), [['p', 'q'], ['p', 'r'], ['q', 'r']]);

  const [pq, pr, qr] = result;

  assert.equal(pq.games, 4);
  assert.equal(pq.pairs, 2);
  assert.equal(pq.wins, 1);
  assert.equal(pq.draws, 1);
  assert.equal(pq.losses, 2);
  assert.equal(pq.mean_score, 0.375);
  assert.deepEqual(pq.pair_scores, [0.75, 0]);
  // Every replicate mean of two pair scores drawn from {0.75, 0} is one of these.
  assert.ok([0, 0.375, 0.75].includes(pq.interval95[0]));
  assert.ok([0, 0.375, 0.75].includes(pq.interval95[1]));
  assert.ok(pq.interval95[0] <= pq.interval95[1]);

  assert.equal(pr.wins, 4);
  assert.equal(pr.draws, 0);
  assert.equal(pr.losses, 0);
  assert.equal(pr.mean_score, 1);
  assert.deepEqual(pr.pair_scores, [1, 1]);
  assert.deepEqual(pr.interval95, [1, 1]);

  assert.equal(qr.wins, 1);
  assert.equal(qr.draws, 2);
  assert.equal(qr.losses, 1);
  assert.equal(qr.mean_score, 0.5);
  // pair 0: two draws, pair 1: a win and a loss — both pairs average to 0.5.
  assert.deepEqual(qr.pair_scores, [0.5, 0.5]);
  assert.deepEqual(qr.interval95, [0.5, 0.5]);

  for (const entry of result) {
    assert.deepEqual(Object.keys(entry).sort(), [
      'a', 'b', 'draws', 'games', 'interval95', 'losses', 'mean_score', 'pair_scores', 'pairs', 'wins',
    ]);
    assert.equal(entry.wins + entry.draws + entry.losses, entry.games);
    assert.equal(entry.pair_scores.length, entry.pairs);
    assert.equal(entry.interval95.length, 2);
  }
});

test('mirror=1 scores are still from a\'s point of view and the pair is the resampling unit', () => {
  // Policy p wins every mirror-0 game and loses every mirror-1 game.
  // Game-level: 6 wins, 6 losses. Pair-level: every pair is exactly 0.5, so a
  // whole-pair bootstrap cannot vary at all. Resampling individual games would
  // give a wide interval instead.
  const policyIds = ['p', 'q'];
  const records = buildGrid(policyIds, 6, (a, b, pair, mirror) => (mirror === 0 ? 1 : 0));
  const [pq] = summarizeMatrix(records, { policyIds, pairs: 6, bootstrapSamples: 3000, seed: 99 });
  assert.equal(pq.wins, 6);
  assert.equal(pq.losses, 6);
  assert.equal(pq.draws, 0);
  assert.equal(pq.mean_score, 0.5);
  assert.deepEqual(pq.pair_scores, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  assert.deepEqual(pq.interval95, [0.5, 0.5]);
});

test('a single pair always yields a degenerate interval equal to its pair score', () => {
  const policyIds = ['p', 'q', 'r'];
  const table = { 'p|q': [[1, 0.5]], 'p|r': [[0, 0]], 'q|r': [[0.5, 1]] };
  const result = summarizeMatrix(buildGrid(policyIds, 1, fromTable(table)), {
    policyIds, pairs: 1, bootstrapSamples: 500, seed: 5,
  });
  assert.deepEqual(result.map((e) => e.interval95), [[0.75, 0.75], [0, 0], [0.75, 0.75]]);
  assert.deepEqual(result.map((e) => e.mean_score), [0.75, 0, 0.75]);
});

test('two pairs at 0 and 1 give the full [0, 1] interval at the default sample count', () => {
  // Replicate means are 0, 0.5 or 1 with probabilities 1/4, 1/2, 1/4. With
  // 10000 replicates and k = 250, the 250th smallest is 0 and the 250th
  // largest is 1 unless the PRNG is badly broken (expected 2500 of each).
  const policyIds = ['p', 'q'];
  const records = buildGrid(policyIds, 2, (a, b, pair) => (pair === 0 ? 0 : 1));
  const [pq] = summarizeMatrix(records, { policyIds, pairs: 2 });
  assert.deepEqual(pq.pair_scores, [0, 1]);
  assert.deepEqual(pq.interval95, [0, 1]);
});

// ───────────────────── shared indices across edges ─────────────────────

test('all edges use the same resampled pair indices in every replicate', () => {
  const policyIds = ['a', 'b', 'c'];
  const table = {
    'a|b': [[1, 1], [0, 0.5], [0.5, 1], [0, 0]], // pair scores 1, .25, .75, 0
    'a|c': [[0, 0], [1, 0.5], [0.5, 0], [1, 1]], // exact complement: 0, .75, .25, 1
    'b|c': [[1, 1], [0, 0.5], [0.5, 1], [0, 0]], // identical to a|b
  };
  const records = buildGrid(policyIds, 4, fromTable(table));
  const [ab, ac, bc] = summarizeMatrix(records, { policyIds, pairs: 4, bootstrapSamples: 2000, seed: 4242 });

  assert.deepEqual(ab.pair_scores, [1, 0.25, 0.75, 0]);
  assert.deepEqual(ac.pair_scores, [0, 0.75, 0.25, 1]);
  assert.ok(ab.interval95[0] < ab.interval95[1], 'interval should be non-degenerate here');

  // Identical pair scores + identical indices ⇒ identical replicates.
  assert.deepEqual(bc.interval95, ab.interval95);

  // Complemented pair scores + identical indices + symmetric quantile rule
  // ⇒ exactly mirrored interval. Independent index streams would not give
  // this exactly.
  const tolerance = 1e-12;
  assert.ok(Math.abs(ac.interval95[0] - (1 - ab.interval95[1])) < tolerance, `${ac.interval95} vs ${ab.interval95}`);
  assert.ok(Math.abs(ac.interval95[1] - (1 - ab.interval95[0])) < tolerance, `${ac.interval95} vs ${ab.interval95}`);
});

// ─────────────────────── independently derived case ───────────────────────

/**
 * Independent re-derivation of the documented procedure, written without
 * reference to the implementation: groups raw records itself, uses Ettinger's
 * published float-state mulberry32 form, generic rejection sampling, and
 * counting-based order statistics instead of sort-and-index.
 */
function oracleSummary(records, { policyIds, pairs, bootstrapSamples, seed }) {
  const edges = edgeOrder(policyIds);
  const pairScores = edges.map(([a, b]) => {
    const scores = [];
    for (let p = 0; p < pairs; p += 1) {
      const games = records.filter((r) => r.a === a && r.b === b && r.pair === p);
      assert.equal(games.length, 2);
      scores.push(games.reduce((acc, r) => acc + r.score, 0) / 2);
    }
    return scores;
  });

  let state = seed;
  const next = () => {
    let t = (state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
  const limit = 2 ** 32 - ((2 ** 32) % pairs);
  const draw = () => {
    for (;;) {
      const u = next();
      if (u < limit) return u % pairs;
    }
  };

  const replicates = edges.map(() => []);
  for (let s = 0; s < bootstrapSamples; s += 1) {
    const indices = Array.from({ length: pairs }, () => draw());
    edges.forEach((_, e) => {
      let sum = 0;
      for (const i of indices) sum += pairScores[e][i];
      replicates[e].push(sum / pairs);
    });
  }

  const k = Math.ceil(bootstrapSamples / 40);
  const kthSmallest = (values) => {
    const distinct = [...new Set(values)].sort((x, y) => x - y);
    return distinct.find((v) => values.filter((x) => x <= v).length >= k);
  };
  const kthLargest = (values) => {
    const distinct = [...new Set(values)].sort((x, y) => y - x);
    return distinct.find((v) => values.filter((x) => x >= v).length >= k);
  };

  return edges.map(([a, b], e) => ({
    a,
    b,
    pair_scores: pairScores[e],
    interval95: [kthSmallest(replicates[e]), kthLargest(replicates[e])],
  }));
}

test('nontrivial bootstrap matches an independent re-derivation of the documented spec', () => {
  const policyIds = ['x', 'y', 'z'];
  const table = {
    'x|y': [[1, 0], [0.5, 0.5], [1, 1], [0, 0.5], [0, 0]], // .5 .5 1 .25 0
    'x|z': [[0, 0], [1, 0.5], [0.5, 0], [1, 1], [0.5, 1]], // 0 .75 .25 1 .75
    'y|z': [[1, 1], [1, 0.5], [0, 0], [0.5, 0], [1, 0]], // 1 .75 0 .25 .5
  };
  const options = { policyIds, pairs: 5, bootstrapSamples: 300, seed: 12345 };
  const records = buildGrid(policyIds, 5, fromTable(table));

  const expected = oracleSummary(records, options);
  const actual = summarizeMatrix(records, options);

  assert.deepEqual(expected.map((e) => e.pair_scores), [
    [0.5, 0.5, 1, 0.25, 0],
    [0, 0.75, 0.25, 1, 0.75],
    [1, 0.75, 0, 0.25, 0.5],
  ]);
  assert.deepEqual(actual.map((e) => e.pair_scores), expected.map((e) => e.pair_scores));

  for (let e = 0; e < expected.length; e += 1) {
    assert.equal(actual[e].a, expected[e].a);
    assert.equal(actual[e].b, expected[e].b);
    for (const side of [0, 1]) {
      assert.ok(
        Math.abs(actual[e].interval95[side] - expected[e].interval95[side]) < 1e-12,
        `edge ${expected[e].a}|${expected[e].b}: ${actual[e].interval95} vs oracle ${expected[e].interval95}`,
      );
    }
    assert.ok(actual[e].interval95[0] < actual[e].interval95[1], 'case must be non-degenerate to be meaningful');
    const lo = Math.min(...actual[e].pair_scores);
    const hi = Math.max(...actual[e].pair_scores);
    assert.ok(actual[e].interval95[0] >= lo && actual[e].interval95[1] <= hi);
  }
});

// ───────────────────── literal known-answer fixture ─────────────────────

/**
 * Frozen literals computed OUTSIDE this codebase (coordinator-supplied,
 * unsigned 32-bit integer arithmetic for mulberry32, seed 0). They are not
 * derived from `mulberry32`, `uniformIndex`, `oracleSummary`, or any other
 * bootstrap implementation, so this test catches a regression that would
 * change the runtime oracle and the implementation in lockstep.
 *
 * Draw order per the module header: replicate s = 0..B−1, pair p = 0..pairs−1,
 * one uniform index per (s, p) from a single PRNG stream. With pairs = 3 the
 * rejection limit is 2^32 − (2^32 mod 3) = 2^32 − 1, so every word below is
 * accepted and index = word mod 3.
 */
const MULBERRY32_SEED0_WORDS = [
  1144304738, 1416247, 958946056, 627933444, 2007157716,
  2340967985, 2642484575, 2787370982, 1958536065, 2496316458,
  1057668038, 420269829, 3880206403, 1753221639, 3093645806,
];
const SEED0_INDICES_MOD3 = [2, 1, 1, 0, 0, 2, 2, 2, 0, 0, 2, 0, 1, 0, 2];

test('mulberry32 seed 0 reproduces the frozen first 15 uint32 words', () => {
  const next = mulberry32(0);
  const words = Array.from({ length: 15 }, () => next());
  assert.deepEqual(words, MULBERRY32_SEED0_WORDS);
});

test('uniformIndex over the seed 0 stream reproduces the frozen indices for n = 3', () => {
  const next = mulberry32(0);
  const indices = Array.from({ length: 15 }, () => uniformIndex(next, 3));
  assert.deepEqual(indices, SEED0_INDICES_MOD3);
  // Every one of the 15 words is below the limit 2^32 − 1, so exactly 15
  // words were consumed (no rejections).
  assert.deepEqual(MULBERRY32_SEED0_WORDS.map((w) => w < 2 ** 32 - 1), new Array(15).fill(true));
  assert.deepEqual(MULBERRY32_SEED0_WORDS.map((w) => w % 3), SEED0_INDICES_MOD3);
});

test('bootstrap known answer: pair scores [0, 0.5, 1], B = 5, seed 0 ⇒ interval [1/3, 2/3]', () => {
  // Records carry both mirrors per pair. Pair 1 is split 1/0 across its
  // mirrors, so its pair score is 0.5 without any draw.
  const policyIds = ['p', 'q'];
  const records = [
    { a: 'p', b: 'q', pair: 0, mirror: 0, score: 0 },
    { a: 'p', b: 'q', pair: 0, mirror: 1, score: 0 },
    { a: 'p', b: 'q', pair: 1, mirror: 0, score: 1 },
    { a: 'p', b: 'q', pair: 1, mirror: 1, score: 0 },
    { a: 'p', b: 'q', pair: 2, mirror: 0, score: 1 },
    { a: 'p', b: 'q', pair: 2, mirror: 1, score: 1 },
  ];
  const [pq] = summarizeMatrix(records, { policyIds, pairs: 3, bootstrapSamples: 5, seed: 0 });

  assert.equal(pq.wins, 3);
  assert.equal(pq.draws, 0);
  assert.equal(pq.losses, 3);
  assert.equal(pq.mean_score, 0.5);
  assert.deepEqual(pq.pair_scores, [0, 0.5, 1]);

  // Frozen indices grouped by replicate (3 per replicate, 5 replicates):
  //   [2,1,1] → (1 + .5 + .5)/3 = 2/3
  //   [0,0,2] → (0 + 0 + 1)/3   = 1/3
  //   [2,2,0] → (1 + 1 + 0)/3   = 2/3
  //   [0,2,0] → (0 + 1 + 0)/3   = 1/3
  //   [1,0,2] → (.5 + 0 + 1)/3  = 1/2
  // B = 5 ⇒ k = ⌈5/40⌉ = 1 ⇒ [smallest, largest] = [1/3, 2/3].
  assert.deepEqual(pq.interval95, [1 / 3, 2 / 3]);
  assert.ok(pq.interval95[0] < pq.interval95[1], 'fixture is non-degenerate');
});

// ───────────────────────────── determinism ─────────────────────────────

test('summaries are deterministic for a given seed and return fresh objects', () => {
  const policyIds = ['p', 'q', 'r'];
  const records = buildGrid(policyIds, 7, (a, b, pair, mirror) => [0, 0.5, 1][(pair * 2 + mirror + a.length) % 3]);
  const options = { policyIds, pairs: 7, bootstrapSamples: 400, seed: 2024 };
  const first = summarizeMatrix(records, options);
  const second = summarizeMatrix(records, options);
  assert.deepEqual(first, second);
  assert.notEqual(first, second);
  assert.notEqual(first[0].pair_scores, second[0].pair_scores);
  assert.notEqual(first[0].interval95, second[0].interval95);

  const defaults = summarizeMatrix(records, { policyIds, pairs: 7 });
  const explicit = summarizeMatrix(records, { policyIds, pairs: 7, bootstrapSamples: 10000, seed: 73291 });
  assert.deepEqual(defaults, explicit);
});

test('mulberry32 is reproducible, uint32-valued, and seed-sensitive', () => {
  const take = (seed, n) => {
    const next = mulberry32(seed);
    return Array.from({ length: n }, () => next());
  };
  const a = take(7, 100);
  const b = take(7, 100);
  assert.deepEqual(a, b);
  for (const v of a) {
    assert.ok(Number.isInteger(v) && v >= 0 && v <= 0xffffffff, `not uint32: ${v}`);
  }
  assert.notDeepEqual(take(8, 100), a);
  assert.ok(new Set(a).size > 90, 'stream should not be stuck');
  assert.throws(() => mulberry32(-1), MatrixValidationError);
  assert.throws(() => mulberry32(2 ** 32), MatrixValidationError);
  assert.throws(() => mulberry32(1.5), MatrixValidationError);
});

test('uniformIndex rejects values at or above the unbiased limit', () => {
  // 2^32 mod 3 = 1, so the only rejected value for n = 3 is 2^32 - 1.
  const feed = [0xffffffff, 7];
  let consumed = 0;
  const next = () => feed[consumed++];
  assert.equal(uniformIndex(next, 3), 1); // 7 mod 3
  assert.equal(consumed, 2);

  // n = 1 always yields 0 but still consumes a draw.
  let calls = 0;
  assert.equal(uniformIndex(() => { calls += 1; return 123456789; }, 1), 0);
  assert.equal(calls, 1);

  const prng = mulberry32(3);
  for (let i = 0; i < 5000; i += 1) {
    const v = uniformIndex(prng, 5);
    assert.ok(Number.isInteger(v) && v >= 0 && v < 5);
  }
  assert.throws(() => uniformIndex(prng, 0), MatrixValidationError);
});

test('percentileInterval95 uses k = ceil(B/40): k-th smallest and k-th largest', () => {
  assert.deepEqual(percentileInterval95([5, 1, 4, 2, 3]), [1, 5]); // B=5, k=1
  const descending = (n) => Array.from({ length: n }, (_, i) => n - 1 - i);

  const forty = descending(40); // k=1
  assert.deepEqual(percentileInterval95(forty), [0, 39]);
  assert.deepEqual(forty, descending(40), 'input must not be mutated');

  assert.deepEqual(percentileInterval95(descending(41)), [1, 39]); // k=2
  assert.deepEqual(percentileInterval95(descending(80)), [1, 78]); // k=2
  assert.deepEqual(percentileInterval95(descending(81)), [2, 78]); // k=3
  assert.deepEqual(percentileInterval95(descending(10000)), [249, 9750]); // k=250
  assert.deepEqual(percentileInterval95(Float64Array.from([0.5, 0.25, 0.75])), [0.25, 0.75]);
  assert.deepEqual(percentileInterval95([2, 2, 2]), [2, 2]);

  assert.throws(() => percentileInterval95([]), MatrixValidationError);
  assert.throws(() => percentileInterval95([1, NaN]), MatrixValidationError);
  assert.throws(() => percentileInterval95('abc'), MatrixValidationError);
});

// ───────────────────────── orientation handling ─────────────────────────

test('records given in the reverse orientation are normalised with complemented scores', () => {
  const policyIds = ['p', 'q'];
  const table = { 'p|q': [[1, 0.5], [0, 0], [0.5, 1]] };
  const canonical = buildGrid(policyIds, 3, fromTable(table));
  const reversed = canonical.map((r) => ({ a: r.b, b: r.a, pair: r.pair, mirror: r.mirror, score: 1 - r.score }));
  const mixed = canonical.map((r, i) => (i % 2 === 0 ? r : reversed[i]));
  const options = { policyIds, pairs: 3, bootstrapSamples: 100, seed: 11 };

  const expected = summarizeMatrix(canonical, options);
  assert.deepEqual(summarizeMatrix(reversed, options), expected);
  assert.deepEqual(summarizeMatrix(mixed, options), expected);
  assert.equal(expected[0].a, 'p');
  assert.equal(expected[0].b, 'q');
  assert.equal(expected[0].wins, 2);
  assert.equal(expected[0].draws, 2);
  assert.equal(expected[0].losses, 2);
});

// ───────────────────────────── malformed grids ─────────────────────────────

test('malformed grids are rejected', () => {
  const policyIds = ['p', 'q', 'r'];
  const options = { policyIds, pairs: 2, bootstrapSamples: 50, seed: 3 };
  const good = () => buildGrid(policyIds, 2, (a, b, pair, mirror) => [1, 0.5, 0][(pair + mirror) % 3]);
  assert.doesNotThrow(() => summarizeMatrix(good(), options));

  const withOne = (patch) => {
    const records = good();
    records[3] = { ...records[3], ...patch };
    return records;
  };

  // Missing / extra / duplicate.
  expectRejected(good().slice(1), options, /expected exactly 12 records/);
  expectRejected([...good(), good()[0]], options, /expected exactly 12 records/);
  const dup = good();
  dup[5] = { ...dup[0] };
  expectRejected(dup, options, /duplicates game/);
  const dupReversed = good();
  dupReversed[5] = { a: dup[0].b, b: dup[0].a, pair: dup[0].pair, mirror: dup[0].mirror, score: 1 - dup[0].score };
  expectRejected(dupReversed, options, /duplicates game/);
  expectRejected([], options, /expected exactly 12 records/);

  // Policies.
  expectRejected(withOne({ a: 'zzz' }), options, /not a known policy id/);
  expectRejected(withOne({ b: 'zzz' }), options, /not a known policy id/);
  expectRejected(withOne({ a: 42 }), options, /not a known policy id/);
  expectRejected(withOne({ b: undefined }), options, /not a known policy id/);
  expectRejected(withOne({ a: 'p', b: 'p' }), options, /against itself/);

  // Pair index.
  expectRejected(withOne({ pair: 2 }), options, /pair must be an integer in \[0, 2\)/);
  expectRejected(withOne({ pair: -1 }), options, /pair must be an integer/);
  expectRejected(withOne({ pair: 0.5 }), options, /pair must be an integer/);
  expectRejected(withOne({ pair: '0' }), options, /pair must be an integer/);
  expectRejected(withOne({ pair: NaN }), options, /pair must be an integer/);

  // Mirror / seat.
  expectRejected(withOne({ mirror: 2 }), options, /mirror must be 0 or 1/);
  expectRejected(withOne({ mirror: '1' }), options, /mirror must be 0 or 1/);
  expectRejected(withOne({ mirror: true }), options, /mirror must be 0 or 1/);
  expectRejected(withOne({ mirror: undefined }), options, /mirror must be 0 or 1/);

  // Score.
  expectRejected(withOne({ score: NaN }), options, /score must be 0, 0.5 or 1/);
  expectRejected(withOne({ score: Infinity }), options, /score must be 0, 0.5 or 1/);
  expectRejected(withOne({ score: -Infinity }), options, /score must be 0, 0.5 or 1/);
  expectRejected(withOne({ score: 0.3 }), options, /score must be 0, 0.5 or 1/);
  expectRejected(withOne({ score: -0.5 }), options, /score must be 0, 0.5 or 1/);
  expectRejected(withOne({ score: 2 }), options, /score must be 0, 0.5 or 1/);
  expectRejected(withOne({ score: '1' }), options, /score must be 0, 0.5 or 1/);
  expectRejected(withOne({ score: undefined }), options, /score must be 0, 0.5 or 1/);

  // Record shape / container.
  const nullRecord = good();
  nullRecord[3] = null;
  expectRejected(nullRecord, options, /record\[3\] must be an object/);
  const arrayRecord = good();
  arrayRecord[3] = ['p', 'q', 0, 0, 1];
  expectRejected(arrayRecord, options, /record\[3\] must be an object/);
  expectRejected('not an array', options, /records must be an array/);
  expectRejected(undefined, options, /records must be an array/);
});

test('invalid options are rejected', () => {
  const policyIds = ['p', 'q'];
  const records = buildGrid(policyIds, 2, () => 0.5);
  const base = { policyIds, pairs: 2, bootstrapSamples: 20, seed: 9 };
  assert.doesNotThrow(() => summarizeMatrix(records, base));

  expectRejected(records, undefined, /options must be an object/);
  expectRejected(records, null, /options must be an object/);

  expectRejected(records, { ...base, policyIds: undefined }, /policyIds must be an array/);
  expectRejected(records, { ...base, policyIds: 'pq' }, /policyIds must be an array/);
  expectRejected(records, { ...base, policyIds: ['p'] }, /at least two/);
  expectRejected(records, { ...base, policyIds: ['p', 'p'] }, /duplicate policy id/);
  expectRejected(records, { ...base, policyIds: ['p', 1] }, /non-empty strings/);
  expectRejected(records, { ...base, policyIds: ['p', ''] }, /non-empty strings/);

  expectRejected(records, { ...base, pairs: 0 }, /pairs must be a positive integer/);
  expectRejected(records, { ...base, pairs: -2 }, /pairs must be a positive integer/);
  expectRejected(records, { ...base, pairs: 2.5 }, /pairs must be a positive integer/);
  expectRejected(records, { ...base, pairs: '2' }, /pairs must be a positive integer/);
  expectRejected(records, { ...base, pairs: undefined }, /pairs must be a positive integer/);

  expectRejected(records, { ...base, bootstrapSamples: 0 }, /bootstrapSamples must be a positive integer/);
  expectRejected(records, { ...base, bootstrapSamples: -1 }, /bootstrapSamples must be a positive integer/);
  expectRejected(records, { ...base, bootstrapSamples: 1.5 }, /bootstrapSamples must be a positive integer/);
  expectRejected(records, { ...base, bootstrapSamples: null }, /bootstrapSamples must be a positive integer/);

  expectRejected(records, { ...base, seed: -1 }, /seed must be an integer/);
  expectRejected(records, { ...base, seed: 2 ** 32 }, /seed must be an integer/);
  expectRejected(records, { ...base, seed: 1.5 }, /seed must be an integer/);
  expectRejected(records, { ...base, seed: NaN }, /seed must be an integer/);
  expectRejected(records, { ...base, seed: '1' }, /seed must be an integer/);

  // Boundary seeds are accepted.
  assert.doesNotThrow(() => summarizeMatrix(records, { ...base, seed: 0 }));
  assert.doesNotThrow(() => summarizeMatrix(records, { ...base, seed: 0xffffffff }));
});

// ───────────────────────────── input preservation ─────────────────────────────

test('inputs are not mutated', () => {
  const policyIds = ['p', 'q', 'r'];
  const records = buildGrid(policyIds, 3, (a, b, pair, mirror) => [0, 1, 0.5][(pair + 2 * mirror) % 3]).reverse();
  const snapshot = JSON.stringify(records);
  const options = { policyIds, pairs: 3, bootstrapSamples: 100, seed: 17 };
  const optionsSnapshot = JSON.stringify(options);

  deepFreeze(records);
  deepFreeze(options);
  const result = summarizeMatrix(records, options);
  assert.equal(result.length, 3);

  assert.equal(JSON.stringify(records), snapshot);
  assert.equal(JSON.stringify(options), optionsSnapshot);
  assert.deepEqual(policyIds, ['p', 'q', 'r']);
  for (const entry of result) assert.notEqual(entry.pair_scores, records);
});
