import test from 'node:test';
import assert from 'node:assert/strict';
import readline from 'node:readline';
import {readFileSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {FullTrainingSession, officialPlan, SPLITS, MAX_TEAM_ATTEMPTS} from '../../packages/full-training/session.mjs';
import {teamArtifact} from '../../packages/reference-node/contract-v1.mjs';
import {partition, teamKey} from '../../packages/dataset/identity.mjs';

const SIDES = ['p1', 'p2'];
const RUN_SEED = 7781;
const ZERO = '0'.repeat(64);
const WORKER = fileURLToPath(new URL('../../packages/full-training/worker.mjs', import.meta.url));

const keysOf = plan => plan.teams.map(t => t.team_key);
const speciesOf = (frame, side) => frame.players[side].own.map(p => p.species);

function liveActions(frame) {
  const actions = {}, request_ids = {};
  for (const side of SIDES) {
    const player = frame.players[side];
    const index = player.mask.findIndex(Boolean);
    if (index >= 0) { actions[side] = index; request_ids[side] = player.request_id; }
  }
  if (!Object.keys(actions).length) throw new Error('No live joint action advertised');
  return {actions, request_ids};
}

const jointMoveBoundary = frame => !frame.outcome &&
  SIDES.every(side => frame.players[side].phase === 'move' && frame.players[side].mask.some(Boolean));

/** Coordinator-internal seam only; actor input never reaches `planFor`. */
class TamperedSession extends FullTrainingSession {
  constructor(mutate) { super(); this.mutate = mutate; }
  planFor(...args) { const plan = officialPlan(...args); this.mutate(plan); return plan; }
}

function workerClient() {
  const child = spawn(process.execPath, [WORKER], {stdio: ['pipe', 'pipe', 'inherit']});
  const replies = readline.createInterface({input: child.stdout, crlfDelay: Infinity})[Symbol.asyncIterator]();
  let id = 0;
  return {
    async send(command) {
      const payload = {...command, id: ++id};
      child.stdin.write(`${JSON.stringify(payload)}\n`);
      const {value, done} = await replies.next();
      if (done) throw new Error('Worker closed stdout before replying');
      const reply = JSON.parse(value);
      assert.equal(reply.id, payload.id);
      return reply;
    },
    close() { child.stdin.end(); child.kill(); },
  };
}

test('official plans admit two distinct teams inside their own frozen R7 partition', () => {
  assert.deepEqual([...SPLITS], ['train', 'development']);
  assert.equal(MAX_TEAM_ATTEMPTS, 1000);
  for (const split of SPLITS) {
    const plan = officialPlan(0, RUN_SEED, split);
    assert.equal(plan.teams.length, 2);
    assert.equal(plan.split, split);
    assert.equal(plan.swap, false);
    assert.equal(plan.provenance.namespace, `full-showdown-f2/${split}`);
    assert.equal(plan.provenance.partition, split);
    assert.equal(plan.provenance.attempt_budget, MAX_TEAM_ATTEMPTS);
    assert.ok(plan.provenance.attempts <= MAX_TEAM_ATTEMPTS);
    const keys = new Set();
    for (const entry of plan.teams) {
      const payload = teamArtifact(entry, ZERO).payload;
      assert.equal(entry.team.length, 6);
      assert.equal(entry.partition, split);
      assert.equal(partition(payload), split, 'admitted team must hash into its own split');
      assert.equal(teamKey(payload), entry.team_key);
      keys.add(entry.team_key);
    }
    assert.equal(keys.size, 2, 'the two admitted teams must be distinct');
  }
});

test('train and development episode pools are disjoint', () => {
  const pools = {train: new Set(), development: new Set()};
  for (const split of SPLITS) {
    for (let episodeIndex = 0; episodeIndex < 3; episodeIndex++) {
      for (const key of keysOf(officialPlan(episodeIndex, RUN_SEED, split))) pools[split].add(key);
    }
  }
  assert.ok(pools.train.size >= 2 && pools.development.size >= 2);
  const overlap = [...pools.train].filter(key => pools.development.has(key));
  assert.deepEqual(overlap, [], 'a team key may never appear in both splits');
});

test('plans are deterministic and swap reverses the identical pair', () => {
  for (const split of SPLITS) {
    const base = officialPlan(1, RUN_SEED, split);
    assert.deepEqual(keysOf(officialPlan(1, RUN_SEED, split)), keysOf(base), 'repeat calls must agree');
    const swapped = officialPlan(1, RUN_SEED, split, true);
    assert.deepEqual(keysOf(swapped), [...keysOf(base)].reverse());
    assert.deepEqual(swapped.seed, base.seed, 'seat swap must reuse the same mechanics seed');
    assert.equal(swapped.swap, true);
    assert.notDeepEqual(keysOf(officialPlan(2, RUN_SEED, split)), keysOf(base), 'episodes must differ');
    assert.notDeepEqual(officialPlan(1, RUN_SEED + 1, split).seed, base.seed, 'run seeds must differ');
  }
});

test('seat-paired evaluation starts the same two teams on reversed seats', async () => {
  const base = new FullTrainingSession(), swapped = new FullTrainingSession();
  try {
    const a = await base.start({episodeIndex: 4, runSeed: RUN_SEED});
    const b = await swapped.start({episodeIndex: 4, runSeed: RUN_SEED, swap: true});
    assert.equal(a.swap, false); assert.equal(b.swap, true);
    assert.equal(a.split, 'train'); assert.equal(b.split, 'train');
    assert.notEqual(a.episode_id, b.episode_id);
    assert.deepEqual(b.provenance.seed, a.provenance.seed);
    assert.deepEqual(b.provenance.teams.map(t => t.team_key), a.provenance.teams.map(t => t.team_key).reverse());
    assert.deepEqual(speciesOf(b, 'p2'), speciesOf(a, 'p1'));
    assert.deepEqual(speciesOf(b, 'p1'), speciesOf(a, 'p2'));
    for (const frame of [a, b]) {
      assert.equal(frame.schema_version, 'full-training-frame-v1');
      for (const side of SIDES) {
        assert.equal(frame.players[side].schema_version, 'full-showdown-player-v1');
        assert.equal(frame.players[side].phase, 'move');
        assert.equal(frame.players[side].mask.length, 14);
        assert.ok(frame.players[side].mask.some(Boolean));
        assert.equal(frame.players[side].own.length, 6);
        assert.equal(frame.players[side].opponent.length, 6);
        assert.equal(frame.players[side].opponent.filter(Boolean).length, 1);
        assert.equal(frame.players[side].opponent[0].ability, null);
      }
      assert.equal(JSON.stringify(frame.players).includes('team_key'), false);
    }
  } finally { base.close(); swapped.close(); }
});

test('development episodes start and keep the F1 actor boundary', async () => {
  const session = new FullTrainingSession();
  try {
    const frame = await session.start({episodeIndex: 0, runSeed: RUN_SEED, split: 'development'});
    assert.equal(frame.split, 'development');
    assert.ok(frame.provenance.teams.every(t => t.partition === 'development'));
    assert.equal(frame.provenance.scenario.namespace, 'full-showdown-f2/development');
    assert.equal(frame.outcome, null);
    assert.equal(frame.submissions, 0);
    await assert.rejects(session.step({actions: {p1: 0, p2: 0}, request_ids: {p1: -1, p2: -1}}), /Stale/);
    assert.equal(session.submissions, 0);
    await assert.rejects(session.step({actions: {p1: 0}, request_ids: {p1: frame.players.p1.request_id}}),
      /exactly the currently pending sides/);
    await assert.rejects(session.start({episodeIndex: 0, runSeed: RUN_SEED}), /already started/);
  } finally { session.close(); }
});

test('final, unknown splits, bad seeds and injected plans are all refused', async () => {
  assert.throws(() => officialPlan(0, RUN_SEED, 'final'), /final/);
  assert.throws(() => officialPlan(0, RUN_SEED, 'holdout'), /Unknown split/);
  assert.throws(() => officialPlan(0, RUN_SEED, 'train', 'yes'), /boolean/);
  assert.throws(() => officialPlan(-1, RUN_SEED), /episode index/);
  assert.throws(() => officialPlan(1.5, RUN_SEED), /episode index/);
  assert.throws(() => officialPlan(0, 2 ** 32), /run seed/);
  assert.throws(() => officialPlan(0, -1), /run seed/);
  assert.throws(() => officialPlan(0, '7781'), /run seed/);
  const cases = [
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'final'}, /final/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development ', swap: false}, /Unknown split/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: null}, /Unknown split/],
    [{episodeIndex: 0, runSeed: RUN_SEED, swap: 1}, /boolean/],
    [{episodeIndex: 0, runSeed: Number.NaN}, /run seed/],
    [{episodeIndex: 0, runSeed: RUN_SEED, plan: officialPlan(0, RUN_SEED)}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, curriculum: true}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, prefix: []}, /Unsupported start option/],
  ];
  for (const [options, expected] of cases) {
    const session = new FullTrainingSession();
    try {
      await assert.rejects(session.start(options), expected);
      assert.equal(session.provenance, undefined, 'a refused episode must never produce provenance');
    } finally { session.close(); }
  }
});

test('every admitted team is revalidated before the engine is created', async () => {
  const foreign = officialPlan(0, RUN_SEED, 'development').teams[0];
  const cases = [
    [plan => { plan.teams[0].partition = 'final'; }, /outside the admitted train partition/],
    [plan => { plan.teams[1] = {...foreign, partition: 'train'}; }, /outside the admitted train partition/],
    [plan => { plan.teams[0].team_key = ZERO; }, /Mismatched team key/],
    [plan => { plan.teams[1] = structuredClone(plan.teams[0]); }, /Duplicate admitted teams/],
    [plan => { plan.teams.pop(); }, /two admitted teams/],
    [plan => { plan.teams[0].team = plan.teams[0].team.slice(0, 5); }, /six-member/],
    [plan => { plan.seed = officialPlan(9, RUN_SEED).seed; }, /Nondeterministic mechanics seed/],
    [plan => { plan.split = 'development'; }, /disagrees/],
    [plan => { plan.swap = true; }, /disagrees/],
  ];
  for (const [mutate, expected] of cases) {
    const session = new TamperedSession(mutate);
    try {
      await assert.rejects(session.start({episodeIndex: 0, runSeed: RUN_SEED}), expected);
      assert.equal(session.provenance, undefined);
    } finally { session.close(); }
  }
});

test('real engine requests continue and cut off at a joint move boundary', async () => {
  const session = new FullTrainingSession();
  try {
    let frame = await session.start({episodeIndex: 3, runSeed: RUN_SEED});
    assert.ok(jointMoveBoundary(frame));
    frame = await session.step(liveActions(frame));
    assert.ok(frame.submitted.length >= 1);
    assert.ok(frame.submitted.every(row => SIDES.includes(row.side) && typeof row.accepted === 'boolean'));
    assert.ok(frame.submissions >= 1);
    for (let guard = 0; guard < 12 && !frame.outcome && !jointMoveBoundary(frame); guard++) {
      frame = await session.step(liveActions(frame));
    }
    assert.ok(frame.outcome || jointMoveBoundary(frame), 'the engine must return to a decidable boundary');
    if (frame.outcome) {
      assert.equal(frame.outcome.kind, 'terminal');
      assert.throws(() => session.cutoff(), /fresh joint move boundary/);
    } else {
      assert.ok(frame.players.p1.turn >= 1);
      const cut = session.cutoff();
      assert.equal(cut.outcome.kind, 'administrative_cutoff');
      assert.deepEqual(cut.outcome.rewards, {p1: 0, p2: 0});
      assert.equal(cut.outcome.winner, null);
      assert.ok(cut.players.p1.mask.some(Boolean) && cut.players.p2.mask.some(Boolean));
      await assert.rejects(session.step(liveActions(frame)), /completed session/);
      assert.throws(() => session.cutoff(), /fresh joint move boundary/);
    }
  } finally { session.close(); }
});

test('worker speaks hello/start/step/cutoff/close and rejects bad wire arguments', async () => {
  const worker = workerClient();
  try {
    const hello = await worker.send({op: 'hello'});
    assert.equal(hello.ok, true);
    assert.equal(hello.data.schema_version, 'full-training-worker-v1');
    assert.equal(typeof hello.data.reference_commit, 'string');
    assert.ok(hello.data.reference_commit.length > 0, 'hello must pin a reference commit');

    const started = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'development', swap: true});
    assert.equal(started.ok, true, started.error);
    assert.equal(started.data.split, 'development');
    assert.equal(started.data.swap, true);
    assert.ok(started.data.provenance.teams.every(t => t.partition === 'development'));
    assert.equal(JSON.stringify(started.data.players).includes('team_key'), false);
    assert.deepEqual(Object.keys(started.data.players).sort(), SIDES);

    const stepped = await worker.send({op: 'step', ...liveActions(started.data)});
    assert.equal(stepped.ok, true, stepped.error);
    assert.ok(stepped.data.submitted.length >= 1);

    const restarted = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: 0});
    assert.equal(restarted.ok, true, restarted.error);
    assert.equal(restarted.data.split, 'train');
    const cut = await worker.send({op: 'cutoff'});
    assert.equal(cut.ok, true, cut.error);
    assert.equal(cut.data.outcome.kind, 'administrative_cutoff');
    assert.deepEqual(cut.data.outcome.rewards, {p1: 0, p2: 0});

    const refusals = [
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'final'}, /final/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'holdout'}, /Unknown split/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: null}, /Unknown split/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, swap: 'yes'}, /boolean/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: -1}, /episode index/],
      [{op: 'start', run_seed: 2 ** 32, episode_index: 0}, /run seed/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, curriculum: true}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, plan: {}}, /Unsupported start field/],
      [{op: 'step', actions: {p1: 0}, request_ids: {p1: 0}}, /Unknown operation or missing session/],
      [{op: 'teleport'}, /Unknown operation/],
    ];
    for (const [command, expected] of refusals) {
      const reply = await worker.send(command);
      assert.equal(reply.ok, false, `expected refusal of ${JSON.stringify(command)}`);
      assert.match(reply.error, expected);
      assert.equal(reply.data, undefined, 'refusals must not leak frames');
    }

    const closed = await worker.send({op: 'close'});
    assert.equal(closed.ok, true);
    assert.deepEqual(closed.data, {closed: true});
  } finally { worker.close(); }
});

test('F2 session and worker open no network surface', () => {
  for (const name of ['session.mjs', 'worker.mjs']) {
    const source = readFileSync(new URL(`../../packages/full-training/${name}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /node:(net|http|https|dgram|tls)|require\(['"](net|http|https)['"]\)|fetch\(|WebSocket|XMLHttpRequest/,
      `${name} must not open a public network surface`);
  }
});

test('real Illusion wait request preserves own HP and roster without a foreign-private leak', () => {
  const evidence = JSON.parse(readFileSync(new URL('./illusion-wait.json', import.meta.url)));
  const session = new FullTrainingSession();
  try {
    for (const side of SIDES) {
      session.views[side].lastOwn = structuredClone(evidence[side].lastOwn);
      session.views[side].ownEvents = new Map(structuredClone(evidence[side].ownEvents));
      Object.assign(session.projection.players[side], {request: structuredClone(evidence[side].request),
        request_id: 50, pending: side === 'p1'});
    }
    // JSON drops undefined properties. Reapply the actual HP-less replace event
    // to reconstruct the in-memory failure instead of pretending JSON kept it.
    session.views.p2.ingest('|replace|p2a: Zoroark|Zoroark-Hisui, L80, M');
    // The actual failed adapter lost Zoroark's condition after the public replace.
    const old = session.views.p2.observe(null, 50);
    assert.equal(old.own.find(p => p.ident === 'Zoroark').condition, undefined);
    const repaired = session.snapshot();
    assert.equal(repaired.players.p2.phase, 'wait');
    assert.ok(repaired.players.p2.mask.every(x => !x));
    assert.equal(repaired.players.p2.own.find(p => p.ident === 'Zoroark').condition, '44/219');
    assert.equal(repaired.players.p2.own.find(p => p.ident === 'Toedscruel').condition, '183/281');
    assert.equal(repaired.players.p2.own.find(p => p.ident === 'Cinderace').tera, 'Fire');
    assert.deepEqual(repaired.players.p2.own.map(p => p.ident),
      evidence.p2.request.side.pokemon.map(p => p.ident.split(': ')[1]));
    session.outcome = {kind:'terminal', winner:'Full p1', turns:40, rewards:{p1:1,p2:0}};
    // Exercise undefined-condition removal before a faint can replace the field.
    assert.equal(session.snapshot().players.p2.own.find(p => p.ident === 'Zoroark').condition, '44/219');
    session.outcome = null;
    session.projection.players.p2.request.side.pokemon[0].condition = '100/219';
    const counterfactual = session.snapshot();
    assert.deepEqual(counterfactual.players.p1, repaired.players.p1);
    assert.equal(counterfactual.players.p2.own[0].condition, '100/219');
    session.views.p2.ingest('|faint|p2a: Zoroark');
    session.outcome = {kind:'terminal', winner:'Full p1', turns:40, rewards:{p1:1,p2:0}};
    const terminal = session.snapshot();
    assert.equal(terminal.players.p2.own.find(p => p.ident === 'Zoroark').condition, '0 fnt');
    assert.ok(terminal.players.p2.mask.every(x => !x));
  } finally { session.close(); }
});
