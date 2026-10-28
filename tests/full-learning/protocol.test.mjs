import test from 'node:test';
import assert from 'node:assert/strict';
import readline from 'node:readline';
import {readFileSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {FullLearningSession, TACTICAL_SIDES, TACTICAL_POLICY, SPLITS, validateTacticalSide} from '../../packages/full-learning/session.mjs';
import {FullTrainingSession} from '../../packages/full-training/session.mjs';
import {chooseOpponent} from '../../packages/showdown-panel/policy.mjs';
import {TacticalView} from '../../packages/showdown-panel/view.mjs';
import {ACTIONS} from '../../packages/full-showdown/actions.mjs';

const SIDES = ['p1', 'p2'];
const RUN_SEED = 7781;
const EPISODE = 2;
const WORKER = fileURLToPath(new URL('../../packages/full-learning/worker.mjs', import.meta.url));
const SWITCH_INDICES = ACTIONS.map((a, i) => a.startsWith('switch ') ? i : -1).filter(i => i >= 0);

const other = side => side === 'p1' ? 'p2' : 'p1';
const actionable = (frame, side) => !frame.outcome && frame.players[side].mask.some(Boolean);

/** Joint action: the advised seat follows its tactical action when actionable,
 * every other pending seat takes its first legal projected action. */
function jointActions(frame, tacticalSide = null) {
  const actions = {}, request_ids = {};
  for (const side of SIDES) {
    const player = frame.players[side];
    if (!player.mask.some(Boolean)) continue;
    const advised = side === tacticalSide ? frame.tactical_actions[side] : -1;
    actions[side] = advised >= 0 ? advised : player.mask.findIndex(Boolean);
    request_ids[side] = player.request_id;
  }
  if (!Object.keys(actions).length) throw new Error('No live joint action advertised');
  return {actions, request_ids};
}

/** Independent re-derivation of the advice from the seat's own player protocol
 * only (public events plus own private lines), with its own TacticalView. */
class IndependentAdvisor {
  constructor(side) { this.side = side; this.view = new TacticalView(); this.cursor = 0; }
  advise(session) {
    const lines = session.projection.players[this.side].protocol;
    for (; this.cursor < lines.length; this.cursor++) {
      const line = lines[this.cursor];
      if (line && !line.startsWith('|request|')) this.view.ingest(line);
    }
    const p = session.projection.players[this.side];
    if (!p.pending) return -1;
    const choice = chooseOpponent(TACTICAL_POLICY, structuredClone(p.request), structuredClone(this.view));
    return choice === null || choice === 'pass' ? -1 : ACTIONS.indexOf(choice);
  }
}

const stripTactical = frame => { const {tactical_actions, ...rest} = frame; return rest; };

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

test('F4 is an additive subclass of the unchanged F2 session', () => {
  assert.ok(FullLearningSession.prototype instanceof FullTrainingSession);
  assert.deepEqual([...TACTICAL_SIDES], SIDES);
  assert.deepEqual([...SPLITS], ['train', 'development']);
  assert.equal(TACTICAL_POLICY, 'tactical-v1');
  assert.equal(validateTacticalSide(null, 'train'), null);
  assert.equal(validateTacticalSide(undefined, 'train'), null);
  assert.equal(validateTacticalSide('p2', 'development'), 'p2');
  assert.throws(() => validateTacticalSide('p1', 'train'), /only admitted on development/);
  assert.throws(() => validateTacticalSide('p3', 'development'), /Unknown tactical side/);
  assert.throws(() => validateTacticalSide(1, 'development'), /Unknown tactical side/);
});

test('without a tactical side F4 frames equal F2 frames plus an all -1 tactical_actions', async () => {
  const f2 = new FullTrainingSession(), f4 = new FullLearningSession();
  try {
    let a = await f2.start({episodeIndex: EPISODE, runSeed: RUN_SEED, swap: true});
    let b = await f4.start({episodeIndex: EPISODE, runSeed: RUN_SEED, swap: true});
    assert.equal(f4.tacticalSide, null);
    assert.equal(f4.tacticalView, null);
    assert.deepEqual(b.tactical_actions, {p1: -1, p2: -1});
    assert.equal(b.schema_version, 'full-training-frame-v1');
    assert.equal(b.episode_id, a.episode_id);
    assert.deepEqual(stripTactical(b), a);
    for (let guard = 0; guard < 6 && !a.outcome; guard++) {
      const joint = jointActions(a);
      a = await f2.step(structuredClone(joint));
      b = await f4.step(structuredClone(joint));
      assert.deepEqual(b.tactical_actions, {p1: -1, p2: -1});
      assert.deepEqual(stripTactical(b), a, `step ${guard} must reproduce the F2 frame exactly`);
    }
  } finally { f2.close(); f4.close(); }
});

test('tactical advice equals the frozen policy applied directly to the own request and own protocol', async () => {
  for (const tacticalSide of SIDES) {
    const f2 = new FullTrainingSession(), f4 = new FullLearningSession();
    const independent = new IndependentAdvisor(tacticalSide);
    try {
      const coords = {episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'development'};
      let reference = await f2.start(coords);
      let frame = await f4.start({...coords, tacticalSide});
      assert.equal(f4.tacticalSide, tacticalSide);
      assert.ok(f4.tacticalView instanceof TacticalView);
      let advised = 0, waited = 0, forced = 0;
      for (let guard = 0; guard < 60; guard++) {
        // F2 mechanics, provenance and projected players are untouched.
        assert.deepEqual(stripTactical(frame), reference, `step ${guard} must reproduce the F2 frame exactly`);
        assert.deepEqual(Object.keys(frame.tactical_actions).sort(), SIDES);
        assert.equal(frame.tactical_actions[other(tacticalSide)], -1, 'the unadvised seat never receives advice');
        assert.equal(JSON.stringify(frame.players).includes('tactical'), false, 'advice must not enter player projections');
        const index = frame.tactical_actions[tacticalSide];
        assert.ok(Number.isInteger(index) && index >= -1 && index < ACTIONS.length);
        const player = frame.players[tacticalSide];
        if (actionable(frame, tacticalSide)) {
          assert.ok(index >= 0, 'an actionable advised seat must receive advice');
          assert.equal(player.mask[index], true, `advice ${ACTIONS[index]} must be legal under the projected mask`);
          assert.equal(index, independent.advise(f4), 'advice must equal the directly computed frozen policy choice');
          if (player.phase !== 'move') { forced++; assert.ok(SWITCH_INDICES.includes(index), 'forced switches only translate to switch actions'); }
          advised++;
        } else {
          assert.equal(index, -1, 'waiting, forced-pass and terminal seats carry -1');
          if (!frame.outcome) waited++;
        }
        if (frame.outcome) break;
        const joint = jointActions(frame, tacticalSide);
        reference = await f2.step(structuredClone(joint));
        frame = await f4.step(structuredClone(joint));
        assert.deepEqual(frame.submitted, reference.submitted);
      }
      assert.ok(advised >= 1, 'at least the opening frame must be advised');
      assert.ok(waited >= 0 && forced >= 0);
      // Public-only knowledge of the foe: percent HP, no private request fields.
      const foe = f4.tacticalView.active[other(tacticalSide)];
      assert.ok(foe, 'the advised view must have seen the foe switch in from its own protocol');
      assert.match(foe.condition, /^(\d+\/100(?: \w+)?|0 fnt)$/, 'foe HP in the advised view must be the public percent form');
      assert.ok(!('stats' in foe) && !('baseAbility' in foe), 'foe entries must carry no private request fields');
    } finally { f2.close(); f4.close(); }
  }
});

test('advice is a private counterfactual of the own seat only: foe private state cannot change it', async () => {
  const session = new FullLearningSession();
  try {
    const frame = await session.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p1'});
    const before = frame.tactical_actions.p1;
    assert.ok(before >= 0);
    const ownRequest = JSON.stringify(session.projection.players.p1.request);
    const ownView = JSON.stringify(session.tacticalView);
    const foeProtocol = session.projection.players.p2.protocol.length;
    // Tamper with the foe's private request: exact team HP (an impossible
    // denominator no real member can carry) and action availability.
    const p2 = session.projection.players.p2;
    for (const mon of p2.request.side.pokemon) mon.condition = '7/777';
    p2.request = {wait: true, side: p2.request.side}; p2.pending = false;
    const again = session.snapshot();
    assert.equal(again.players.p2.phase, 'wait');
    assert.ok(again.players.p2.mask.every(x => !x));
    assert.equal(again.tactical_actions.p1, before, 'foe private state must not influence the advice');
    assert.equal(again.tactical_actions.p2, -1);
    // Evaluating advice must not mutate the own request or the advised view.
    assert.equal(JSON.stringify(session.projection.players.p1.request), ownRequest);
    assert.equal(JSON.stringify(session.tacticalView), ownView);
    assert.equal(session.tacticalCursor, session.projection.players.p1.protocol.length);
    assert.equal(session.projection.players.p2.protocol.length, foeProtocol);
    // The own seat's HP in the advised view is the private exact form, proving
    // the view is fed from the own player channel and not the spectator channel.
    const own = session.tacticalView.active.p1;
    assert.ok(own && typeof own.condition === 'string');
    assert.doesNotMatch(JSON.stringify(session.tacticalView), /7\/777/, 'tampered foe private data must be absent from the advised view');
    assert.ok(JSON.stringify(again.players.p2.own).includes('7/777'), 'the tamper must have been visible to the foe projection itself');
  } finally { session.close(); }
});

test('waiting and forced-switch masks: -1 while waiting, a legal switch index when forced', async () => {
  const session = new FullLearningSession();
  try {
    const frame = await session.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p2'});
    assert.ok(frame.tactical_actions.p2 >= 0);
    const p1 = session.projection.players.p1, p2 = session.projection.players.p2;
    const p1Live = structuredClone(p1.request);
    // Advised seat waiting: a wait request keeps its own side payload but no actions.
    const live = structuredClone(p2.request);
    p2.request = {wait: true, side: structuredClone(live.side)}; p2.pending = false; p2.request_id++;
    const waiting = session.snapshot();
    assert.equal(waiting.players.p2.phase, 'wait');
    assert.ok(waiting.players.p2.mask.every(x => !x));
    assert.deepEqual(waiting.tactical_actions, {p1: -1, p2: -1});
    // Advised seat forced to replace its fainted active member.
    const forced = structuredClone(live);
    delete forced.active;
    forced.forceSwitch = [true];
    forced.side.pokemon[0].condition = '0 fnt';
    p2.request = forced; p2.pending = true; p2.request_id++;
    p1.request = {wait: true, side: structuredClone(p1.request.side)}; p1.pending = false; p1.request_id++;
    const replacing = session.snapshot();
    assert.ok(replacing.players.p1.mask.every(x => !x));
    assert.equal(replacing.tactical_actions.p1, -1);
    const index = replacing.tactical_actions.p2;
    assert.ok(SWITCH_INDICES.includes(index), `forced replacement advice must be a switch, got ${index}`);
    assert.equal(replacing.players.p2.mask[index], true);
    assert.ok(replacing.players.p2.mask.every((legal, i) => !legal || SWITCH_INDICES.includes(i)), 'a forced switch advertises switches only');
    const direct = chooseOpponent(TACTICAL_POLICY, structuredClone(forced), structuredClone(session.tacticalView));
    assert.equal(ACTIONS[index], direct);
    // A forced pass (no replacement owed) is coordinator control, never advice.
    // p1 is restored to its live move request so the frame itself stays decidable.
    p1.request = structuredClone(p1Live); p1.pending = true; p1.request_id++;
    p2.request = {...structuredClone(forced), forceSwitch: [false]}; p2.request_id++;
    const passing = session.snapshot();
    assert.ok(passing.players.p2.mask.every(x => !x));
    assert.deepEqual(passing.tactical_actions, {p1: -1, p2: -1});
  } finally { session.close(); }
});

test('an illegal frozen choice is refused rather than replaced by another policy', async () => {
  const session = new FullLearningSession();
  try {
    await session.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p1'});
    const mask = ACTIONS.map(() => false);
    assert.throws(() => session.tacticalAction('p1', mask), /not legal under the current projected mask/);
    const foreign = session.projection.players.p1;
    const saved = foreign.request;
    foreign.request = structuredClone(session.projection.players.p2.request);
    assert.throws(() => session.tacticalAction('p1', ACTIONS.map(() => true)), /foreign private request/);
    foreign.request = saved;
  } finally { session.close(); }
});

test('train, final, unknown sides and unknown options are refused before anything starts', async () => {
  const cases = [
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p1'}, /only admitted on development/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'train', tacticalSide: 'p2'}, /only admitted on development/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'final', tacticalSide: 'p1'}, /final/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'holdout', tacticalSide: 'p1'}, /Unknown split/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p3'}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: 'P1'}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: true}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: ['p1']}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p1', swap: 'yes'}, /boolean/],
    [{episodeIndex: -1, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p1'}, /episode index/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', policy: 'tactical-v1'}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tactical_side: 'p1'}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p1', plan: {}}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, curriculum: true}, /Unsupported start option/],
  ];
  for (const [options, expected] of cases) {
    const session = new FullLearningSession();
    try {
      await assert.rejects(session.start(options), expected);
      assert.equal(session.started, false, 'a refused episode must never start');
      assert.equal(session.provenance, undefined, 'a refused episode must never produce provenance');
      assert.equal(session.tacticalView, null);
    } finally { session.close(); }
  }
  await assert.rejects(new FullLearningSession().start(null), /Expected start options object/);
  await assert.rejects(new FullLearningSession().start([]), /Expected start options object/);
  const started = new FullLearningSession();
  try {
    await started.start({episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p1'});
    await assert.rejects(started.start({episodeIndex: 0, runSeed: RUN_SEED, split: 'development'}), /already started/);
  } finally { started.close(); }
});

test('worker speaks full-learning-worker-v1, dispatches tactical_side and cleans up on refusal', async () => {
  const worker = workerClient();
  try {
    const hello = await worker.send({op: 'hello'});
    assert.equal(hello.ok, true);
    assert.equal(hello.data.schema_version, 'full-learning-worker-v1');
    assert.equal(typeof hello.data.reference_commit, 'string');
    assert.ok(hello.data.reference_commit.length > 0);

    const started = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: EPISODE, split: 'development', swap: true, tactical_side: 'p2'});
    assert.equal(started.ok, true, started.error);
    assert.equal(started.data.schema_version, 'full-training-frame-v1');
    assert.equal(started.data.split, 'development');
    assert.equal(started.data.swap, true);
    assert.equal(started.data.tactical_actions.p1, -1);
    assert.ok(started.data.tactical_actions.p2 >= 0 && started.data.players.p2.mask[started.data.tactical_actions.p2]);
    assert.equal(JSON.stringify(started.data.players).includes('team_key'), false);
    assert.equal(JSON.stringify(started.data.players).includes('tactical'), false);

    const stepped = await worker.send({op: 'step', ...jointActions(started.data, 'p2')});
    assert.equal(stepped.ok, true, stepped.error);
    assert.ok(stepped.data.submitted.length >= 1);
    assert.deepEqual(Object.keys(stepped.data.tactical_actions).sort(), SIDES);
    assert.equal(stepped.data.tactical_actions.p1, -1);

    const tacticalStart = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: EPISODE,
      split: 'development', tactical_side: 'p2'});
    assert.ok(tacticalStart.data.tactical_actions.p2 >= 0);
    const tacticalCutoff = await worker.send({op: 'cutoff'});
    assert.equal(tacticalCutoff.ok, true, tacticalCutoff.error);
    assert.equal(tacticalCutoff.data.outcome.kind, 'administrative_cutoff');
    assert.deepEqual(tacticalCutoff.data.tactical_actions, {p1: -1, p2: -1});
    assert.deepEqual(tacticalCutoff.data.players, tacticalStart.data.players);

    const plain = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: 0});
    assert.equal(plain.ok, true, plain.error);
    assert.equal(plain.data.split, 'train');
    assert.deepEqual(plain.data.tactical_actions, {p1: -1, p2: -1});
    const explicitNull = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: 0, tactical_side: null});
    assert.equal(explicitNull.ok, true, explicitNull.error);
    assert.deepEqual(explicitNull.data.tactical_actions, {p1: -1, p2: -1});
    const cut = await worker.send({op: 'cutoff'});
    assert.equal(cut.ok, true, cut.error);
    assert.equal(cut.data.outcome.kind, 'administrative_cutoff');

    const refusals = [
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, tactical_side: 'p1'}, /only admitted on development/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'train', tactical_side: 'p2'}, /only admitted on development/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'final', tactical_side: 'p1'}, /final/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'holdout', tactical_side: 'p1'}, /Unknown split/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'development', tactical_side: 'p3'}, /Unknown tactical side/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'development', tactical_side: 1}, /Unknown tactical side/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'development', tacticalSide: 'p1'}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'development', policy: 'tactical-v1'}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, plan: {}}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, swap: 'yes'}, /boolean/],
      [{op: 'start', run_seed: 2 ** 32, episode_index: 0, split: 'development', tactical_side: 'p1'}, /run seed/],
    ];
    for (const [command, expected] of refusals) {
      const reply = await worker.send(command);
      assert.equal(reply.ok, false, `expected refusal of ${JSON.stringify(command)}`);
      assert.match(reply.error, expected);
      assert.equal(reply.data, undefined, 'refusals must not leak frames');
      // Error cleanup: the previous session is gone, so a step has no session.
      const orphan = await worker.send({op: 'step', actions: {p1: 0}, request_ids: {p1: 0}});
      assert.equal(orphan.ok, false);
      assert.match(orphan.error, /Unknown operation or missing session/);
    }
    assert.match((await worker.send({op: 'teleport'})).error, /Unknown operation/);

    const closed = await worker.send({op: 'close'});
    assert.equal(closed.ok, true);
    assert.deepEqual(closed.data, {closed: true});
  } finally { worker.close(); }
});

test('F4 session and worker open no network surface and leave F2 sources untouched by import', () => {
  for (const name of ['session.mjs', 'worker.mjs']) {
    const source = readFileSync(new URL(`../../packages/full-learning/${name}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /node:(net|http|https|dgram|tls)|require\(['"](net|http|https)['"]\)|fetch\(|WebSocket|XMLHttpRequest/,
      `${name} must not open a public network surface`);
  }
  const f4 = readFileSync(new URL('../../packages/full-learning/session.mjs', import.meta.url), 'utf8');
  assert.match(f4, /from '\.\.\/full-training\/session\.mjs'/, 'F4 must reuse the F2 session rather than fork it');
  assert.match(f4, /chooseOpponent\(TACTICAL_POLICY/, 'F4 must call the frozen tactical policy');
  assert.doesNotMatch(f4, /BattleStream|stream\.write|'>p[12]/, 'F4 must never drive the engine itself');
});
