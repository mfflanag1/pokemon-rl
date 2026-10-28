import test from 'node:test';
import assert from 'node:assert/strict';
import readline from 'node:readline';
import {readFileSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {OpponentMixtureSession, TACTICAL_SIDES, TACTICAL_SPLITS, TACTICAL_POLICY, SPLITS,
  validateMixtureSide, validateMixtureStart} from '../../packages/opponent-mixture/session.mjs';
import {FullTrainingSession} from '../../packages/full-training/session.mjs';
import {FullLearningSession, validateTacticalSide as validateF4Side} from '../../packages/full-learning/session.mjs';
import {chooseOpponent} from '../../packages/showdown-panel/policy.mjs';
import {TacticalView} from '../../packages/showdown-panel/view.mjs';
import {ACTIONS} from '../../packages/full-showdown/actions.mjs';

const SIDES = ['p1', 'p2'];
const RUN_SEED = 7781;
const EPISODE = 2;
const WORKER = fileURLToPath(new URL('../../packages/opponent-mixture/worker.mjs', import.meta.url));
const F4_SOURCE = new URL('../../packages/full-learning/session.mjs', import.meta.url);
const SWITCH_INDICES = ACTIONS.map((a, i) => a.startsWith('switch ') ? i : -1).filter(i => i >= 0);

const other = side => side === 'p1' ? 'p2' : 'p1';
const actionable = (frame, side) => !frame.outcome && frame.players[side].mask.some(Boolean);
const stripTactical = frame => { const {tactical_actions, ...rest} = frame; return rest; };

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
 * only (public events plus own private lines), with its own TacticalView. It
 * never touches the session's advisor view or the other seat's request. */
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

/** Advised run against an F2 reference fed the identical joint actions. Checks
 * frame parity, legality, isolation from projections and agreement with the
 * independent advisor on every frame. Returns the advised/waited/forced tallies. */
async function runAdvisedParity(split, tacticalSide, guardLimit) {
  const f2 = new FullTrainingSession(), f5 = new OpponentMixtureSession();
  const independent = new IndependentAdvisor(tacticalSide);
  try {
    const coords = {episodeIndex: EPISODE, runSeed: RUN_SEED, split};
    let reference = await f2.start(coords);
    let frame = await f5.start({...coords, tacticalSide});
    assert.equal(f5.tacticalSide, tacticalSide);
    assert.equal(f5.split, split);
    assert.ok(f5.tacticalView instanceof TacticalView);
    let advised = 0, waited = 0, forced = 0;
    for (let guard = 0; guard < guardLimit; guard++) {
      // F2 mechanics, provenance and projected players are untouched.
      assert.deepEqual(stripTactical(frame), reference, `${split}/${tacticalSide} step ${guard} must reproduce the F2 frame exactly`);
      assert.equal(frame.schema_version, 'full-training-frame-v1');
      assert.equal(frame.split, split);
      assert.deepEqual(Object.keys(frame.tactical_actions).sort(), SIDES);
      assert.equal(frame.tactical_actions[other(tacticalSide)], -1, 'the unadvised seat never receives advice');
      assert.equal(JSON.stringify(frame.players).includes('tactical'), false, 'advice must not enter player projections');
      assert.equal(JSON.stringify(frame.provenance).includes('tactical'), false, 'advice must not enter provenance');
      const index = frame.tactical_actions[tacticalSide];
      assert.ok(Number.isInteger(index) && index >= -1 && index < ACTIONS.length);
      const player = frame.players[tacticalSide];
      if (actionable(frame, tacticalSide)) {
        assert.ok(index >= 0, 'an actionable advised seat must receive advice');
        assert.equal(player.mask[index], true, `advice ${ACTIONS[index]} must be legal under the projected mask`);
        assert.equal(index, independent.advise(f5), 'advice must equal the directly computed frozen policy choice');
        if (player.phase !== 'move') { forced++; assert.ok(SWITCH_INDICES.includes(index), 'forced switches only translate to switch actions'); }
        advised++;
      } else {
        assert.equal(index, -1, 'waiting, forced-pass and terminal seats carry -1');
        if (!frame.outcome) waited++;
      }
      if (frame.outcome) break;
      const joint = jointActions(frame, tacticalSide);
      reference = await f2.step(structuredClone(joint));
      frame = await f5.step(structuredClone(joint));
      assert.deepEqual(frame.submitted, reference.submitted);
    }
    assert.ok(advised >= 1, 'at least the opening frame must be advised');
    // Public-only knowledge of the foe: percent HP, no private request fields.
    const foe = f5.tacticalView.active[other(tacticalSide)];
    assert.ok(foe, 'the advised view must have seen the foe switch in from its own protocol');
    assert.match(foe.condition, /^(\d+\/100(?: \w+)?|0 fnt)$/, 'foe HP in the advised view must be the public percent form');
    assert.ok(!('stats' in foe) && !('baseAbility' in foe), 'foe entries must carry no private request fields');
    return {advised, waited, forced};
  } finally { f2.close(); f5.close(); }
}

test('F5 is an additive subclass of the unchanged F2 session, copied from F4 rather than bypassing it', () => {
  assert.ok(OpponentMixtureSession.prototype instanceof FullTrainingSession);
  assert.ok(!(OpponentMixtureSession.prototype instanceof FullLearningSession), 'F5 must not subclass F4 and override its validator');
  assert.deepEqual([...TACTICAL_SIDES], SIDES);
  assert.deepEqual([...TACTICAL_SPLITS], ['train', 'development']);
  assert.deepEqual([...SPLITS], ['train', 'development']);
  assert.equal(TACTICAL_POLICY, 'tactical-v1');
  for (const split of SPLITS) {
    assert.equal(validateMixtureSide(null, split), null);
    assert.equal(validateMixtureSide(undefined, split), null);
    assert.equal(validateMixtureSide('p1', split), 'p1');
    assert.equal(validateMixtureSide('p2', split), 'p2');
    assert.throws(() => validateMixtureSide('p3', split), /Unknown tactical side/);
    assert.throws(() => validateMixtureSide('P1', split), /Unknown tactical side/);
    assert.throws(() => validateMixtureSide(1, split), /Unknown tactical side/);
    assert.throws(() => validateMixtureSide(['p1'], split), /Unknown tactical side/);
  }
  assert.throws(() => validateMixtureSide('p1', 'final'), /final/);
  assert.throws(() => validateMixtureSide('p2', 'final'), /final/);
  assert.throws(() => validateMixtureSide('p1', 'holdout'), /only admitted on train or development/);
  assert.throws(() => validateMixtureSide('p1', undefined), /only admitted on train or development/);
  // A null seat on final is still null here; F2 refuses the split itself at start.
  assert.equal(validateMixtureSide(null, 'final'), null);
  const admitted = validateMixtureStart({episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p2'});
  assert.deepEqual(admitted, {episode: {episodeIndex: 0, runSeed: RUN_SEED}, tacticalSide: 'p2'});
  assert.throws(() => validateMixtureStart({episodeIndex: 0, runSeed: RUN_SEED, split: 'final', tacticalSide: 'p1'}), /final/);
  assert.throws(() => validateMixtureStart({episodeIndex: 0, runSeed: RUN_SEED, opponentRequest: {}}), /Unsupported start option/);
});

test('frozen F4 still refuses train tactical seats: F5 adds train admission without touching F4', async () => {
  assert.throws(() => validateF4Side('p1', 'train'), /only admitted on development/);
  assert.throws(() => validateF4Side('p2', 'train'), /only admitted on development/);
  assert.equal(validateF4Side('p1', 'development'), 'p1');
  const f4 = new FullLearningSession();
  try {
    await assert.rejects(f4.start({episodeIndex: 0, runSeed: RUN_SEED, split: 'train', tacticalSide: 'p1'}), /only admitted on development/);
    assert.equal(f4.started, false);
  } finally { f4.close(); }
  const source = readFileSync(F4_SOURCE, 'utf8');
  assert.match(source, /split !== 'development'/, 'the frozen F4 validator must keep its development-only guard');
  // The same coordinates are admitted by F5 - the only intended difference.
  assert.equal(validateMixtureSide('p1', 'train'), 'p1');
});

test('without a tactical side F5 frames equal F2 frames plus an all -1 tactical_actions', async () => {
  const f2 = new FullTrainingSession(), f5 = new OpponentMixtureSession();
  try {
    let a = await f2.start({episodeIndex: EPISODE, runSeed: RUN_SEED, swap: true});
    let b = await f5.start({episodeIndex: EPISODE, runSeed: RUN_SEED, swap: true});
    assert.equal(f5.tacticalSide, null);
    assert.equal(f5.tacticalView, null);
    assert.equal(b.split, 'train');
    assert.deepEqual(b.tactical_actions, {p1: -1, p2: -1});
    assert.equal(b.schema_version, 'full-training-frame-v1');
    assert.equal(b.episode_id, a.episode_id);
    assert.deepEqual(stripTactical(b), a);
    for (let guard = 0; guard < 6 && !a.outcome; guard++) {
      const joint = jointActions(a);
      a = await f2.step(structuredClone(joint));
      b = await f5.step(structuredClone(joint));
      assert.deepEqual(b.tactical_actions, {p1: -1, p2: -1});
      assert.deepEqual(stripTactical(b), a, `step ${guard} must reproduce the F2 frame exactly`);
    }
  } finally { f2.close(); f5.close(); }
});

test('train tactical advice in either seat equals the frozen policy on own request and own protocol with exact F2 frame parity', async () => {
  for (const tacticalSide of SIDES) {
    const {advised} = await runAdvisedParity('train', tacticalSide, 60);
    assert.ok(advised >= 1, `train/${tacticalSide} must be advised`);
  }
});

test('development tactical advice remains available in either seat with exact F2 frame parity', async () => {
  for (const tacticalSide of SIDES) {
    const {advised} = await runAdvisedParity('development', tacticalSide, 12);
    assert.ok(advised >= 1, `development/${tacticalSide} must be advised`);
  }
});

test('train and development advised episodes stay on disjoint F2 team pools and seeds', async () => {
  const train = new OpponentMixtureSession(), dev = new OpponentMixtureSession();
  try {
    const a = await train.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'train', tacticalSide: 'p1'});
    const b = await dev.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p1'});
    assert.equal(a.provenance.split, 'train');
    assert.equal(b.provenance.split, 'development');
    assert.ok(a.provenance.teams.every(t => t.partition === 'train'));
    assert.ok(b.provenance.teams.every(t => t.partition === 'development'));
    const trainKeys = new Set(a.provenance.teams.map(t => t.team_key));
    assert.ok(b.provenance.teams.every(t => !trainKeys.has(t.team_key)), 'team pools must not overlap across splits');
    assert.notEqual(JSON.stringify(a.provenance.seed), JSON.stringify(b.provenance.seed), 'mechanics seeds are namespaced per split');
    assert.notEqual(a.episode_id, b.episode_id);
    assert.ok(a.tactical_actions.p1 >= 0 && b.tactical_actions.p1 >= 0);
  } finally { train.close(); dev.close(); }
});

test('advice is a private counterfactual of the own seat only: foe private state cannot change it (train)', async () => {
  const session = new OpponentMixtureSession();
  try {
    const frame = await session.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'train', tacticalSide: 'p1'});
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

test('waiting and forced-switch masks on train: -1 while waiting, a legal switch index when forced', async () => {
  const session = new OpponentMixtureSession();
  try {
    const frame = await session.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'train', tacticalSide: 'p2'});
    assert.ok(frame.tactical_actions.p2 >= 0);
    const p1 = session.projection.players.p1, p2 = session.projection.players.p2;
    const p1Live = structuredClone(p1.request);
    const live = structuredClone(p2.request);
    p2.request = {wait: true, side: structuredClone(live.side)}; p2.pending = false; p2.request_id++;
    const waiting = session.snapshot();
    assert.equal(waiting.players.p2.phase, 'wait');
    assert.ok(waiting.players.p2.mask.every(x => !x));
    assert.deepEqual(waiting.tactical_actions, {p1: -1, p2: -1});
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
    const direct = chooseOpponent(TACTICAL_POLICY, structuredClone(forced), structuredClone(session.tacticalView));
    assert.equal(ACTIONS[index], direct);
    // A forced pass (no replacement owed) is coordinator control, never advice.
    p1.request = structuredClone(p1Live); p1.pending = true; p1.request_id++;
    p2.request = {...structuredClone(forced), forceSwitch: [false]}; p2.request_id++;
    const passing = session.snapshot();
    assert.ok(passing.players.p2.mask.every(x => !x));
    assert.deepEqual(passing.tactical_actions, {p1: -1, p2: -1});
  } finally { session.close(); }
});

test('an illegal frozen choice or a foreign private request is refused rather than substituted', async () => {
  const session = new OpponentMixtureSession();
  try {
    await session.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'train', tacticalSide: 'p1'});
    const mask = ACTIONS.map(() => false);
    assert.throws(() => session.tacticalAction('p1', mask), /not legal under the current projected mask/);
    const own = session.projection.players.p1;
    const saved = own.request;
    own.request = structuredClone(session.projection.players.p2.request);
    assert.throws(() => session.tacticalAction('p1', ACTIONS.map(() => true)), /foreign private request/);
    own.request = saved;
  } finally { session.close(); }
});

test('cutoff clears advice on a train advised episode while keeping the live F2 observations', async () => {
  for (const tacticalSide of SIDES) {
    const session = new OpponentMixtureSession();
    try {
      const frame = await session.start({episodeIndex: EPISODE, runSeed: RUN_SEED, split: 'train', tacticalSide});
      assert.ok(frame.tactical_actions[tacticalSide] >= 0, 'the opening train frame must be advised');
      const cut = session.cutoff();
      assert.equal(cut.outcome.kind, 'administrative_cutoff');
      assert.deepEqual(cut.outcome.rewards, {p1: 0, p2: 0});
      assert.deepEqual(cut.tactical_actions, {p1: -1, p2: -1}, 'advice must be cleared after the administrative outcome');
      assert.deepEqual(cut.players, frame.players, 'cutoff keeps the live projected observations for bootstrapping');
      assert.deepEqual(cut.provenance, frame.provenance);
      // Any later rendering is terminal and carries no advice either.
      const after = session.snapshot();
      assert.equal(after.outcome.kind, 'administrative_cutoff');
      assert.deepEqual(after.tactical_actions, {p1: -1, p2: -1});
      await assert.rejects(session.step(jointActions(frame, tacticalSide)), /completed session/);
    } finally { session.close(); }
  }
});

test('final, unknown sides, unknown/private options are refused before anything starts; train is admitted', async () => {
  const cases = [
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'final', tacticalSide: 'p1'}, /final/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'final'}, /final/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'holdout', tacticalSide: 'p1'}, /Unknown split/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'train', tacticalSide: 'p3'}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'P1'}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: true}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: ['p1']}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, split: 'development', tacticalSide: 'p3'}, /Unknown tactical side/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p1', swap: 'yes'}, /boolean/],
    [{episodeIndex: -1, runSeed: RUN_SEED, tacticalSide: 'p1'}, /episode index/],
    [{episodeIndex: 0, runSeed: 2 ** 32, tacticalSide: 'p1'}, /run seed/],
    [{episodeIndex: 0, runSeed: RUN_SEED, policy: 'tactical-v1'}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tactical_side: 'p1'}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p1', plan: {}}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, curriculum: true}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p1', opponentRequest: {}}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p1', mixture: 0.5}, /Unsupported start option/],
    [{episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p1', tacticalView: {}}, /Unsupported start option/],
  ];
  for (const [options, expected] of cases) {
    const session = new OpponentMixtureSession();
    try {
      await assert.rejects(session.start(options), expected);
      assert.equal(session.started, false, 'a refused episode must never start');
      assert.equal(session.provenance, undefined, 'a refused episode must never produce provenance');
      assert.equal(session.tacticalView, null);
      assert.equal(session.tacticalSide, null);
    } finally { session.close(); }
  }
  await assert.rejects(new OpponentMixtureSession().start(null), /Expected start options object/);
  await assert.rejects(new OpponentMixtureSession().start([]), /Expected start options object/);
  const started = new OpponentMixtureSession();
  try {
    const frame = await started.start({episodeIndex: 0, runSeed: RUN_SEED, tacticalSide: 'p1'});
    assert.equal(frame.split, 'train');
    assert.ok(frame.tactical_actions.p1 >= 0);
    await assert.rejects(started.start({episodeIndex: 0, runSeed: RUN_SEED}), /already started/);
  } finally { started.close(); }
});

test('worker speaks opponent-mixture-worker-v1, dispatches tactical_side on train and development, cleans up on refusal', async () => {
  const worker = workerClient();
  try {
    const hello = await worker.send({op: 'hello'});
    assert.equal(hello.ok, true);
    assert.equal(hello.data.schema_version, 'opponent-mixture-worker-v1');
    assert.equal(typeof hello.data.reference_commit, 'string');
    assert.ok(hello.data.reference_commit.length > 0);

    // Train tactical in both seats over the wire.
    for (const side of SIDES) {
      const started = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: EPISODE, split: 'train', tactical_side: side});
      assert.equal(started.ok, true, started.error);
      assert.equal(started.data.schema_version, 'full-training-frame-v1');
      assert.equal(started.data.split, 'train');
      assert.equal(started.data.tactical_actions[other(side)], -1);
      assert.ok(started.data.tactical_actions[side] >= 0 && started.data.players[side].mask[started.data.tactical_actions[side]]);
      assert.equal(JSON.stringify(started.data.players).includes('team_key'), false);
      assert.equal(JSON.stringify(started.data.players).includes('tactical'), false);
      const stepped = await worker.send({op: 'step', ...jointActions(started.data, side)});
      assert.equal(stepped.ok, true, stepped.error);
      assert.ok(stepped.data.submitted.length >= 1);
      assert.deepEqual(Object.keys(stepped.data.tactical_actions).sort(), SIDES);
      assert.equal(stepped.data.tactical_actions[other(side)], -1);
    }

    // Implicit train split with a tactical seat, then cutoff clears advice.
    const implicit = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: EPISODE, tactical_side: 'p2'});
    assert.equal(implicit.ok, true, implicit.error);
    assert.equal(implicit.data.split, 'train');
    assert.ok(implicit.data.tactical_actions.p2 >= 0);
    const cut = await worker.send({op: 'cutoff'});
    assert.equal(cut.ok, true, cut.error);
    assert.equal(cut.data.outcome.kind, 'administrative_cutoff');
    assert.deepEqual(cut.data.tactical_actions, {p1: -1, p2: -1});
    assert.deepEqual(cut.data.players, implicit.data.players);

    // Development with swap keeps working exactly as in F4.
    const dev = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: EPISODE, split: 'development', swap: true, tactical_side: 'p2'});
    assert.equal(dev.ok, true, dev.error);
    assert.equal(dev.data.split, 'development');
    assert.equal(dev.data.swap, true);
    assert.equal(dev.data.tactical_actions.p1, -1);
    assert.ok(dev.data.tactical_actions.p2 >= 0 && dev.data.players.p2.mask[dev.data.tactical_actions.p2]);

    const plain = await worker.send({op: 'start', run_seed: RUN_SEED, episode_index: 0, tactical_side: null});
    assert.equal(plain.ok, true, plain.error);
    assert.deepEqual(plain.data.tactical_actions, {p1: -1, p2: -1});

    const refusals = [
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'final', tactical_side: 'p1'}, /final/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'final'}, /final/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'holdout', tactical_side: 'p1'}, /Unknown split/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'train', tactical_side: 'p3'}, /Unknown tactical side/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, tactical_side: 1}, /Unknown tactical side/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, split: 'development', tactical_side: 'P2'}, /Unknown tactical side/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, tacticalSide: 'p1'}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, policy: 'tactical-v1'}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, plan: {}}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, opponent_request: {}}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, mixture: 0.5}, /Unsupported start field/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: 0, swap: 'yes'}, /boolean/],
      [{op: 'start', run_seed: 2 ** 32, episode_index: 0, tactical_side: 'p1'}, /run seed/],
      [{op: 'start', run_seed: RUN_SEED, episode_index: -1, tactical_side: 'p1'}, /episode index/],
    ];
    for (const [command, expected] of refusals) {
      const reply = await worker.send(command);
      assert.equal(reply.ok, false, `expected refusal of ${JSON.stringify(command)}`);
      assert.match(reply.error, expected);
      assert.equal(reply.data, undefined, 'refusals must not leak frames');
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

test('F5 sources open no network surface, reuse F2 directly, never import F4 and never drive the engine', () => {
  for (const name of ['session.mjs', 'worker.mjs']) {
    const source = readFileSync(new URL(`../../packages/opponent-mixture/${name}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /node:(net|http|https|dgram|tls)|require\(['"](net|http|https)['"]\)|fetch\(|WebSocket|XMLHttpRequest/,
      `${name} must not open a public network surface`);
    assert.doesNotMatch(source, /full-learning/, `${name} must not import or bypass the frozen F4 package`);
  }
  const f5 = readFileSync(new URL('../../packages/opponent-mixture/session.mjs', import.meta.url), 'utf8');
  assert.match(f5, /from '\.\.\/full-training\/session\.mjs'/, 'F5 must reuse the F2 session rather than fork it');
  assert.match(f5, /chooseOpponent\(TACTICAL_POLICY/, 'F5 must call the frozen tactical policy');
  assert.doesNotMatch(f5, /BattleStream|stream\.write|'>p[12]/, 'F5 must never drive the engine itself');
  assert.doesNotMatch(f5, /officialPlan\(|generateTeam|seedFor/, 'F5 must not re-implement F2 team admission or seeding');
});
