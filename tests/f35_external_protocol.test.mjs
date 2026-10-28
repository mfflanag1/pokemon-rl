import test from 'node:test';
import assert from 'node:assert/strict';
import {ExternalBenchmarkSession} from '../packages/external-benchmark/session.mjs';
import {seedFor} from '../packages/reference-node/engine.mjs';
import {choiceFor} from '../packages/full-showdown/actions.mjs';
import {validateCommand, submittedStep} from '../packages/teacher-assisted/external-worker.mjs';

const start = {op: 'start', id: 1, run_seed: 102735001, episode_index: 0, swap: false,
  split: 'train', external_side: 'p2', tactical_side: null};

test('F35 split admission is explicit and refuses final before session creation', () => {
  assert.equal(validateCommand({...start}).split, 'train');
  assert.equal(validateCommand({...start, split: 'development'}).split, 'development');
  for (const split of ['final', undefined, 'test', null]) assert.throws(() => validateCommand({...start, split}));
  assert.throws(() => validateCommand({...start, private_team: {}}));
  assert.throws(() => validateCommand({...start, tactical_side: 'p2'}));
  assert.throws(() => validateCommand({...start, swap: 0}));
});

test('fresh train/development session projections retain declared partition and isolate teacher', async () => {
  for (const split of ['train', 'development']) {
    const session = new ExternalBenchmarkSession('p2');
    try {
      const frame = await session.start({episodeIndex: 0, runSeed: 102735001, split, swap: false, tacticalSide: null});
      assert.equal(frame.split, split);
      assert(frame.provenance.teams.every(team => team.partition === split));
      const expected = session.projection.players.p2.protocol.slice();
      session.projection.players.p1.protocol.push('|request|PRIVATE_OTHER_SEAT_SENTINEL');
      const external = session.externalInput();
      assert.deepEqual(external.protocol, expected);
      assert.equal(JSON.stringify(external).includes('SENTINEL'), false);
      assert.equal(external.request_id, frame.players.p2.request_id);
      assert.equal(frame.observable_history.p2.side, 'p2');
    } finally { session.close(); }
  }
});

test('submitted evidence binds request-relative command/action before step', async () => {
  const session = new ExternalBenchmarkSession('p2');
  try {
    const frame = await session.start({episodeIndex: 0, runSeed: 102735001, split: 'train', swap: false, tacticalSide: null});
    const actions = {}, request_ids = {}, choices = {};
    for (const side of ['p1', 'p2']) {
      actions[side] = frame.players[side].mask.findIndex(Boolean);
      request_ids[side] = frame.players[side].request_id;
      choices[side] = choiceFor(actions[side], session.projection.players[side].request);
    }
    const next = await submittedStep(session, {actions, request_ids});
    for (const sub of next.submitted) {
      assert.equal(sub.request_id, request_ids[sub.side]);
      assert.equal(sub.action, actions[sub.side]);
      assert.equal(sub.command, choices[sub.side]);
      assert.equal(typeof sub.accepted, 'boolean');
    }
  } finally { session.close(); }
});

test('real hidden Arena Trap rejection refreshes the same game and preserves accepted opponent submission', async () => {
  const session = new ExternalBenchmarkSession('p1');
  const mon = (species, ability, moves) => ({species, ability, moves, level: 50});
  const bench = () => [mon('Squirtle', 'Torrent', ['watergun']), mon('Charmander', 'Blaze', ['ember']),
    mon('Bulbasaur', 'Overgrow', ['vinewhip']), mon('Eevee', 'Run Away', ['tackle']), mon('Magikarp', 'Swift Swim', ['splash'])];
  try {
    // This hand-built engineering fixture never enters official partition data.
    session.started = true; session.split = 'train'; session.swap = false;
    session.episodeId = 'f35-engineering-hidden-trap';
    session.provenance = {engineering_fixture: true};
    await session.stream.write(`>start ${JSON.stringify({formatid: 'gen9anythinggoes@@@!Team Preview', seed: seedFor('f35-hidden-trap-v1')})}\n` +
      `>player p1 ${JSON.stringify({name: 'Full p1', team: [mon('Pikachu', 'Static', ['thunderbolt']), ...bench()]})}\n` +
      `>player p2 ${JSON.stringify({name: 'Full p2', team: [mon('Dugtrio', 'Arena Trap', ['earthquake']), ...bench()]})}`);
    await session.passControls();
    const initial = session.snapshot();
    assert.equal(initial.players.p1.mask[9], true);
    const first = await submittedStep(session, {actions: {p1: 9, p2: 0},
      request_ids: {p1: initial.players.p1.request_id, p2: initial.players.p2.request_id}});
    const rejected = first.submitted.find(s => s.side === 'p1');
    assert.equal(rejected.accepted, false);
    assert.match(rejected.feedback[0], /\[Unavailable choice\].*trapped/);
    assert.equal(rejected.command, 'switch 2');
    assert.equal(first.submitted.find(s => s.side === 'p2').accepted, true);
    assert.equal(first.players.p1.mask[9], false);
    assert(first.players.p1.request_id > initial.players.p1.request_id);
    assert.equal(first.players.p2.mask.some(Boolean), false);
    const refreshed = await submittedStep(session, {actions: {p1: 0}, request_ids: {p1: first.players.p1.request_id}});
    assert.equal(refreshed.submitted.length, 1);
    assert.equal(refreshed.submitted[0].accepted, true);
    assert.equal(refreshed.episode_id, initial.episode_id);
  } finally { session.close(); }
});
