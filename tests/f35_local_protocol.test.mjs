import test from 'node:test';
import assert from 'node:assert/strict';
import {TeacherAssistedSession} from '../packages/teacher-assisted/worker.mjs';
import {StructuredSession} from '../packages/structured-policy/session.mjs';
import {createActor} from '../packages/showdown-panel/policy.mjs';
import {ACTIONS} from '../packages/full-showdown/actions.mjs';

const options = {episodeIndex: 0, runSeed: 35001, split: 'development', swap: false};

test('F35 adds coordinator-only baseline advice with unchanged actor projections', async () => {
  const actual = new TeacherAssistedSession(), original = new StructuredSession();
  try {
    let frame = await actual.start({...options, baselineSide: 'p2'});
    let previous = await original.start(options);
    assert.deepEqual(frame.players, previous.players);
    assert.deepEqual(frame.observable_history, previous.observable_history);
    assert.deepEqual(frame.provenance, previous.provenance);
    const independent = createActor('baseline-v1');
    for (const line of actual.projection.players.p2.protocol) {
      if (line && !line.startsWith('|request|')) independent.state.ingest(line);
    }
    const expected = independent.choose(structuredClone(actual.projection.players.p2.request), independent.state);
    assert.equal(frame.baseline_actions.p2, ACTIONS.indexOf(expected));
    assert.equal(frame.baseline_actions.p1, -1);
    assert.ok(frame.players.p2.mask[frame.baseline_actions.p2]);
    const command = {actions: {p1: frame.players.p1.mask.findIndex(Boolean), p2: frame.baseline_actions.p2},
      request_ids: {p1: frame.players.p1.request_id, p2: frame.players.p2.request_id}};
    frame = await actual.step(command); previous = await original.step(command);
    assert.deepEqual(frame.players, previous.players);
    assert.deepEqual(frame.observable_history, previous.observable_history);
    assert.deepEqual(frame.submitted, previous.submitted);
  } finally { actual.close(); original.close(); }
});

test('baseline training admission fails before team generation', async () => {
  const session = new TeacherAssistedSession();
  try {
    session.planFor = () => { throw new Error('Must not generate teams'); };
    await assert.rejects(session.start({...options, split: 'train', baselineSide: 'p1'}), /development-only/);
    assert.equal(session.started, false);
  } finally { session.close(); }
});

test('baseline rejects competing adviser and invalid seat', async () => {
  for (const extra of [{baselineSide: 'spectator'}, {baselineSide: 'p1', tacticalSide: 'p2'}]) {
    const session = new TeacherAssistedSession();
    try { await assert.rejects(session.start({...options, ...extra})); assert.equal(session.started, false); }
    finally { session.close(); }
  }
});

test('explicit unavailable-choice engineering fixture refreshes request and accepts next legal action', async () => {
  const session = new TeacherAssistedSession();
  try {
    let frame = await session.start({...options, split: 'train'});
    const advertisedSwitch = frame.players.p1.mask.findIndex((legal, action) => legal && action >= 8);
    assert.ok(advertisedSwitch >= 8);
    const oldRequest = frame.players.p1.request_id;
    // Test-only intervention after advertisement produces the real reference
    // unavailable-choice response; no production session exposes this seam.
    session.stream.battle.p1.active[0].trapped = true;
    frame = await session.step({actions: {p1: advertisedSwitch, p2: frame.players.p2.mask.findIndex(Boolean)},
      request_ids: {p1: oldRequest, p2: frame.players.p2.request_id}});
    assert.equal(frame.submitted[0].accepted, false);
    assert.match(frame.submitted[0].feedback[0], /^\[Unavailable choice\].*trapped/);
    assert.equal(frame.submitted[1].accepted, true);
    assert.ok(frame.players.p1.request_id > oldRequest);
    assert.equal(frame.players.p1.mask[advertisedSwitch], false);
    assert.equal(frame.players.p2.mask.some(Boolean), false);
    assert.equal(frame.submissions, 2);
    frame = await session.step({actions: {p1: frame.players.p1.mask.findIndex(Boolean)},
      request_ids: {p1: frame.players.p1.request_id}});
    assert.equal(frame.submitted.length, 1);
    assert.equal(frame.submitted[0].accepted, true);
    assert.equal(frame.submissions, 3);
    assert.ok(frame.players.p1.turn > 1);
    const cutoff = session.cutoff();
    assert.equal(cutoff.outcome.kind, 'administrative_cutoff');
    assert.deepEqual(cutoff.baseline_actions, {p1: -1, p2: -1});
  } finally { session.close(); }
});
