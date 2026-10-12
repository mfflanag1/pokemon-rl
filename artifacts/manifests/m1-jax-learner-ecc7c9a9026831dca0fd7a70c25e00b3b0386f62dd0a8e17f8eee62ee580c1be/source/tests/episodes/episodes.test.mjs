import test from 'node:test';
import assert from 'node:assert/strict';
import {canonical, hashJSON, parseJSON} from '../../packages/battle-contract/canonical.mjs';
import {defaultSpec, runEpisode} from '../../packages/reference-node/runner.mjs';
import {EpisodeSession, runPhaseEpisode, replayPhaseEpisode} from '../../packages/episodes/session.mjs';
import {episodeManifest} from '../../packages/episodes/manifest.mjs';
import {validateEpisode} from '../../packages/episodes/contract.mjs';

const spec = defaultSpec('r1-smoke-8');
const submit = (session, side, id, action) => {
  const view = session.view(side);
  return {submission_id: id, side, request_id: view.request_id, view_id: view.view_id, action: action ?? view.admissible_actions[0]};
};
function rehash(bundle) {
  const p = bundle.episode.payload;
  for (const d of p.decisions) {
    const {decision_id, ...body} = d;
    d.decision_id = hashJSON(['r4-decision-1', p.episode_key, body]);
  }
  p.transitions.forEach((t, i) => {
    t.decision_id = p.decisions[i].decision_id;
    if (t.target.kind === 'decision') t.target.id = p.decisions[t.target.sequence].decision_id;
  });
  const {content_hash, artifact_id, ...body} = bundle.episode;
  bundle.episode.content_hash = hashJSON(body);
  bundle.episode.artifact_id = `phase_episode:${bundle.episode.content_hash}`;
  return bundle;
}

test('complete R4 episode preserves R1 visible mechanics and replays after canonical serialization', async () => {
  const a = await runPhaseEpisode(spec), old = await runEpisode(spec), p = a.episode.payload;
  assert.deepEqual(p.protocol, {p1: old.protocol.p1, p2: old.protocol.p2});
  assert.equal(p.decisions.length, old.decisions.length);
  assert.equal(p.outcome.turn, old.outcome.turns);
  assert.equal(p.controller, spec.policy_id);
  for (const d of p.decisions) assert.equal(d.view.protocol_prefix_hash, hashJSON(p.protocol[d.side].slice(0, d.view.protocol_end)));
  const replay = await replayPhaseEpisode(parseJSON(canonical(a)));
  assert.equal(replay.episode.content_hash, a.episode.content_hash);
  assert.ok(p.frames.some(f => f.phase === 'switch_continuation'));
  assert.ok(p.frames.some(f => f.phase === 'single_replacement'));
  for (const [i, t] of p.transitions.entries()) if (t.target.kind === 'decision') {
    assert.equal(p.decisions[t.target.sequence].side, p.decisions[i].side);
    assert.ok(t.target.sequence > i);
    assert.equal(t.reward_numerator, 0);
  }
});

test('natural hidden re-requests preserve the already chosen opponent action and zero-reward linkage', async () => {
  const a = await runPhaseEpisode(spec, {order: 'p2-first'}), p = a.episode.payload;
  const rejected = p.decisions.filter(d => !d.accepted);
  assert.equal(rejected.length, 2);
  for (const d of rejected) {
    const before = p.frames[d.sequence], after = p.frames[d.sequence + 1], t = p.transitions[d.sequence];
    assert.equal(d.side, 'p1');
    assert.notEqual(before.commitments.p2, null);
    assert.equal(after.commitments.p2, before.commitments.p2);
    assert.equal(after.audit.choices.p2, before.audit.choices.p2);
    assert.equal(after.audit.mechanics_hash, before.audit.mechanics_hash);
    assert.equal(after.group, before.group);
    assert.equal(after.requests.p1.request_id, before.requests.p1.request_id + 1);
    assert.equal(t.target.kind, 'decision');
    assert.equal(t.reward_numerator, 0);
    assert.equal(t.terminal, false);
  }
  assert.equal((await replayPhaseEpisode(a)).episode.content_hash, a.episode.content_hash);
});

test('budget reached on a hidden rejection waits for a fresh group; IDs distinguish order and action content', async () => {
  const full = await runPhaseEpisode(spec), rejected = full.episode.payload.decisions.find(d => !d.accepted);
  assert.ok(rejected);
  const cut = await runPhaseEpisode(spec, {truncateAfter: rejected.sequence + 1});
  assert.equal(cut.episode.payload.frames.at(-1).event, 'resolved');
  assert.ok(cut.episode.payload.decisions.length > rejected.sequence + 1);
  assert.equal((await replayPhaseEpisode(cut)).episode.content_hash, cut.episode.content_hash);
  for (const phase of ['switch_continuation', 'single_replacement']) {
    const boundary = full.episode.payload.frames.findIndex(f => f.phase === phase);
    assert.ok(boundary > 0);
    const overflow = await runPhaseEpisode(spec, {truncateAfter: boundary});
    assert.ok(overflow.episode.payload.decisions.length > boundary);
    assert.equal(overflow.episode.payload.frames.at(-1).phase, 'joint_move');
    assert.equal(overflow.episode.payload.frames.at(-1).event, 'resolved');
    assert.equal((await replayPhaseEpisode(overflow)).episode.content_hash, overflow.episode.content_hash);
  }
  const a = await runPhaseEpisode(spec, {truncateAfter: 2, order: 'p1-first'});
  const b = await runPhaseEpisode(spec, {truncateAfter: 2, order: 'p2-first'});
  assert.notEqual(a.episode.payload.episode_key, b.episode.payload.episode_key);
  assert.notEqual(a.episode.payload.decisions[0].decision_id, b.episode.payload.decisions[0].decision_id);
  const sessions = await Promise.all([new EpisodeSession(spec).start(), new EpisodeSession(spec).start()]);
  try {
    const choices = sessions[0].view('p1').admissible_actions;
    const first = await sessions[0].submit(submit(sessions[0], 'p1', 'same-id', choices[0]));
    const second = await sessions[1].submit(submit(sessions[1], 'p1', 'same-id', choices[1]));
    assert.equal(sessions[0].key, sessions[1].key);
    assert.notEqual(first.decision_id, second.decision_id);
  } finally { sessions.forEach(s => s.close()); }
});

test('concurrent submissions serialize, exact retries are idempotent, stale or invalid choices fail before writes', async () => {
  const session = await new EpisodeSession(defaultSpec('r4-transport')).start();
  try {
    const p1 = submit(session, 'p1', 'one'), p2 = submit(session, 'p2', 'two');
    const results = await Promise.all([session.submit(p1), session.submit(p2)]);
    assert.equal(session.decisions.length, 2);
    assert.deepEqual(results.map(r => r.sequence), [0, 1]);
    const writes = session.reference.input_tape.length;
    assert.deepEqual(await session.submit(p1), results[0]);
    assert.equal(session.reference.input_tape.length, writes);
    await assert.rejects(session.submit({...p1, action: {kind: 'switch', slot: 0}}), /reused/);
    await assert.rejects(session.submit({...p1, submission_id: 'stale'}), /Stale|inactive/);
    if (session.frames.at(-1).pending_sides.includes('p1')) {
      await assert.rejects(session.submit(submit(session, 'p1', 'invalid', {kind: 'switch', slot: 0})), /forbidden/);
    }
    assert.equal(session.reference.input_tape.length, writes);
  } finally { session.close(); }
});

test('actor view does not change when only the opponent has submitted its hidden choice', async () => {
  const session = await new EpisodeSession(defaultSpec('r4-private-commit')).start();
  try {
    const before = session.view('p2');
    await session.submit(submit(session, 'p1', 'p1-choice'));
    assert.deepEqual(session.view('p2'), before);
    assert.notEqual(session.frames.at(-1).commitments.p1, null);
    assert.throws(() => session.truncate(), /fresh/);
  } finally { session.close(); }
});

test('constructed single and double faint replacements enter, retain commitments, and exit correctly', async () => {
  for (const fixture of ['single-faint-p1', 'single-faint-p2', 'double-faint']) {
    const a = await runPhaseEpisode(spec, {fixture, truncateAfter: 1}), p = a.episode.payload;
    assert.equal(p.frames[0].phase, fixture === 'double-faint' ? 'double_replacement' : 'single_replacement');
    assert.equal(p.frames.at(-1).phase, 'joint_move');
    assert.equal(p.outcome.status, 'truncated');
    if (fixture === 'double-faint') {
      assert.equal(p.decisions.length, 2);
      assert.equal(p.frames[1].group, 0);
      assert.equal(p.frames[1].commitments.p1, 0);
      assert.deepEqual(p.frames[1].pending_sides, ['p2']);
    }
    assert.equal((await replayPhaseEpisode(a)).episode.content_hash, a.episode.content_hash);
    assert.throws(() => validateEpisode(a, episodeManifest('official')), /manifest/);
  }
});

test('a constructed reference tie cancels the unsubmitted request and yields terminal half reward', async () => {
  const a = await runPhaseEpisode(spec, {fixture: 'tie-after-first-choice'}), p = a.episode.payload;
  assert.equal(p.decisions.length, 1);
  assert.equal(p.outcome.status, 'terminal');
  assert.equal(p.outcome.reason, 'tie');
  assert.deepEqual(p.frames.at(-1).pending_sides, []);
  assert.equal(p.transitions[0].reward_numerator, 1);
  assert.equal(p.transitions[0].reward_denominator, 2);
  assert.equal(p.transitions[0].terminal, true);
  assert.equal((await replayPhaseEpisode(a)).episode.content_hash, a.episode.content_hash);
});

test('constructed live continuations, mixed requests and revival targets cover both player roles', async () => {
  for (const fixture of ['live-switch-p1', 'live-switch-p2', 'mixed-switch', 'revival-p1', 'revival-p2']) {
    const a = await runPhaseEpisode(spec, {fixture, truncateAfter: 1}), p = a.episode.payload;
    assert.equal(p.frames[0].phase, fixture.startsWith('revival') ? 'revival' : fixture === 'mixed-switch' ? 'mixed_switch' : 'switch_continuation');
    if (fixture.startsWith('revival')) {
      const side = fixture.endsWith('p1') ? 'p1' : 'p2';
      assert.equal(p.decisions[0].side, side);
      assert.deepEqual(p.decisions[0].action, {kind: 'switch', slot: 1});
      assert.equal(p.frames[0].audit.mechanics.sides.find(s => s.id === side).pokemon_left, 5);
      assert.equal(p.frames[1].audit.mechanics.sides.find(s => s.id === side).pokemon_left, 6);
    }
    assert.equal((await replayPhaseEpisode(a)).episode.content_hash, a.episode.content_hash);
  }
});

test('explicit truncation finishes the current group and bootstraps each same-player trajectory', async () => {
  const a = await runPhaseEpisode(spec, {truncateAfter: 1}), p = a.episode.payload;
  assert.ok(p.decisions.length >= 2);
  assert.equal(p.outcome.status, 'truncated');
  assert.equal(p.outcome.winner, null);
  assert.deepEqual(p.frames.at(-1).pending_sides, ['p1', 'p2']);
  for (const t of p.transitions) {
    assert.equal(t.terminal, false);
    assert.equal(t.reward_numerator, 0);
  }
  for (const side of ['p1', 'p2']) assert.equal(p.transitions.filter(t => t.side === side).at(-1).target.kind, 'bootstrap');
  assert.equal((await replayPhaseEpisode(a)).episode.content_hash, a.episode.content_hash);
  await assert.rejects(runPhaseEpisode({...spec, max_submissions: 1}), /safety cap/);
});

test('semantic corruption remains rejected after artifact hashes are recomputed', async () => {
  const a = await runPhaseEpisode(spec, {truncateAfter: 2}), manifest = episodeManifest();
  for (const mutate of [
    p => { p.transitions[0].side = 'p2'; },
    p => { p.transitions[0].reward_numerator = 2; },
    p => { p.frames[0].phase = 'double_replacement'; },
    p => { p.decisions[0].command = 'switch 1'; },
    p => { p.input_tape[0] += '\n>eval 1'; },
    p => { p.decisions[0].view.protocol_delta[0] = '|message|future information'; },
  ]) {
    const b = structuredClone(a); mutate(b.episode.payload);
    assert.throws(() => validateEpisode(rehash(b), manifest));
  }
});
