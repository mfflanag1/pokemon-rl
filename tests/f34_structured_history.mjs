import test from 'node:test';
import assert from 'node:assert/strict';
import {PublicHistory, HISTORY_CONTRACT} from '../packages/structured-policy/history.mjs';
import {PlayerProjection} from '../packages/reference-node/observations.mjs';
import {StructuredSession} from '../packages/structured-policy/session.mjs';
import {OpponentMixSession} from '../packages/opponent-mix/session.mjs';

const kind = name => HISTORY_CONTRACT.event_kind.indexOf(name);

test('split privacy, private requests and end payload never cross history boundary', () => {
  const p = new PlayerProjection();
  p.ingest('update', '|switch|p1a: Pikachu|Pikachu, L80|100/100\n|switch|p2a: Charizard|Charizard, L80|100/100\n|turn|1\n|split|p1\n|-damage|p1a: Pikachu|73/211\n|-damage|p1a: Pikachu|35/100');
  p.ingest('sideupdate', 'p1\n|request|{"side":{"id":"p1","secret":"private-own"}}');
  p.ingest('end', '{"teams":["secret-opponent-team"],"winner":"Full p1"}');
  const a = new PublicHistory('p1').consume(p.players.p1.protocol);
  const b = new PublicHistory('p2').consume(p.players.p2.protocol);
  assert.equal(a.events.at(-1).numeric[1], 73 / 211);
  assert.equal(b.events.at(-1).numeric[1], .35);
  assert.equal(a.events.at(-1).ids[1], 1);
  assert.equal(b.events.at(-1).ids[1], 2);
  assert.equal(a.total_events, 4);
  assert.doesNotMatch(JSON.stringify([a, b]), /secret|request|73\/211|private-own/);
  assert.throws(() => new PublicHistory('p1').consume(['|split|p1']), /Unsplit/);
});

test('bounded ordered history keeps durable move executions and never invents PP', () => {
  const p = new PublicHistory('p1');
  const lines = ['|switch|p2a: disguise|Pikachu, L80|100/100', '|turn|1',
    '|move|p2a: disguise|Sleep Talk|p2a: disguise',
    '|move|p2a: disguise|Thunderbolt|p1a: own|[from]move: Sleep Talk',
    '|-ability|p1a: own|Pressure'];
  for (let turn = 2; turn <= 35; turn++) lines.push(`|turn|${turn}`);
  const first = p.consume(lines);
  assert.equal(first.events.length, 32);
  assert.deepEqual(first.events.map(e => e.numeric[0]), Array.from({length: 32}, (_, i) => i + 4));
  assert.equal(first.public_move_counts[1].reduce((sum, row) => sum + row[1], 0), 2);
  assert.deepEqual(p.consume(lines), first);
  lines.push('|replace|p2a: Zoroark|Zoroark, L80|100/100', '|move|p2a: Zoroark|Thunderbolt|p1a: own');
  const second = p.consume(lines);
  assert.equal(second.public_move_counts[1].reduce((sum, row) => sum + row[1], 0), 3);
  assert.equal(second.events.at(-1).ids[0], kind('move'));
  first.events[0].ids[0] = 999; // Snapshots cannot mutate the retained state.
  assert.notEqual(p.snapshot().events[0].ids[0], 999);
  assert.throws(() => p.consume([]), /rewind/);
  assert.deepEqual(new PublicHistory('p1').snapshot().public_move_counts, [[], []]);
});

test('unsupported categories are unknown and unknown text is never retained', () => {
  const h = new PublicHistory('p2').consume(['|switch|p1a: secret nickname|Unlisted Species, L80|100/100',
    '|move|p1a: secret nickname|Unlisted Move|p2a: other', '|message|secret arbitrary text']);
  assert.equal(h.events[0].ids[4], 0);
  assert.equal(h.events[1].ids[3], 0);
  assert.deepEqual(h.public_move_counts, [[], [[0, 1]]]);
  assert.doesNotMatch(JSON.stringify(h), /secret|Unlisted/);
});

test('event sides use the documented relative-seat enum', () => {
  const h = new PublicHistory('p1').consume(['|turn|1', '|move|p1a: own|Tackle|p2a: foe',
    '|-damage|p2a: foe|80/100', '|move|p1a: own|Splash|[notarget]']);
  assert.deepEqual(HISTORY_CONTRACT.event_side, ['<unknown>', 'own', 'opponent']);
  assert.deepEqual(h.events.map(event => event.ids.slice(1, 3)), [[0, 0], [1, 2], [2, 0], [1, 0]]);
});

test('additive session preserves frozen players and fresh cutoff bootstrap history', async () => {
  const a = new StructuredSession(), b = new OpponentMixSession();
  try {
    let x = await a.start({episodeIndex: 3, runSeed: 1025262699, split: 'train', tacticalSide: 'p2'});
    let y = await b.start({episodeIndex: 3, runSeed: 1025262699, split: 'train', tacticalSide: 'p2'});
    const opening = structuredClone(x.observable_history);
    for (let i = 0; i < 8 && !x.outcome; i++) {
      assert.deepEqual(x.players, y.players);
      assert.deepEqual(x.tactical_actions, y.tactical_actions);
      assert.deepEqual(a.snapshot().observable_history, x.observable_history);
      const actions = {}, request_ids = {};
      for (const side of ['p1', 'p2']) if (x.players[side].mask.some(Boolean)) {
        actions[side] = side === 'p2' ? x.tactical_actions[side] : x.players[side].mask.indexOf(true);
        request_ids[side] = x.players[side].request_id;
      }
      x = await a.step({actions, request_ids}); y = await b.step({actions, request_ids});
      for (const side of ['p1', 'p2']) {
        assert.ok(x.observable_history[side].events.every(event => event.numeric[0] <= x.players[side].turn));
      }
    }
    assert.ok(x.observable_history.p1.total_events > opening.p1.total_events);
    while (!x.outcome && !['p1', 'p2'].every(s => x.players[s].phase === 'move' && x.players[s].mask.some(Boolean))) {
      const actions = {}, request_ids = {};
      for (const side of ['p1', 'p2']) if (x.players[side].mask.some(Boolean)) {
        actions[side] = x.players[side].mask.indexOf(true); request_ids[side] = x.players[side].request_id;
      }
      x = await a.step({actions, request_ids});
    }
    assert.equal(x.outcome, null);
    const cutoff = a.cutoff();
    assert.equal(cutoff.outcome.kind, 'administrative_cutoff');
    assert.deepEqual(cutoff.players, x.players);
    assert.deepEqual(cutoff.observable_history, x.observable_history);
    assert.ok(cutoff.players.p1.mask.some(Boolean));
  } finally { a.close(); b.close(); }
});

test('terminal history includes final observed events and reset starts empty', async () => {
  const a = new StructuredSession();
  try {
    const start = await a.start({episodeIndex: 0, runSeed: 19, split: 'train'});
    for (const side of ['p1', 'p2']) {
      const active = start.players[side].own.find(mon => mon.active);
      for (const viewer of ['p1', 'p2']) a.projection.receive(viewer, `|faint|${side}a: ${active.ident}`);
    }
    a.projection.ingest('end', '{"teams":["forbidden terminal payload"]}');
    a.outcome = {kind: 'terminal', winner: null, turns: start.players.p1.turn, rewards: {p1: .5, p2: .5}};
    const terminal = a.snapshot();
    for (const side of ['p1', 'p2']) {
      assert.ok(terminal.players[side].mask.every(value => !value));
      assert.equal(terminal.observable_history[side].events.at(-1).ids[0], kind('faint'));
      assert.doesNotMatch(JSON.stringify(terminal.observable_history[side]), /forbidden/);
    }
    const fresh = new StructuredSession();
    try { assert.equal(fresh.histories.p1.snapshot().total_events, 0); }
    finally { fresh.close(); }
  } finally { a.close(); }
});
