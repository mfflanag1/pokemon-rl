import test from 'node:test';
import assert from 'node:assert/strict';
import {BattleStream, seedFor} from '../../packages/reference-node/engine.mjs';
import {PlayerProjection} from '../../packages/reference-node/observations.mjs';
import {legalChoices, PublicBattle} from '../../packages/showdown-bot/protocol.mjs';
import {chooseHeuristic} from '../../packages/showdown-bot/policy.mjs';
import {applyFixture} from '../../packages/episodes/fixtures.mjs';

const mon = (species, moves = ['tackle'], ability = 'Run Away') => ({species, moves, ability, level: 50});
async function session(own, foe) {
  const projection = new PlayerProjection(), messages = [];
  const stream = new class extends BattleStream {
    constructor() { super({noCatch: true}); }
    pushMessage(type, data) { messages.push({type, data}); projection.ingest(type, data); }
  }();
  await stream.write(`>start ${JSON.stringify({formatid: 'gen9anythinggoes@@@!Team Preview', seed: seedFor('bot-fixtures-v1')})}\n` +
    `>player p1 ${JSON.stringify({name: 'BaselineBot', team: own})}\n` +
    `>player p2 ${JSON.stringify({name: 'OtherPlayer', team: foe})}`);
  return {stream, projection, messages, close: () => stream.battle?.destroy()};
}

test('real hidden trapping rejects a visible switch and refreshes the action set', async () => {
  const s = await session([mon('Pikachu', ['thunderbolt'], 'Static'), mon('Raichu', ['thunderbolt'], 'Static')],
    [mon('Dugtrio', ['earthquake'], 'Arena Trap'), mon('Squirtle', ['watergun'], 'Torrent')]);
  try {
    assert.ok(legalChoices(s.projection.players.p1.request).includes('switch 2'));
    await s.stream.write('>p1 switch 2');
    assert.ok(s.projection.errors.some(e => e.message.startsWith('[Unavailable choice]')));
    assert.ok(s.projection.players.p1.request.active[0].trapped);
    assert.ok(!legalChoices(s.projection.players.p1.request).some(c => c.startsWith('switch ')));
    const before = s.projection.errors.length;
    await s.stream.write('>p1 move 1\n>p2 move 1');
    assert.equal(s.projection.errors.length, before);
  } finally { s.close(); }
});

test('constructed Revival Blessing request accepts the fainted-slot codec choice', async () => {
  const s = await session([mon('Rabsca', ['revivalblessing', 'psychic'], 'Synchronize'), mon('Pikachu')],
    [mon('Squirtle'), mon('Bulbasaur')]);
  try {
    applyFixture(s, 'revival-p1', 'initial');
    const r = s.projection.players.p1.request;
    assert.equal(r.side.pokemon[0].reviving, true);
    assert.deepEqual(legalChoices(r), ['switch 2']);
    await s.stream.write('>p1 switch 2');
    assert.equal(s.projection.errors.length, 0);
    assert.ok(s.stream.battle.p1.pokemon[1].hp > 0);
  } finally { s.close(); }
});

test('unrevealed reserve change produces identical actual player inputs and policy choice', async () => {
  const own = [mon('Squirtle', ['watergun', 'tackle'], 'Torrent'), mon('Bulbasaur')];
  const a = await session(own, [mon('Fennekin', ['ember'], 'Blaze'), mon('Pikachu')]);
  const b = await session(own, [mon('Fennekin', ['ember'], 'Blaze'), mon('Raichu')]);
  try {
    const view = s => {
      const p = s.projection.players.p1, state = new PublicBattle();
      p.protocol.forEach(line => state.ingest(line));
      return {request: p.request, state};
    };
    const one = view(a), two = view(b);
    assert.deepEqual(one, two);
    assert.equal(chooseHeuristic(one.request, one.state), chooseHeuristic(two.request, two.state));
    assert.notEqual(a.stream.battle.p2.pokemon[1].species.id, b.stream.battle.p2.pokemon[1].species.id);
  } finally { a.close(); b.close(); }
});

test('pinned simulator emits public startup/turn updates before their normal requests', async () => {
  const s = await session([mon('Squirtle', ['watergun'], 'Torrent'), mon('Bulbasaur')],
    [mon('Squirtle', ['watergun'], 'Torrent'), mon('Bulbasaur')]);
  try {
    let firstRequest = s.messages.findIndex(m => m.type === 'sideupdate' && m.data.includes('|request|'));
    assert.ok(firstRequest > 0);
    assert.ok(s.messages.slice(0, firstRequest).some(m => m.type === 'update' && m.data.includes('|turn|1')));
    const cursor = s.messages.length;
    await s.stream.write('>p1 move 1\n>p2 move 1');
    const next = s.messages.slice(cursor);
    firstRequest = next.findIndex(m => m.type === 'sideupdate' && m.data.includes('|request|'));
    assert.ok(firstRequest > 0);
    assert.ok(next.slice(0, firstRequest).some(m => m.type === 'update' && m.data.includes('|turn|2')));
  } finally { s.close(); }
});
