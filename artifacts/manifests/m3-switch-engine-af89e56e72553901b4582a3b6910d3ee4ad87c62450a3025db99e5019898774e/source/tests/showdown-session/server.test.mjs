import test from 'node:test';
import assert from 'node:assert/strict';
import {BattleStream, seedFor} from '../../packages/reference-node/engine.mjs';
import {legalChoices} from '../../packages/showdown-bot/protocol.mjs';

test('pinned server clears a hidden-trapping rejection before echoing the refreshed reconnect request', async () => {
  // Establish the oracle's frozen empty-PotD configuration before importing server methods.
  const {default: server} = await import('../../vendor/pokemon-showdown/dist/server/room-battle.js');
  const messages = [], replay = [];
  const player = id => ({active: true, request: {rqid: 0, request: '', isWait: false, choice: ''},
    slot: id, updateChannel() {}, sendRoom(data) { messages.push({id, data}); }});
  const p1 = player('p1'), p2 = player('p2');
  const host = {p1, p2, players: [p1, p2], playerTable: {recoverbot: p1}, rqid: 0, requestCount: 0,
    roomid: 'battle-fixture-1', timer: {nextRequest() {}}, started: true, ended: false};
  const stream = new class extends BattleStream {
    constructor() { super({noCatch: true}); }
    pushMessage(type, data) {
      if (type === 'sideupdate') server.RoomBattle.prototype.receive.call(host, [type, ...data.split('\n')]);
    }
  }();
  let pending;
  host.stream = {write(data) { pending = stream.write(data); return pending; }};
  const mon = (species, ability, moves) => ({species, ability, moves, level: 50});
  try {
    await stream.write(`>start ${JSON.stringify({formatid: 'gen9anythinggoes@@@!Team Preview', seed: seedFor('session-trapping-v1')})}\n` +
      `>player p1 ${JSON.stringify({name: 'RecoverBot', team: [mon('Pikachu', 'Static', ['thunderbolt']), mon('Squirtle', 'Torrent', ['watergun'])]})}\n` +
      `>player p2 ${JSON.stringify({name: 'Other', team: [mon('Dugtrio', 'Arena Trap', ['earthquake']), mon('Bulbasaur', 'Overgrow', ['vinewhip'])]})}`);
    const original = JSON.parse(p1.request.request); assert(legalChoices(original).includes('switch 2'));
    server.RoomBattle.prototype.choose.call(host, {id: 'recoverbot'}, `switch 2|${p1.request.rqid}`); await pending;
    const error = messages.findIndex(m => m.data.startsWith('|error|[Unavailable choice]'));
    assert(error >= 0); assert(messages[error + 1].data.startsWith('|request|'));
    const refreshed = JSON.parse(p1.request.request); assert(refreshed.active[0].trapped);
    assert(refreshed.rqid > original.rqid); assert.equal(p1.request.choice, ''); assert.equal(p1.request.isWait, false);
    server.RoomBattle.prototype.onConnect.call(host, {id: 'recoverbot', sendTo(room, data) { replay.push({room, data}); }});
    assert.equal(replay.length, 1); assert.equal(replay[0].data, '|request|' + p1.request.request);
    assert(!replay[0].data.includes('|sentchoice|'));
    assert(legalChoices(refreshed).every(c => !c.startsWith('switch ')));
  } finally { stream.battle?.destroy(); }
});
