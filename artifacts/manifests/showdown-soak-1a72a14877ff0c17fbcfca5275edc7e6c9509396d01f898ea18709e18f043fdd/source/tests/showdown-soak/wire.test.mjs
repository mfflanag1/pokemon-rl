import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {assertJSON, actorInput} from '../../packages/showdown-soak/wire.mjs';

const plugin = (config = {bindaddress: '127.0.0.1', serverid: 'local-baseline'}) => {
  const sandbox = {exports: {}, Config: config, Chat: {ErrorMessage: Error}, Teams: {pack: team => JSON.stringify(team)}};
  vm.runInNewContext(readFileSync(new URL('../../packages/showdown-soak/team-receipt.cjs', import.meta.url), 'utf8'), sandbox);
  return sandbox.exports.commands.soakteam;
};
test('team receipt uses only the authenticated caller and sends privately', async () => {
  const user = {named: true, id: 'alice'}, team = [{species: 'Pikachu'}], replies = [];
  const room = {battle: {getTeam: async actual => { assert.equal(actual, user); return team; }}};
  await plugin().call({sendReply: line => replies.push(line)}, '', room, user);
  assert.deepEqual(replies, ['|soakteam|[{"species":"Pikachu"}]']);
});
test('team receipt denies a target, outsider, unnamed identity or non-local configuration', async () => {
  const room = {battle: {getTeam: async () => null}}, user = {named: true};
  await assert.rejects(plugin().call({}, 'bob', room, user), /Local own-team/);
  await assert.rejects(plugin().call({}, '', room, user), /Only a participant/);
  await assert.rejects(plugin().call({}, '', room, {named: false}), /Local own-team/);
  await assert.rejects(plugin({bindaddress: '0.0.0.0', serverid: 'local-baseline'}).call({}, '', room, user), /Local own-team/);
  await assert.rejects(plugin({bindaddress: '127.0.0.1', serverid: 'public'}).call({}, '', room, user), /Local own-team/);
});
test('actor snapshots reject fields that JSON would silently erase or distort', () => {
  for (const bad of [new Map(), new Set(), new Date(), undefined, NaN, Infinity, 1n, () => 1]) {
    assert.throws(() => assertJSON({nested: {bad}}), /Non-JSON|Non-finite/);
  }
});
test('captured actor inputs are detached from the live room', () => {
  const room = {request: {rqid: 2}, state: {moves: ['tackle'], known: {}}};
  const captured = actorInput(room); room.request.rqid++; room.state.moves.push('growl');
  assert.deepEqual(captured, {request: {rqid: 2}, state: {moves: ['tackle'], known: {}}});
});
