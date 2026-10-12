import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';
import {Teams, Dex} from '../../packages/reference-node/engine.mjs';
import {developmentFormat, teamReceiptLine} from '../../packages/showdown-soak/teams.mjs';
import {formatSource} from '../../packages/showdown-soak/local-server.mjs';

const require = createRequire(import.meta.url);
const {TeamValidator} = require('../../vendor/pokemon-showdown/dist/sim/team-validator.js');
const {Format} = require('../../vendor/pokemon-showdown/dist/sim/dex-formats.js');
const fixture = JSON.parse(readFileSync(new URL('./fixtures/constructed-rejection.json', import.meta.url)));
const admitted = () => Teams.unpack(Teams.pack(fixture.entry.team));
const sandbox = {exports: {}, Teams};
vm.runInNewContext(formatSource([fixture.entry]), sandbox);
const format = new Format({...sandbox.exports.Formats[0], effectType: 'Format'});
const validator = new TeamValidator(format);

test('official generated sets retain their values despite constructed-legality rejection', () => {
  const team = admitted(), before = structuredClone(team);
  const errors = new TeamValidator(new Format({...developmentFormat(), effectType: 'Format'})).validateTeam(structuredClone(team));
  assert.deepEqual(errors, fixture.constructed_errors);
  assert(errors.length > 0);
  assert.equal(validator.validateTeam(team), null);
  assert.deepEqual(team, before);
});
test('the allowlist rejects changed battle values, order and incomplete or unknown teams', () => {
  for (const mutate of [t => t[0].level--, t => t[0].ivs.atk = (t[0].ivs.atk + 1) % 32,
    t => t[0].item = 'Master Ball', t => t[0].moves[0] = 'splash', t => t.reverse(), t => t.pop()]) {
    const team = admitted(); mutate(team); assert(validator.validateTeam(team)?.length);
  }
  assert(validator.validateTeam([])?.length);
  assert(validator.validateTeam(null)?.length);
});
test('local team admission preserves the official resolved battle rules', () => {
  const official = Dex.formats.get('gen9randombattle');
  const actual = Dex.formats.getRuleTable(format), expected = Dex.formats.getRuleTable(official);
  assert.deepEqual([...actual], [...expected]);
  assert.deepEqual([...actual.valueRules], [...expected.valueRules]);
  assert.equal(format.mod, official.mod); assert.equal(format.gameType, official.gameType);
  assert.equal(format.rated, false); assert.equal(format.searchShow, false);
});
test('canonical packing and simulator default nicknames remain stable', () => {
  const team = admitted(), packed = Teams.pack(team);
  assert.equal(Teams.pack(Teams.unpack(packed)), packed);
  const expected = structuredClone(team);
  for (const set of expected) if (!set.name || set.name === set.species) set.name = Dex.species.get(set.species).baseSpecies;
  const receipt = teamReceiptLine(fixture.entry.team).slice('|soakteam|'.length);
  assert.equal(receipt, Teams.pack(expected));
  assert.equal(Teams.pack(Teams.unpack(receipt)), receipt);
});
