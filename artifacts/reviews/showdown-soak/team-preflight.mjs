import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, existsSync} from 'node:fs';
import {createRequire} from 'node:module';
import vm from 'node:vm';
import {freshTeams, developmentFormat, teamReceiptLine} from '../../../packages/showdown-soak/teams.mjs';
import {formatSource} from '../../../packages/showdown-soak/local-server.mjs';
import {Teams} from '../../../packages/reference-node/engine.mjs';
const require = createRequire(import.meta.url);
const {TeamValidator} = require('../../../vendor/pokemon-showdown/dist/sim/team-validator.js');
const {Format} = require('../../../vendor/pokemon-showdown/dist/sim/dex-formats.js');
const teams = freshTeams(false), sandbox = {exports: {}, Teams};
vm.runInNewContext(formatSource(teams), sandbox);
const validator = new TeamValidator(new Format({...sandbox.exports.Formats[0], effectType: 'Format'}));
const constructed = new TeamValidator(new Format({...developmentFormat(), effectType: 'Format'}));
let rejected = 0; const examples = [];
for (const entry of teams) {
  const team = Teams.unpack(Teams.pack(entry.team)), before = structuredClone(team);
  assert.equal(validator.validateTeam(team), null); assert.deepEqual(team, before);
  const packed = Teams.pack(team);
  assert.equal(Teams.pack(Teams.unpack(packed)), packed);
  const receipt = teamReceiptLine(entry.team).slice('|soakteam|'.length);
  assert.equal(Teams.pack(Teams.unpack(receipt)), receipt);
  const errors = constructed.validateTeam(structuredClone(team));
  if (errors) {
    rejected++; if (examples.length < 5) examples.push({team_key: entry.team_key, errors});
    const fixture = 'tests/showdown-soak/fixtures/constructed-rejection.json';
    if (!existsSync(fixture)) writeFileSync(fixture, JSON.stringify({entry, constructed_errors: errors}, null, 2) + '\n');
  }
}
const report = {status: 'pass', teams: teams.length, exact_allowlist_accepts: teams.length,
  validation_mutations: 0, canonical_pack_roundtrips: teams.length, receipt_roundtrips: teams.length,
  constructed_validator_rejections: rejected, examples,
  note: 'The first pack/unpack canonicalizes move display names. Repeated canonical serialization is stable. No official set values changed and no teams replaced.'};
writeFileSync('artifacts/reviews/showdown-soak/production-team-normalization.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
