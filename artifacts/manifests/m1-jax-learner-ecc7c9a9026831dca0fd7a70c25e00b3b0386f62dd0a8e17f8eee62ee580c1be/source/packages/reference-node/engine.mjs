import {createRequire} from 'node:module';
import path from 'node:path';
import {ROOT, UPSTREAM, pin, verifyBuild, verifySource, fileHash, hashJSON} from './provenance.mjs';

const receipt = verifyBuild();
const harnessHashes = Object.fromEntries(['provenance.mjs', 'engine.mjs', 'observations.mjs', 'runner.mjs'].map(name => [
  name, fileHash(path.join(ROOT, 'packages/reference-node', name)),
]));
// A private local harness, with no server configuration or Pokemon of the Day.
if (global.Config !== undefined) throw new Error('Reference harness needs a fresh process without server Config');
Object.defineProperty(global, 'Config', {value: Object.freeze({potd: pin.potd}), writable: false});
const require = createRequire(import.meta.url);
export const {BattleStream, getPlayerStreams, Dex, Teams, PRNG} = require(path.join(UPSTREAM, 'dist/sim/index.js'));
export const {extractChannelMessages} = require(path.join(UPSTREAM, 'dist/sim/battle.js'));
export const {RandomPlayerAI} = require(path.join(UPSTREAM, 'dist/sim/tools/random-player-ai.js'));

export function describe() {
  verifySource();
  for (const [name, expected] of Object.entries(harnessHashes)) {
    if (fileHash(path.join(ROOT, 'packages/reference-node', name)) !== expected) throw new Error('Harness source changed; restart the worker');
  }
  const format = Dex.formats.get(pin.format_id);
  const rules = Dex.formats.getRuleTable(format);
  const body = {
    schema_version: 'r1-manifest-1',
    repository: pin.repository,
    commit: pin.commit,
    source_dirty: false,
    lock_sha256: pin.package_lock_sha256,
    build: receipt,
    harness_source_hashes: harnessHashes,
    format: {
      id: format.id, name: format.name, mod: format.mod, game_type: format.gameType,
      team: format.team, ruleset: format.ruleset,
      resolved_rules: [...rules.entries()].sort(([a], [b]) => a < b ? -1 : 1),
      value_rules: [...rules.valueRules.entries()].sort(([a], [b]) => a < b ? -1 : 1),
    },
    configuration: {potd: pin.potd, server_config_loaded: false, custom_formats: false},
    source_hashes: Object.fromEntries([
      'LICENSE', 'config/formats.ts', 'data/rulesets.ts', 'sim/prng.ts',
      'data/random-battles/gen9/teams.ts', 'data/random-battles/gen9/sets.json',
    ].map(name => [name, fileHash(path.join(UPSTREAM, name))])),
    seed_encoding: 'sodium, followed by exactly 64 lower-case hex digits',
    observation_version: 'r1-player-protocol-1',
    normalization_version: 'r1-timestamps-1',
  };
  return {...body, manifest_id: hashJSON(body)};
}

export function seedFor(namespace) {
  if (typeof namespace !== 'string' || !namespace) throw new TypeError('Seed namespace must be a nonempty string');
  return `sodium,${hashJSON(['pokemon-reference-r1', namespace])}`;
}

export function checkSeed(seed) {
  if (typeof seed !== 'string' || !/^sodium,[0-9a-f]{64}$/.test(seed)) throw new Error('Expected an explicit canonical sodium seed');
}

export function generateTeam(seed) {
  checkSeed(seed);
  const team = Teams.generate(pin.format_id, {seed});
  return {seed, team: structuredClone(team), team_hash: hashJSON(team)};
}
