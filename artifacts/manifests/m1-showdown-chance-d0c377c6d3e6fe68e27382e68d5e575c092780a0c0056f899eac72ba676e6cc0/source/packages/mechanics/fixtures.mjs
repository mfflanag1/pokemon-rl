import assert from 'node:assert/strict';
import {BattleStream, seedFor} from '../reference-node/engine.mjs';
import {hashJSON} from '../reference-node/provenance.mjs';

const mon = (species, ability, moves, item = '', teraType = 'Normal') => ({species, name: species, ability, moves, item,
  teraType, level: 100, nature: 'Serious', evs: {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0},
  ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31}});
const idle = () => mon('Blissey', 'Natural Cure', ['Splash']);

export const SCENARIOS = [
  {id: 'sleep-talk', teams: [[mon('Snorlax', 'Thick Fat', ['Sleep Talk', 'Tackle'])], [mon('Breloom', 'Technician', ['Spore'])]],
    turns: [['move 1', 'move 1']], entries: ['moves:sleeptalk', 'moves:tackle', 'conditions:slp'], families: ['move-calling', 'conditions'],
    check: (s, log) => { assert.equal(s[0][0].status, 'slp'); assert.ok(log.some(l => l.includes('|Tackle|') && l.includes('[from] move: Sleep Talk'))); }},
  {id: 'imposter', teams: [[mon('Ditto', 'Imposter', ['Transform'])], [mon('Blissey', 'Natural Cure', ['Tackle', 'Soft-Boiled'])]],
    turns: [], entries: ['abilities:imposter', 'moves:transform'], families: ['identity-copy', 'move-calling'],
    check: s => { const [a, b] = s.map(t => t[0]); assert.equal(a.species, 'blissey'); assert.equal(a.transformed, true);
      assert.equal(a.ability, 'naturalcure'); assert.deepEqual(a.moves.map(m => m.id), b.moves.map(m => m.id));
      assert.ok(a.moves.every(m => m.pp === 5)); assert.equal(a.maxhp, 237); assert.equal(a.level, 100); }},
  {id: 'magic-bounce', teams: [[mon('Smeargle', 'Own Tempo', ['Spore'])], [mon('Espeon', 'Magic Bounce', ['Splash'])]],
    turns: [['move 1', 'move 1']], entries: ['abilities:magicbounce', 'moves:spore'], families: ['move-calling', 'callback-dispatch'],
    check: (s, log) => { assert.equal(s[0][0].status, 'slp'); assert.equal(s[1][0].status, ''); assert.ok(log.some(l => l.includes('[from] ability: Magic Bounce'))); }},
  {id: 'dancer', teams: [[mon('Butterfree', 'Compound Eyes', ['Quiver Dance'])], [mon('Oricorio', 'Dancer', ['Splash'])]],
    turns: [['move 1', 'move 1']], entries: ['abilities:dancer', 'moves:quiverdance'], families: ['move-calling', 'callback-dispatch'],
    check: (s, log) => { for (const team of s) for (const stat of ['spa', 'spd', 'spe']) assert.equal(team[0].boosts[stat], 1);
      assert.ok(log.some(l => l.includes('[from] ability: Dancer'))); }},
  {id: 'trace', teams: [[mon('Porygon2', 'Trace', ['Splash'])], [mon('Gyarados', 'Intimidate', ['Splash'])]],
    turns: [], entries: ['abilities:trace', 'abilities:intimidate'], families: ['identity-copy', 'ability-events'],
    check: s => { assert.equal(s[0][0].ability, 'intimidate'); assert.equal(s[0][0].base_ability, 'trace'); }},
  {id: 'ogerpon-tera', teams: [[mon('Ogerpon-Wellspring', 'Water Absorb', ['Protect'], 'Wellspring Mask', 'Water')], [idle()]],
    turns: [['move 1 terastallize', 'move 1']], entries: ['forms:ogerponwellspringtera', 'abilities:embodyaspectwellspring', 'items:wellspringmask'], families: ['tera', 'species-state'],
    check: s => { assert.equal(s[0][0].species, 'ogerponwellspringtera'); assert.equal(s[0][0].ability, 'embodyaspectwellspring'); assert.equal(s[0][0].tera, 'Water'); }},
  {id: 'recycle', teams: [[mon('Snorlax', 'Thick Fat', ['Rest', 'Recycle'], 'Chesto Berry')], [mon('Pikachu', 'Static', ['Tackle'])]],
    turns: [['move 1', 'move 1'], ['move 2', 'move 1']], entries: ['moves:recycle', 'moves:rest', 'items:chestoberry'], families: ['item-events', 'conditions'],
    check: (s, log) => { assert.equal(s[0][0].item, 'chestoberry'); assert.equal(s[0][0].status, '');
      assert.ok(log.some(l => l.includes('|-enditem|') && l.includes('Chesto Berry')));
      assert.ok(log.some(l => l.includes('|-item|') && l.includes('[from] move: Recycle'))); }},
  {id: 'zero-to-hero', teams: [[mon('Palafin', 'Zero to Hero', ['Flip Turn']), idle()], [idle()]],
    turns: [['move 1', 'move 1'], ['switch 2', null], ['switch 2', 'move 1']], entries: ['abilities:zerotohero', 'forms:palafinhero', 'moves:flipturn'], families: ['species-state', 'battle-core'],
    check: s => { const p = s[0].find(p => p.base_species === 'palafinhero'); assert.ok(p); assert.equal(p.species, 'palafinhero'); assert.equal(p.active, true); }},
];

class CaptureStream extends BattleStream {
  constructor() { super({noCatch: true, keepAlive: true}); this.messages = []; }
  push(message) { this.messages.push(message); }
}
function snapshot(battle) {
  return battle.sides.map(side => side.pokemon.map(p => ({species: p.species.id, base_species: p.baseSpecies.id,
    hp: p.hp, maxhp: p.maxhp, level: p.level, status: p.status, item: p.item, last_item: p.lastItem,
    ability: p.ability, base_ability: p.baseAbility, transformed: p.transformed, active: p.isActive,
    tera: p.terastallized ?? null, boosts: {...p.boosts}, moves: p.moveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp})),
    volatiles: Object.keys(p.volatiles).sort()})));
}

export async function runScenario(scenario, swap = false) {
  const seed = seedFor(`r6-fixture-v1/${scenario.id}/${swap ? 'reverse' : 'forward'}`);
  const teams = swap ? [...scenario.teams].reverse() : scenario.teams;
  const tape = [`>start ${JSON.stringify({formatid: 'gen9randombattle', seed})}`,
    ...teams.map((team, i) => `>player p${i + 1} ${JSON.stringify({name: `P${i + 1}`, team})}`)];
  for (const actions of scenario.turns) {
    const choices = swap ? [...actions].reverse() : actions;
    tape.push(choices.flatMap((a, i) => a ? [`>p${i + 1} ${a}`] : []).join('\n'));
  }
  const stream = new CaptureStream();
  try {
    const checkpoints = [];
    for (const command of tape) {
      await stream.write(command);
      if (stream.battle?.started) checkpoints.push({turn: stream.battle.turn, request: stream.battle.requestState,
        seed: stream.battle.prng.getSeed(), state: snapshot(stream.battle)});
    }
    const protocol = stream.messages.flatMap(m => m.split('\n')).map(l => /^\|t:\|/.test(l) ? '|t:|<normalized>' : l);
    assert.ok(!protocol.some(l => l.startsWith('|error|')), protocol.filter(l => l.startsWith('|error|')).join('\n'));
    const state = snapshot(stream.battle);
    scenario.check(swap ? [...state].reverse() : state, protocol);
    return {id: `${scenario.id}/${swap ? 'reverse' : 'forward'}`, scenario: scenario.id,
      profile: 'r6-constructed-fixed-teams-v1', evidence: 'constructed-scripted-reference', official_generator_witness: false,
      seed, teams, tape, checkpoints, protocol, entries: scenario.entries, fixture_families: scenario.families,
      assertions_passed: true, result_hash: hashJSON({checkpoints, protocol})};
  } finally { stream.destroy(); }
}

export async function captureFixtures() {
  const cases = [];
  for (const scenario of SCENARIOS) for (const swap of [false, true]) {
    const first = await runScenario(scenario, swap), replay = await runScenario(scenario, swap);
    assert.deepEqual(replay, first, `Reference replay mismatch: ${first.id}`);
    cases.push({...first, replay_verified: true});
  }
  const body = {schema_version: 'r6-fixtures-1', artifact_kind: 'reference-fixtures', cases,
    scope: 'Scripted fixed teams, two seats, initialization replay. Checkpoints are selected diagnostics, not restart states or custom-engine parity. These teams need not be generated by the official generator.'};
  return {...body, content_hash: hashJSON(body)};
}
