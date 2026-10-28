import {createRequire} from 'node:module';
import {Dex, seedFor} from '../reference-node/engine.mjs';
import {UPSTREAM} from '../reference-node/provenance.mjs';
import {hashJSON} from '../battle-contract/canonical.mjs';
import {admitTrainingTeam} from '../training-curriculum/index.mjs';
const {TeamValidator} = createRequire(import.meta.url)(`${UPSTREAM}/dist/sim/index.js`);
const validator = new TeamValidator('gen9anythinggoes');
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
function mon(species, moves, ability, teraType = 'Normal', level = 90) {
  const data = Dex.species.get(species);
  return {name: species, species, speciesId: data.id, gender: data.gender ?? 'M', shiny: false,
    level, moves, ability, item: '', teraType, role: 'Constructed F6 diagnostic',
    evs: Object.fromEntries(STATS.map(s => [s, s === 'hp' ? 1 : 0])), ivs: Object.fromEntries(STATS.map(s => [s, 31]))};
}
function admit(team, key) {
  for (let attempt = 0; attempt < 32; attempt++) {
    const candidate = structuredClone(team);
    // Fixed, outcome-independent admission stream changes only a filler IV.
    candidate.at(-1).ivs.atk = 31 - attempt;
    const problems = validator.validateTeam(structuredClone(candidate));
    if (problems) throw new Error(`Invalid F6 team: ${problems.join('; ')}`);
    const entry = admitTrainingTeam({seed: seedFor(`f6/team/${key}/${attempt}`),
      team: candidate, team_hash: hashJSON(candidate)});
    if (entry) return entry;
  }
  throw new Error('F6 train admission budget exhausted');
}
export const CONTEXTS = Array.from({length: 16}, (_, i) => ({
  family: i < 8 ? 'switch' : 'tera', threat: i % 8 < 4 ? 'electric' : 'water',
  order: Math.floor(i / 2) % 2, side: i % 2 ? 'p2' : 'p1'}));
const cache = new Map();
export function scenarioPlan({context, panel, episode, run_seed}) {
  if (!Number.isInteger(context) || context < 0 || context >= 16 ||
      !['train', 'variant'].includes(panel) || !Number.isSafeInteger(episode) || episode < 0 ||
      !Number.isInteger(run_seed) || run_seed < 0 || run_seed >= 2 ** 32) throw new Error('Invalid F6 coordinates');
  const key = `${panel}/${context}`, c = CONTEXTS[context];
  if (!cache.has(key)) {
    const level = panel === 'train' ? 90 : 100;
    const foeSpecies = c.threat === 'electric' ? (panel === 'train' ? 'Jolteon' : 'Raichu') :
      (panel === 'train' ? 'Floatzel' : 'Golduck');
    const foeAbility = {Jolteon: 'Volt Absorb', Raichu: 'Static', Floatzel: 'Water Veil', Golduck: 'Damp'}[foeSpecies];
    const attack = c.threat === 'electric' ? 'thunderbolt' : 'surf';
    const fire = [mon('Magmar', ['leer', 'flamethrower'], 'Flame Body', 'Fire', level),
      mon('Arcanine', ['flamethrower'], 'Flash Fire', 'Fire', level),
      mon('Typhlosion', ['flamethrower'], 'Blaze', 'Fire', level),
      mon('Incineroar', ['flamethrower'], 'Blaze', 'Fire', level),
      mon('Magmortar', ['flamethrower'], 'Flame Body', 'Fire', level)];
    for (const p of fire) p.evs = {hp: 252, atk: 0, def: 0, spa: 0, spd: 252, spe: 4};
    let own, prefix;
    if (c.family === 'switch') {
      const counters = [mon('Dugtrio', ['earthquake'], 'Sand Force', 'Ground', level),
        mon('Vaporeon', ['surf'], 'Water Absorb', 'Water', level)];
      if (c.order) counters.reverse();
      own = [mon('Blissey', ['pound'], 'Natural Cure', 'Normal', level), ...counters,
        mon('Staraptor', ['aerialace'], 'Reckless', 'Normal', level), ...fire.slice(0, 2)];
      prefix = [{own: 'move 1', foe: 'move 1'}];
    } else {
      // Reveal the attack against Magmar, then switch to Gyarados while Agility
      // resolves. The learner sees the opponent move before deciding about Tera.
      own = [fire[0], mon('Gyarados', ['waterfall'], 'Moxie', 'Ground', level), ...fire.slice(1)];
      if (c.order) [own[1], own[2]] = [own[2], own[1]];
      prefix = [{own: 'move 1', foe: 'move 1'},
        {own: `switch ${c.order ? 3 : 2}`, foe: 'move 2'}];
    }
    const foe = [mon(foeSpecies, [attack, foeSpecies === 'Golduck' ? 'amnesia' : 'agility'], foeAbility, 'Normal', level),
      mon('Blissey', ['pound'], 'Natural Cure', 'Normal', level),
      mon('Dugtrio', ['earthquake'], 'Sand Force', 'Ground', level),
      mon('Vaporeon', ['surf'], 'Water Absorb', 'Water', level), ...fire.slice(1, 3)];
    const other = c.side === 'p1' ? 'p2' : 'p1';
    const ownEntry = admit(own, `${key}/own`), foeEntry = admit(foe, `${key}/foe`);
    cache.set(key, {teams: c.side === 'p1' ? [ownEntry, foeEntry] : [foeEntry, ownEntry],
      prefix: prefix.map(row => ({[c.side]: row.own, [other]: row.foe}))});
  }
  return {...structuredClone(cache.get(key)), seed: seedFor(`f6/${panel}/${run_seed}/${episode}/${context}`),
    provenance: {source: 'constructed-decision-canary', panel, context, ...c, split: 'train',
      run_seed, episode, prefix_actions_trainable: false, objective: 'one-turn-exposed-member-HP-retention'}};
}
