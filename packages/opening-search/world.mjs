import {isDeepStrictEqual} from 'node:util';
import {OpponentMixSession} from '../opponent-mix/session.mjs';
import {Dex, generateTeam, seedFor} from '../reference-node/engine.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {partition} from '../dataset/identity.mjs';
import {hashJSON} from '../reference-node/provenance.mjs';

const SIDES = ['p1', 'p2'];
const clone = x => structuredClone(x);
const id = x => String(x).toLowerCase().replace(/[^a-z0-9]/g, '');

export function publicOpening(session, side) {
  if (!SIDES.includes(side)) throw new Error('Unknown side');
  const frame = session.snapshot();
  if (session.submissions || frame.outcome || frame.players[side].turn !== 1 ||
      !SIDES.every(s => frame.players[s].phase === 'move')) throw new Error('Opening only');
  const view = session.projection.players[side];
  // Starting at |start| removes administrative metadata; private request lines
  // are replaced by exactly this player's current request.
  const first = view.protocol.findIndex(l => l === '|start' || l === '|start|');
  if (first < 0) throw new Error('Missing public start');
  const lines = view.protocol.slice(first).filter(l =>
    l && !l.startsWith('|request|') && !l.startsWith('|t:|'));
  return clone({side, request: view.request, public_lines: lines, player: frame.players[side]});
}

export function validateInput(input) {
  if (!input || Object.keys(input).sort().join() !== 'player,public_lines,request,side' ||
      !SIDES.includes(input.side) || input.request?.side?.id !== input.side ||
      input.player?.side !== input.side || input.player.turn !== 1 || input.player.phase !== 'move' ||
      !['|start','|start|'].includes(input.public_lines?.[0]) || input.public_lines.some(l =>
        typeof l !== 'string' || l.startsWith('|request|') || l.startsWith('|split|')))
    throw new Error('Invalid player-only opening');
  if (input.request.side.pokemon.length !== 6 || input.player.own.length !== 6)
    throw new Error('Expected six own members');
  return input;
}

// Pinned random battles use neutral natures and IV 0/31 with EV <=85. Match
// the actual private stats, then require full opening reconstruction equality.
export function ownTeam(input) {
  validateInput(input);
  return input.request.side.pokemon.map(p => {
    const species = p.details.split(',')[0], dex = Dex.species.get(species);
    const level = Number(/(?:^|, )L(\d+)/.exec(p.details)?.[1] ?? 100);
    const hp = /^(\d+)\/(\d+)$/.exec(p.condition);
    if (!dex.exists || !hp || hp[1] !== hp[2] || p.terastallized || dex.battleOnly)
      throw new Error('Unsupported own opening form/condition');
    const evs = {}, ivs = {};
    for (const stat of ['hp','atk','def','spa','spd','spe']) {
      const target = stat === 'hp' ? Number(hp[2]) : p.stats[stat];
      let found = false;
      for (const iv of [31,0]) {
        for (let ev = 85; ev >= 0; ev--) {
          const value = stat === 'hp' && dex.baseStats.hp === 1 ? 1 :
            Math.floor((2*dex.baseStats[stat]+iv+Math.floor(ev/4))*level/100) +
            (stat === 'hp' ? level+10 : 5);
          if (value === target) { ivs[stat] = iv; evs[stat] = ev; found = true; break; }
        }
        if (found) break;
      }
      if (!found) throw new Error('Private stat cannot be reconstructed');
    }
    return {name:p.ident.split(': ').slice(1).join(': '), species, level,
      gender:/(?:^|, )(M|F)(?:,|$)/.exec(p.details)?.[1] ?? '',
      shiny:p.details.includes(', shiny'), moves:clone(p.moves),
      ability:p.baseAbility ?? p.ability, item:p.item, teraType:p.teraType,
      evs, ivs, nature:'Serious'};
  });
}

export class HypotheticalSession extends OpponentMixSession {
  async startWorld(teams, mechanicsSeed) {
    this.started = true; this.split = 'development'; this.swap = false;
    this.episodeId = 'f33-hypothetical'; this.provenance = {hypothetical:true};
    await this.stream.write(`>start ${JSON.stringify({formatid:'gen9randombattle',seed:mechanicsSeed})}\n`+
      teams.map((team,i) => `>player p${i+1} ${JSON.stringify({name:`Full p${i+1}`,team})}`).join('\n'));
    await this.passControls();
    return this.snapshot();
  }
}

export async function openWorld(world) {
  const session = new HypotheticalSession();
  try { await session.startWorld(world.teams, world.mechanics_seed); return session; }
  catch (error) { session.close(); throw error; }
}

export function admittedHypothesis(team, seed) {
  return partition(teamArtifact({team,seed,team_hash:hashJSON(team)},'0'.repeat(64)).payload)==='train';
}

function comparable(input) {
  const result = clone(input);
  delete result.player.request_id;
  delete result.request.rqid;
  return result;
}

// An explicitly approximate prior: generate complete training-partition teams,
// consider each of their six cyclic lead rotations, and retain openings whose
// entire projected input matches. It is not the official conditional posterior.
// No coordinator seed, true opposing team, or live mechanics state is an input.
export async function sampleWorlds(input, searchSeed, {worlds, attempts}) {
  validateInput(input);
  if (!Number.isSafeInteger(searchSeed) || searchSeed < 0 || searchSeed >= 2**32 ||
      !Number.isInteger(worlds) || worlds < 1 || worlds > 8 ||
      !Number.isInteger(attempts) || attempts < 1 || attempts > 512) throw new Error('Fixed search budget');
  let own;
  try { own = ownTeam(input); }
  catch (error) { return {worlds:[], attempts:0, reason:error.message}; }
  const foe = input.side === 'p1' ? 'p2' : 'p1';
  const observed = input.player.opponent.find(p => p?.active);
  if (!observed) return {worlds:[],attempts:0,reason:'Missing public active opponent'};
  const accepted = [], seen = new Set(); let used = 0;
  for (; used < attempts && accepted.length < worlds; used++) {
    const entry = generateTeam(seedFor(`f33/hypotheses/${searchSeed}/${used}`));
    if (partition(teamArtifact(entry,'0'.repeat(64)).payload) !== 'train') continue;
    for (let rotation = 0; rotation < 6 && accepted.length < worlds; rotation++) {
      const team = [...entry.team.slice(rotation), ...entry.team.slice(0,rotation)];
      const lead = team[0], apparent = id(lead.ability) === 'illusion' ? team[5] : lead;
      if (id(apparent.species) !== id(observed.species)) continue;
      // R7 hashes ordered teams. A permitted source team does not imply that
      // its lead rotation is permitted; re-admit the actual simulated team.
      if (!admittedHypothesis(team,entry.seed)) continue;
      const key = JSON.stringify(team);
      if (seen.has(key)) continue;
      const teams = input.side === 'p1' ? [own,team] : [team,own];
      const world = {teams,mechanics_seed:seedFor(`f33/chance/${searchSeed}/${accepted.length}`),
        source_attempt:used, rotation, side:input.side, foe};
      const session = await openWorld(world);
      try {
        if (!isDeepStrictEqual(comparable(publicOpening(session,input.side)),comparable(input))) continue;
        seen.add(key); accepted.push(world);
      } finally { session.close(); }
    }
  }
  return {worlds:accepted.length === worlds ? accepted : [], attempts:used,
    matched:accepted.length, reason:accepted.length === worlds ? null : 'Hypothesis budget exhausted'};
}
