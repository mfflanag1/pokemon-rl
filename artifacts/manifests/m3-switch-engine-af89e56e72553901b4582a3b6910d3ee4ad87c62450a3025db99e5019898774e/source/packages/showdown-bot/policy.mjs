import {Dex, PRNG} from '../reference-node/engine.mjs';
import {legalChoices} from './protocol.mjs';

export const POLICY_ID = 'public-matchup-heuristic-v1';
const speciesOf = mon => mon.species ?? mon.details?.split(',')[0];
const hpFraction = condition => {
  const [hp, max = '100'] = (condition ?? '100/100').split(' ')[0].split('/');
  return Number(hp) / Number(max);
};
function typesOf(mon) {
  if (!mon) return [];
  const tera = mon.tera || mon.terastallized;
  if (tera && tera !== 'Stellar') return [tera];
  return [...new Set([...(mon.types ?? Dex.species.get(speciesOf(mon)).types), ...(mon.addedType ? [mon.addedType] : [])])];
}
function effectiveness(type, defender) {
  const types = typesOf(defender);
  if (!types.length) return 1;
  return Dex.getImmunity(type, types) ? 2 ** Dex.getEffectiveness(type, types) : 0;
}

// A ranking heuristic, NOT a damage calculator: no hidden items, abilities, stats or moves are inferred as facts.
function moveScore(moveId, own, foe, {tera = null, advertised = null} = {}) {
  const move = Dex.moves.get(moveId);
  if (!move.exists) return 1;
  const fraction = hpFraction(own.condition);
  if (move.category === 'Status') {
    if ((move.heal || move.id === 'rest') && fraction < 0.45 && !own.condition?.includes(' slp')) return 95 * (1 - fraction);
    return 4; // A deliberately weak fallback; no status/setup search in this baseline.
  }
  let type = advertised?.type ?? move.type;
  if (move.id === 'terablast' && (tera || own.tera || own.terastallized)) type = tera || own.tera || own.terastallized;
  const power = advertised?.basePower || move.basePower || (move.damage ? 70 : 50);
  const natural = Dex.species.get(speciesOf(own)).types;
  const current = typesOf(own);
  let stab = current.includes(type) ? 1.5 : 1;
  const teraType = tera || own.tera || own.terastallized;
  if (teraType && teraType !== 'Stellar') {
    stab = natural.includes(type) ? 1.5 : 1;
    if (type === teraType) stab = natural.includes(type) ? 2 : 1.5;
  }
  let offense = own.stats?.[move.category === 'Physical' ? 'atk' : 'spa'] ?? 100;
  if (move.category === 'Physical' && own.condition?.includes(' brn')) offense *= 0.5;
  const accuracy = move.accuracy === true ? 1 : move.accuracy / 100;
  return power * stab * effectiveness(type, foe) * accuracy * (offense / 100) + Math.max(0, move.priority) * 3;
}

function switchScore(mon, foe) {
  const attacks = (mon.moves ?? []).map(move => moveScore(move, mon, foe));
  const incoming = Math.max(1, ...typesOf(foe).map(type => effectiveness(type, mon)));
  return Math.max(1, ...attacks) / Math.sqrt(incoming) * (0.5 + 0.5 * hpFraction(mon.condition));
}

export function chooseHeuristic(request, state) {
  const choices = legalChoices(request);
  if (!choices.length) return null;
  if (request.teamPreview || choices[0] === 'pass') return choices[0];
  const seat = request.side.id, foe = state.active[seat === 'p1' ? 'p2' : 'p1'];
  const team = request.side.pokemon;
  if (request.forceSwitch) {
    return choices.map(choice => [choice, switchScore(team[Number(choice.split(' ')[1]) - 1], foe)])
      .sort((a, b) => b[1] - a[1])[0][0];
  }
  const own = {...team.find(p => p.active), ...state.active[seat]};
  // Public events supply displayed typing; the current private request supplies exact own HP/stats/moves.
  Object.assign(own, team.find(p => p.active));
  const active = request.active[0];
  const scores = choices.filter(c => c.startsWith('move ')).map(choice => {
    const m = active.moves[Number(choice.split(' ')[1]) - 1];
    const tera = choice.endsWith(' terastallize') ? active.canTerastallize : null;
    let score = moveScore(m.id ?? m.move, own, foe, {tera, advertised: m});
    if (tera) score = score * 0.85 - 8; // Spend Tera only for a substantial immediate offensive gain.
    return [choice, score];
  });
  const best = scores.sort((a, b) => b[1] - a[1])[0];
  // Avoid switching loops: switch only out of very poor attacks into a materially better matchup.
  if (best[1] < 45 && hpFraction(own.condition) > 0.25) {
    const switches = choices.filter(c => c.startsWith('switch ')).map(choice =>
      [choice, switchScore(team[Number(choice.split(' ')[1]) - 1], foe) * 0.55 - 20]);
    const candidate = switches.sort((a, b) => b[1] - a[1])[0];
    if (candidate?.[1] > Math.max(30, best[1] * 1.5)) return candidate[0];
  }
  return best[0];
}

export function makePolicy(kind = 'heuristic', seed = [1, 2, 3, 4]) {
  if (kind === 'heuristic') return chooseHeuristic;
  if (kind !== 'random') throw new Error('Unknown policy');
  const rng = new PRNG(seed);
  return request => {
    const choices = legalChoices(request);
    if (!choices.length) return null;
    const moves = choices.filter(c => c.startsWith('move ') && !c.endsWith(' terastallize'));
    const pool = moves.length && rng.random() < 0.9 ? moves : choices.filter(c => !c.endsWith(' terastallize'));
    return pool[rng.random(pool.length)];
  };
}
