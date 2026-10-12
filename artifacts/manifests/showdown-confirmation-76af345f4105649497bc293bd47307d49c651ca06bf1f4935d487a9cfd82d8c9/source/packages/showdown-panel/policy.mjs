import {Dex} from '../reference-node/engine.mjs';
import {legalChoices, PublicBattle, toID} from '../showdown-bot/protocol.mjs';
import {makePolicy} from '../showdown-bot/policy.mjs';
import {TacticalView} from './view.mjs';

export const POLICY_IDS = ['baseline-v1', 'random-v1', 'damage-v1', 'tactical-v1'];
export const ASSUMPTIONS = Object.freeze({opponent_ivs: 31, opponent_evs_per_stat: 85,
  opponent_nature: 'neutral', unknown_item: 'no modifier', unknown_ability: 'no modifier', random_damage_multiplier: 0.925});
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const species = mon => Dex.species.get(mon?.species ?? mon?.details?.split(',')[0] ?? '');
const level = mon => mon?.level ?? Number(/(?:^|, )L(\d+)/.exec(mon?.details ?? '')?.[1] ?? 100);
const stage = n => (n ?? 0) >= 0 ? (2 + (n ?? 0)) / 2 : 2 / (2 - n);
const accuracyStage = n => n >= 0 ? (3 + n) / 3 : 3 / (3 - n);
export const fraction = mon => {
  const [hp, max = 100] = (mon?.condition ?? '100/100').split(' ')[0].split('/').map(Number);
  return clamp(hp / max, 0, 1);
};
const status = mon => (mon?.condition ?? '').split(' ')[1] ?? '';
export function types(mon, tera = mon?.tera || mon?.terastallized) {
  if (tera && tera !== 'Stellar') return [tera];
  return [...new Set([...(mon?.types ?? species(mon).types ?? []), ...(mon?.addedType ? [mon.addedType] : [])])];
}
function estimate(mon, stat) {
  if (stat !== 'hp' && Number.isFinite(mon?.stats?.[stat])) return mon.stats[stat];
  if (stat === 'hp' && mon?.stats && mon.condition.includes('/')) return Number(mon.condition.split(' ')[0].split('/')[1]);
  const base = species(mon).baseStats?.[stat] ?? 80;
  if (stat === 'hp' && base === 1) return 1;
  const value = Math.floor((2 * base + 31 + Math.floor(85 / 4)) * level(mon) / 100);
  return value + (stat === 'hp' ? level(mon) + 10 : 5);
}
function effectiveStat(mon, stat) {
  let value = estimate(mon, stat) * stage(mon?.boosts?.[stat]);
  const item = toID(mon?.item), ability = toID(mon?.ability ?? mon?.baseAbility);
  if (stat === 'atk' && ['hugepower', 'purepower'].includes(ability)) value *= 2;
  if (stat === 'atk' && ability === 'guts' && status(mon)) value *= 1.5;
  if (item === {atk: 'choiceband', spa: 'choicespecs', spe: 'choicescarf'}[stat]) value *= 1.5;
  if (stat === 'spe' && status(mon) === 'par') value *= 0.5;
  if (item === 'assaultvest' && stat === 'spd') value *= 1.5;
  if (item === 'eviolite' && ['def', 'spd'].includes(stat) && species(mon).nfe) value *= 1.5;
  return Math.max(1, value);
}
function effectiveness(type, defender, tera) {
  const defendingTypes = types(defender, tera);
  if (!Dex.getImmunity(type, defendingTypes)) return 0;
  const ability = toID(defender?.ability ?? defender?.baseAbility);
  const immunity = {levitate: 'Ground', eartheater: 'Ground', flashfire: 'Fire', wellbakedbody: 'Fire',
    waterabsorb: 'Water', stormdrain: 'Water', dryskin: 'Water', voltabsorb: 'Electric', lightningrod: 'Electric',
    motordrive: 'Electric', sapsipper: 'Grass'};
  if (immunity[ability] === type) return 0;
  return 2 ** Dex.getEffectiveness(type, defendingTypes);
}

/** Approximate damage fraction, not a mechanics simulator or hidden-set oracle.
 * Defender stats without an own request use explicit neutral public-species proxies. */
export function damageProxy(moveId, own, foe, view, tera = null, advertised = null) {
  const move = Dex.moves.get(moveId); if (!move.exists || move.category === 'Status') return 0;
  const currentTera = tera || own.tera || own.terastallized;
  const type = move.id === 'terablast' && currentTera ? currentTera : advertised?.type ?? move.type;
  const category = move.id === 'terablast' && currentTera && effectiveStat(own, 'atk') > effectiveStat(own, 'spa') ? 'Physical' : move.category;
  const attack = category === 'Physical' ? 'atk' : 'spa', defense = category === 'Physical' ? 'def' : 'spd';
  let power = advertised?.basePower ?? move.basePower;
  if (!power) power = 60; // Unmodeled dynamic-power callbacks have an explicit fixed proxy.
  const ability = toID(own.ability ?? own.baseAbility), item = toID(own.item);
  if (move.multihit) power *= typeof move.multihit === 'number' ? move.multihit :
    ability === 'skilllink' ? move.multihit[1] : item === 'loadeddice' ? 4.5 : 3;
  if (ability === 'technician' && (move.basePower || power) <= 60) power *= 1.5;
  const natural = species(own).types ?? [], current = types(own);
  let stab = current.includes(type) ? 1.5 : 1;
  if (currentTera && currentTera !== 'Stellar') {
    stab = natural.includes(type) ? 1.5 : 1;
    if (currentTera === type) stab = natural.includes(type) ? 2 : 1.5;
  }
  if (ability === 'adaptability' && stab > 1) stab += 0.5;
  let modifier = stab * effectiveness(type, foe) * 0.925;
  if (category === 'Physical' && status(own) === 'brn' && ability !== 'guts' && move.id !== 'facade') modifier *= 0.5;
  if (item === 'lifeorb') modifier *= 1.3;
  if (move.id === 'facade' && status(own)) power *= 2;
  if (view.weather === 'raindance') modifier *= type === 'Water' ? 1.5 : type === 'Fire' ? 0.5 : 1;
  if (view.weather === 'sunnyday') modifier *= type === 'Fire' ? 1.5 : type === 'Water' ? 0.5 : 1;
  let damage = ((2 * level(own) / 5 + 2) * power * effectiveStat(own, attack) / effectiveStat(foe, defense) / 50 + 2) * modifier;
  if (move.damage === 'level') damage = modifier ? level(own) : 0;
  else if (typeof move.damage === 'number') damage = modifier ? move.damage : 0;
  else if (['superfang', 'naturesmadness', 'ruination'].includes(move.id)) damage = modifier ? estimate(foe, 'hp') * fraction(foe) / 2 : 0;
  let accuracy = move.accuracy === true ? 1 : move.accuracy / 100;
  if (ability === 'noguard' || toID(foe?.ability) === 'noguard') accuracy = 1;
  else if (move.accuracy !== true) accuracy *= accuracyStage(clamp((own.boosts?.accuracy ?? 0) - (foe?.boosts?.evasion ?? 0), -6, 6));
  const raw = Math.max(0, damage / estimate(foe, 'hp'));
  let utility = Math.min(raw, fraction(foe)) * clamp(accuracy, 0, 1);
  if (move.flags.charge && item !== 'powerherb' && !(view.weather === 'sunnyday' && ['solarbeam', 'solarblade'].includes(move.id))) utility *= 0.55;
  if (move.self?.volatileStatus === 'mustrecharge') utility *= 0.65;
  if (move.selfdestruct && fraction(own) > 0.25) utility *= 0.6;
  if (move.id === 'suckerpunch') utility *= 0.65;
  if (['fakeout', 'firstimpression'].includes(move.id) && view.turn > (own.enteredTurn ?? -2) + 1) utility = 0;
  if (raw >= fraction(foe) && move.priority > 0 && utility > 0) utility += 0.12 * clamp(accuracy, 0, 1);
  return utility;
}
function ownMon(mon, view, seat, active = false) {
  return {...(active ? view.active[seat] : {}), ...mon, species: mon.details.split(',')[0],
    boosts: active ? view.active[seat]?.boosts ?? {} : {},
    enteredTurn: active ? view.active[seat]?.enteredTurn : view.turn};
}
function incomingProxy(own, foe, view) {
  const revealed = (foe?.moves ?? []).filter(id => Dex.moves.get(id).category !== 'Status');
  const attacks = revealed.map(id => damageProxy(id, foe, own, view));
  // Unknown moves are assumptions, never inferred facts: 80-power STAB proxies
  // using the public species' larger attacking stat and each visible type.
  const offense = Math.max(effectiveStat(foe, 'atk'), effectiveStat(foe, 'spa'));
  for (const type of types(foe)) attacks.push(((2 * level(foe) / 5 + 2) * 80 * offense /
    Math.min(effectiveStat(own, 'def'), effectiveStat(own, 'spd')) / 50 + 2) * 1.5 * 0.925 * effectiveness(type, own) / estimate(own, 'hp'));
  return Math.min(1, Math.max(0.05, ...attacks));
}
function entryCost(mon, view, seat) {
  if (toID(mon.item) === 'heavydutyboots') return 0;
  const h = view.hazards[seat], grounded = !types(mon).includes('Flying') && toID(mon.ability ?? mon.baseAbility) !== 'levitate';
  return (h.stealthrock ? effectiveness('Rock', mon) / 8 : 0) +
    (grounded && h.spikes ? [0, 1 / 8, 1 / 6, 1 / 4][h.spikes] : 0);
}
export function switchValue(mon, foe, view, seat, entering = true) {
  return Math.max(0, ...mon.moves.map(id => damageProxy(id, mon, foe, view))) -
    0.65 * incomingProxy(mon, foe, view) + 0.12 * fraction(mon) - (entering ? entryCost(mon, view, seat) : 0);
}
function statusValue(move, own, foe, view, request) {
  const hp = fraction(own), danger = incomingProxy(own, foe, view), foeStatus = status(foe);
  if (own.volatiles?.taunt) return 0;
  if (move.heal || move.id === 'rest') {
    if (hp > 0.65 || hp <= danger && effectiveStat(own, 'spe') < effectiveStat(foe, 'spe')) return 0;
    return Math.min(1 - hp, move.heal ? move.heal[0] / move.heal[1] : 1) * (move.id === 'rest' ? 0.65 : 0.95);
  }
  if (move.boosts && move.target === 'self' && hp > danger * 1.7) {
    let gain = 0;
    for (const stat of ['atk', 'spa', 'spe']) if ((move.boosts[stat] ?? 0) > 0 && (own.boosts?.[stat] ?? 0) < 2) {
      const useful = stat === 'spe' || own.moves.some(id => Dex.moves.get(id).category === (stat === 'atk' ? 'Physical' : 'Special'));
      if (useful) gain += move.boosts[stat] * 0.14;
    }
    return Math.min(0.42, gain) * (1 - danger);
  }
  if (move.status && !foeStatus) {
    if (['slp', 'tox', 'brn', 'par'].includes(move.status)) {
      if (move.status === 'tox' && types(foe).some(t => ['Steel', 'Poison'].includes(t))) return 0;
      if (move.status === 'brn' && types(foe).includes('Fire') || move.status === 'par' && types(foe).includes('Electric')) return 0;
      return (move.status === 'slp' ? 0.35 : 0.24) * (move.accuracy === true ? 1 : move.accuracy / 100);
    }
  }
  const enemy = request.side.id === 'p1' ? 'p2' : 'p1';
  if (['stealthrock', 'spikes', 'stickyweb'].includes(move.sideCondition) && !view.hazards[enemy][move.sideCondition] && view.turn < 12) return 0.22;
  if (['defog', 'rapidspin', 'mortalspin'].includes(move.id) && Object.values(view.hazards[request.side.id]).some(Boolean)) return 0.22;
  return 0.01;
}

export function chooseOpponent(kind, request, view) {
  const choices = legalChoices(request); if (!choices.length) return null;
  if (request.teamPreview || choices[0] === 'pass') return choices[0];
  const seat = request.side.id, foe = view.active[seat === 'p1' ? 'p2' : 'p1'];
  const team = request.side.pokemon;
  if (request.forceSwitch) {
    // Revival choices reference fainted slots; rank their prospective full-health matchup.
    return choices.map(choice => {
      let mon = ownMon(team[Number(choice.split(' ')[1]) - 1], view, seat);
      if (team[0].reviving) { const hp = estimate(mon, 'hp'); mon = {...mon, condition: `${hp}/${hp}`}; }
      return [choice, switchValue(mon, foe, view, seat)];
    }).sort((a, b) => b[1] - a[1])[0][0];
  }
  const own = ownMon(team.find(p => p.active), view, seat, true), active = request.active[0];
  const scores = choices.filter(c => c.startsWith('move ')).map(choice => {
    const advertised = active.moves[Number(choice.split(' ')[1]) - 1], move = Dex.moves.get(advertised.id ?? advertised.move);
    const tera = choice.endsWith(' terastallize') ? active.canTerastallize : null;
    let score = move.category === 'Status' ? (kind === 'tactical-v1' ? statusValue(move, own, foe, view, request) : 0.005) :
      damageProxy(move.id, own, foe, view, tera, advertised);
    if (tera) score = score * 0.9 - 0.04;
    if (kind === 'tactical-v1' && move.drain) score += Math.min(1 - fraction(own), score * move.drain[0] / move.drain[1]) * 0.35;
    return [choice, score];
  }).sort((a, b) => b[1] - a[1]);
  const best = scores[0];
  if (kind === 'tactical-v1' && best[1] < 0.22 && fraction(own) > 0.3) {
    const current = switchValue(own, foe, view, seat, false);
    const candidate = choices.filter(c => c.startsWith('switch ')).map(choice =>
      [choice, switchValue(ownMon(team[Number(choice.split(' ')[1]) - 1], view, seat), foe, view, seat)])
      .sort((a, b) => b[1] - a[1])[0];
    if (candidate && candidate[1] > current + 0.3 && candidate[1] > best[1] + 0.15) return candidate[0];
  }
  return best[0];
}

export function createActor(id, seed) {
  if (!POLICY_IDS.includes(id)) throw new Error('Unknown panel policy');
  if (id === 'baseline-v1' || id === 'random-v1') return {state: new PublicBattle(), choose: makePolicy(id === 'baseline-v1' ? 'heuristic' : 'random', seed)};
  return {state: new TacticalView(), choose: (request, state) => chooseOpponent(id, request, state)};
}
