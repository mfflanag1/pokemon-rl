import {PublicBattle, toID} from '../showdown-bot/protocol.mjs';

const effectID = value => toID((value ?? '').replace(/^(move|ability|item): /, ''));
const STATS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];

/** Extra observed events for the frozen opponents; no access to a simulator or
 * the other player's request. The sealed candidate keeps its original reducer. */
export class TacticalView extends PublicBattle {
  constructor() {
    super(); this.known = {}; this.weather = ''; this.terrain = '';
    this.hazards = {p1: {}, p2: {}};
  }
  ingest(line) {
    const [, event, ident, value, amount] = line.split('|');
    const previous = this.active[ident?.slice(0, 2)];
    super.ingest(line);
    const seat = ident?.slice(0, 2), mon = this.active[seat];
    if (['switch', 'drag', 'replace'].includes(event)) {
      const priorMemory = structuredClone(this.known[ident] ?? {moves: [], ability: null, item: null});
      const memory = structuredClone(priorMemory);
      if (event === 'replace' && previous) {
        // Only transfer this appearance's reveals; earlier knowledge of the
        // disguised species belongs to that species, not the Illusion user.
        memory.moves = [...new Set([...memory.moves, ...previous.entryReveals.moves])];
        if (previous.entryReveals.ability != null) memory.ability = previous.entryReveals.ability;
        if (previous.entryReveals.item != null) memory.item = previous.entryReveals.item;
        if (previous.ident !== ident) this.known[previous.ident] = structuredClone(previous.memoryAtEntry);
      }
      this.known[ident] = memory;
      Object.assign(mon, {level: Number(/(?:^|, )L(\d+)/.exec(value)?.[1] ?? 100),
        boosts: event === 'replace' ? {...previous?.boosts} : {},
        memoryAtEntry: priorMemory,
        entryReveals: event === 'replace' ? structuredClone(previous.entryReveals) : {moves: [], ability: null, item: null},
        enteredTurn: event === 'replace' ? previous?.enteredTurn ?? this.turn : this.turn, ...structuredClone(memory)});
    }
    if (mon && event === 'move' && mon.ident === ident) {
      const id = toID(value); if (!mon.moves.includes(id)) mon.moves.push(id);
      this.known[ident].moves = [...mon.moves];
      if (!mon.entryReveals.moves.includes(id)) mon.entryReveals.moves.push(id);
    }
    if (mon && ['-ability', '-item', '-enditem'].includes(event)) {
      const key = event === '-ability' ? 'ability' : 'item';
      mon[key] = event === '-enditem' ? '' : toID(value);
      this.known[mon.ident][key] = mon[key];
      mon.entryReveals[key] = mon[key];
    }
    if (mon && ['-boost', '-unboost', '-setboost'].includes(event) && STATS.includes(value)) {
      const n = Number(amount); if (!Number.isFinite(n)) throw new Error('Malformed observed stat stage');
      mon.boosts[value] = Math.max(-6, Math.min(6, event === '-setboost' ? n :
        (mon.boosts[value] ?? 0) + (event === '-boost' ? n : -n)));
    }
    if (mon && event === '-clearboost') mon.boosts = {};
    if (mon && ['-clearnegativeboost', '-clearpositiveboost'].includes(event)) {
      for (const key of Object.keys(mon.boosts)) {
        if (event === '-clearnegativeboost' ? mon.boosts[key] < 0 : mon.boosts[key] > 0) delete mon.boosts[key];
      }
    }
    // Pinned Psych Up/Costar emit recipient first, despite the protocol prose.
    const other = this.active[value?.slice(0, 2)];
    if (mon && other?.ident === value && event === '-copyboost') mon.boosts = {...other.boosts};
    if (mon && other?.ident === value && event === '-swapboost') {
      const stats = amount && !amount.startsWith('[') ? amount.split(',').map(s => s.trim()) : STATS;
      for (const stat of stats.filter(s => STATS.includes(s))) {
        [mon.boosts[stat], other.boosts[stat]] = [other.boosts[stat] ?? 0, mon.boosts[stat] ?? 0];
      }
    }
    if (mon && event === '-status') mon.condition = `${(mon.condition ?? '100/100').split(' ')[0]} ${value}`;
    if (mon && event === '-curestatus') mon.condition = (mon.condition ?? '100/100').split(' ')[0];
    if (event === '-clearallboost') for (const active of Object.values(this.active)) active.boosts = {};
    if (mon && event === '-invertboost') for (const key of Object.keys(mon.boosts)) mon.boosts[key] *= -1;
    if (event === '-weather') this.weather = value === '[upkeep]' ? this.weather : effectID(ident);
    if (event === '-fieldstart' && effectID(ident).endsWith('terrain')) this.terrain = effectID(ident);
    if (event === '-fieldend' && effectID(ident) === this.terrain) this.terrain = '';
    if (['-sidestart', '-sideend'].includes(event) && this.hazards[seat]) {
      const id = effectID(value), max = {stealthrock: 1, spikes: 3, toxicspikes: 2, stickyweb: 1}[id];
      if (max) this.hazards[seat][id] = event === '-sideend' ? 0 : Math.min(max, (this.hazards[seat][id] ?? 0) + 1);
    }
  }
}
