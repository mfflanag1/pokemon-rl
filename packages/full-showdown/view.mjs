import {Dex} from '../reference-node/engine.mjs';
import {toID} from '../showdown-bot/protocol.mjs';
import {requestActions} from './actions.mjs';

const clone = value => structuredClone(value);
const name = ident => ident?.split(': ').slice(1).join(': ');
const seat = ident => ident?.slice(0, 2);
const effect = text => toID((text ?? '').replace(/^(move|ability|item): /, ''));
const details = text => ({species: text?.split(',')[0] ?? null,
  level: Number(/(?:^|, )L(\d+)/.exec(text ?? '')?.[1] ?? 100)});
const fresh = (ident, text, condition) => ({ident: name(ident), ...details(text), condition,
  active: true, moves: [], ability: null, item: null, tera: null, boosts: {}, types: null});

/** Consumes one player's already projected public events only. Six opponent slots
 * mean six observed identities, not omniscient team order. Illusion replacement
 * repairs the current appearance; its previously known disguise remains intact. */
export class RosterView {
  constructor(side) {
    if (!['p1', 'p2'].includes(side)) throw new Error('Invalid viewer');
    this.side = side; this.foe = side === 'p1' ? 'p2' : 'p1';
    this.slots = Array(6).fill(null); this.active = -1; this.entry = null;
    this.turn = 0; this.weather = null; this.terrain = null;
    this.sideConditions = {p1: {}, p2: {}}; this.ownEvents = new Map();
    this.lastOwn = null;
  }

  ingest(line) {
    const [, kind, ident, value, extra] = line.split('|');
    if (kind === 'request' || kind === 'split') throw new Error('Private/unprojected line in public reducer');
    if (kind === 'turn') this.turn = Number(ident);
    if (kind === '-weather') this.weather = effect(ident) === 'none' ? null : effect(ident);
    if (kind === '-fieldstart' && effect(ident).endsWith('terrain')) this.terrain = effect(ident);
    if (kind === '-fieldend' && effect(ident) === this.terrain) this.terrain = null;
    if (['-sidestart', '-sideend'].includes(kind) && this.sideConditions[seat(ident)]) {
      const id = effect(value), limit = {spikes: 3, toxicspikes: 2}[id] ?? 1;
      this.sideConditions[seat(ident)][id] = kind === '-sideend' ? 0 :
        Math.min(limit, (this.sideConditions[seat(ident)][id] ?? 0) + 1);
    }
    if (seat(ident) === this.side && name(ident)) {
      const own = this.ownEvents.get(name(ident)) ?? {};
      if (['switch', 'drag', 'replace'].includes(kind)) {
        for (const known of this.ownEvents.values()) known.active = false;
        Object.assign(own, {condition: extra ?? own.condition, active: true, boosts: {}});
      }
      if (['-damage', '-heal', '-sethp'].includes(kind)) own.condition = value;
      if (kind === 'faint') own.condition = '0 fnt';
      if (kind === '-terastallize') own.tera = value;
      if (kind === '-status') own.condition = `${(own.condition ?? '100/100').split(' ')[0]} ${value}`;
      if (kind === '-curestatus') own.condition = (own.condition ?? '100/100').split(' ')[0];
      if (['-boost', '-unboost', '-setboost'].includes(kind)) {
        own.boosts ??= {}; own.boosts[value] = Math.max(-6, Math.min(6,
          kind === '-setboost' ? Number(extra) : (own.boosts[value] ?? 0) + (kind === '-boost' ? 1 : -1) * Number(extra)));
      }
      if (kind === '-clearboost') own.boosts = {};
      this.ownEvents.set(name(ident), own);
    }
    if (seat(ident) !== this.foe) return;
    const key = name(ident);
    if (['switch', 'drag'].includes(kind)) {
      this.slots.forEach(mon => { if (mon) mon.active = false; });
      let index = this.slots.findIndex(mon => mon?.ident === key);
      const existing = index >= 0;
      if (!existing) index = this.slots.indexOf(null);
      if (index < 0) throw new Error('Public roster exceeded six identities');
      this.entry = {index, existed: existing, prior: clone(this.slots[index]), reveals: {moves: []}};
      this.slots[index] = {...fresh(ident, value, extra), ...(this.slots[index] ?? {}),
        ...details(value), condition: extra, boosts: {}, active: true};
      this.active = index;
    } else if (kind === 'replace') {
      if (!this.entry || this.active < 0) throw new Error('Illusion replacement without appearance');
      const disguise = this.slots[this.active], old = this.entry;
      this.slots[old.index] = old.existed ? {...clone(old.prior), active: false} : null;
      let index = this.slots.findIndex(mon => mon?.ident === key);
      if (index < 0) index = this.slots.indexOf(null);
      if (index < 0) throw new Error('Illusion correction exceeded roster');
      const previous = this.slots[index];
      this.slots[index] = {...fresh(ident, value, extra ?? disguise.condition), ...(previous ?? {}),
        ...details(value), condition: extra ?? disguise.condition, boosts: clone(disguise.boosts),
        active: true, tera: disguise.tera,
        ...old.reveals, moves: [...new Set([...(previous?.moves ?? []), ...old.reveals.moves])]};
      this.active = index; this.entry = {...old, index};
    }
    const mon = this.slots.find(m => m?.ident === key);
    if (!mon) return;
    if (kind === 'move') {
      const move = toID(value); if (!mon.moves.includes(move)) mon.moves.push(move);
      if (this.slots[this.active] === mon && !this.entry.reveals.moves.includes(move)) this.entry.reveals.moves.push(move);
    }
    if (['-ability', '-item', '-enditem'].includes(kind)) {
      const field = kind === '-ability' ? 'ability' : 'item';
      mon[field] = kind === '-enditem' ? '' : toID(value);
      if (this.slots[this.active] === mon) this.entry.reveals[field] = mon[field];
    }
    if (['-damage', '-heal', '-sethp'].includes(kind)) mon.condition = value;
    if (kind === 'faint') mon.condition = '0 fnt';
    if (kind === '-status') mon.condition = `${mon.condition.split(' ')[0]} ${value}`;
    if (kind === '-curestatus') mon.condition = mon.condition.split(' ')[0];
    if (kind === '-terastallize') mon.tera = value;
    if (['detailschange', '-formechange'].includes(kind)) Object.assign(mon, details(value), {types: null});
    if (kind === '-start' && value === 'typechange') mon.types = extra.split('/');
    if (['-boost', '-unboost', '-setboost'].includes(kind)) mon.boosts[value] = Math.max(-6, Math.min(6,
      kind === '-setboost' ? Number(extra) : (mon.boosts[value] ?? 0) + (kind === '-boost' ? 1 : -1) * Number(extra)));
    if (kind === '-clearboost') mon.boosts = {};
  }

  observe(request, requestId) {
    if (request?.side) {
      if (request.side.id !== this.side) throw new Error('Foreign private request rejected');
      this.lastOwn = request.side.pokemon.map(p => ({ident: name(p.ident), ...details(p.details),
        condition: p.condition, active: p.active, moves: [...p.moves],
        ability: p.ability ?? p.baseAbility ?? null, item: p.item ?? null,
        tera: p.terastallized || null, tera_type: p.teraType ?? null, stats: clone(p.stats ?? {}),
        reviving: Boolean(p.reviving), boosts: clone(this.ownEvents.get(name(p.ident))?.boosts ?? {})}));
    }
    if (!this.lastOwn || this.lastOwn.length !== 6) throw new Error('Expected exactly six own members');
    const own = clone(this.lastOwn);
    if (!request?.side) for (const p of own) Object.assign(p, clone(this.ownEvents.get(p.ident) ?? {}));
    const {mask} = requestActions(request);
    return {schema_version: 'full-showdown-player-v1', side: this.side, request_id: requestId,
      turn: this.turn, phase: request?.wait || !request ? 'wait' : request.forceSwitch ?
        (request.side.pokemon.some(p => p.reviving) ? 'revival' : 'replace') : 'move',
      own, opponent: clone(this.slots), active_moves: (request?.active?.[0]?.moves ?? []).map(m => {
        const id = toID(m.id ?? m.move);
        // Gen9 getMoves emits Recharge as a request control, not a Dex move.
        // These encoder conventions denote a typeless, nondamaging action;
        // its request slot and legality still come entirely from Showdown.
        const move = id === 'recharge' ? {id, basePower: 0, type: '???', category: 'Status'} : Dex.moves.get(id);
        if (id !== 'recharge' && !move.exists) throw new Error(`Unknown Gen9 request move: ${id}`);
        return {id: move.id, pp: m.pp ?? null, maxpp: m.maxpp ?? null, disabled: Boolean(m.disabled),
          base_power: move.basePower, type: move.type, category: move.category};
      }), weather: this.weather, terrain: this.terrain,
      side_conditions: {own: clone(this.sideConditions[this.side]), opponent: clone(this.sideConditions[this.foe])}, mask};
  }
}
