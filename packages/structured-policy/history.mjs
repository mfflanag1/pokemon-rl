import {readFileSync} from 'node:fs';

export const HISTORY_CONTRACT = JSON.parse(readFileSync(new URL('./contract.json', import.meta.url)));
const catalog = JSON.parse(readFileSync(new URL('../../contracts/player-features-v1/catalog.json', import.meta.url)));
const indexes = Object.fromEntries(Object.entries(catalog.domains).map(([key, values]) => [key, new Map(values.map((value, i) => [value, i]))]));
const vocabulary = Object.fromEntries(['event_kind', 'event_side', 'status', 'event_aux'].map(key => [key, new Map(HISTORY_CONTRACT[key].map((value, i) => [value, i]))]));
const id = value => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const effect = value => id(String(value ?? '').replace(/^(move|ability|item): /, ''));
const species = value => String(value ?? '').split(',')[0];
const seat = value => /^p[12](?:a)?:/.test(value ?? '') ? value.slice(0, 2) : null;
const field = (domain, value) => value === null || value === undefined ? 0 : value === '' ? 1 :
  indexes[domain].get(catalog.names[domain][String(value).toLowerCase()]) ?? 0;
const enumId = (domain, value) => vocabulary[domain].get(value) ?? 0;

function health(condition) {
  if (typeof condition !== 'string') return [0, 0];
  if (condition.includes('fnt') || condition.split(' ')[0] === '0') return [0, 1];
  const match = /^(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)(?: |$)/.exec(condition);
  return match && Number(match[2]) > 0 ? [Math.min(1, Number(match[1]) / Number(match[2])), 1] : [0, 0];
}

/** Only an already split player channel is accepted. Requests, end payloads,
 * raw strings, and unrecognized protocol fields never become policy features. */
export class PublicHistory {
  constructor(side) {
    if (!['p1', 'p2'].includes(side)) throw new Error('Invalid history side');
    this.side = side;
    this.cursor = 0;
    this.turn = 0;
    this.events = [];
    this.totalEvents = 0;
    this.activeSpecies = {p1: null, p2: null};
    this.moveCounts = [new Map(), new Map()];
  }

  relative(value) {
    // This relative-seat enum uses 0 for either unknown or inapplicable seats.
    // It is separate from the R5 entity convention (0 unknown, 1 absent).
    const side = seat(value);
    return side === null ? 0 : side === this.side ? 1 : 2;
  }

  ingest(line) {
    if (typeof line !== 'string') throw new Error('Expected protocol line');
    const [, rawKind, actor, value, extra, ...tags] = line.split('|');
    if (rawKind === 'split') throw new Error('Unsplit protocol rejected');
    if (rawKind === 'request') return;
    const kind = (rawKind ?? '').replace(/^-/, '');
    if (!vocabulary.event_kind.has(kind)) return;
    if (kind === 'turn') {
      const turn = Number(actor);
      if (!Number.isSafeInteger(turn) || turn < this.turn) throw new Error('Invalid history turn order');
      this.turn = turn;
    }
    const actorSide = this.relative(actor), targetSide = kind === 'move' ? this.relative(extra) : 0;
    if (['switch', 'drag', 'replace', 'detailschange', 'formechange'].includes(kind) && seat(actor)) {
      this.activeSpecies[seat(actor)] = species(value);
    }
    const ids = [enumId('event_kind', kind), actorSide, targetSide, 1,
      actorSide ? field('species', this.activeSpecies[seat(actor)]) : 1, 1, 1, 1, 1, 1];
    let hp = [0, 0], magnitude = 0;
    if (kind === 'move') {
      ids[3] = field('moves', value);
      if (actorSide) {
        const counts = this.moveCounts[actorSide - 1];
        counts.set(ids[3], (counts.get(ids[3]) ?? 0) + 1);
      }
    }
    if (['switch', 'drag', 'replace'].includes(kind)) hp = health(extra);
    if (['damage', 'heal', 'sethp'].includes(kind)) hp = health(value);
    if (kind === 'faint') { hp = [0, 1]; ids[8] = enumId('status', 'fnt'); }
    if (kind === 'status' || kind === 'curestatus') ids[8] = enumId('status', value);
    if (['boost', 'unboost', 'setboost'].includes(kind)) {
      ids[9] = enumId('event_aux', effect(value));
      magnitude = Number(extra);
      if (!Number.isFinite(magnitude) || Math.abs(magnitude) > 12) throw new Error('Invalid public boost amount');
      if (kind === 'unboost') magnitude *= -1;
    }
    if (kind === 'ability') ids[5] = field('abilities', value);
    if (kind === 'item' || kind === 'enditem') ids[6] = field('items', value);
    if (kind === 'terastallize') ids[7] = field('types', value);
    if (['start', 'end', 'sidestart', 'sideend', 'cant', 'activate', 'prepare'].includes(kind)) {
      ids[9] = enumId('event_aux', effect(value));
      if (value?.startsWith('move: ')) ids[3] = field('moves', value.slice(6));
      if (value?.startsWith('ability: ')) ids[5] = field('abilities', value.slice(9));
      if (value?.startsWith('item: ')) ids[6] = field('items', value.slice(6));
      if (['typechange', 'typeadd'].includes(value)) ids[7] = field('types', extra?.split('/')[0]);
      if (kind === 'cant') ids[8] = enumId('status', value);
    }
    if (['weather', 'fieldstart', 'fieldend'].includes(kind)) ids[9] = enumId('event_aux', effect(actor));
    for (const tag of [extra, ...tags]) {
      if (tag?.startsWith('[from] ability: ')) ids[5] = field('abilities', tag.slice(16));
      if (tag?.startsWith('[from] item: ')) ids[6] = field('items', tag.slice(13));
    }
    this.events.push({ids, numeric: [this.turn, ...hp, magnitude]});
    this.totalEvents++;
    if (this.events.length > HISTORY_CONTRACT.window) this.events.shift();
  }

  consume(protocol) {
    if (!Array.isArray(protocol) || protocol.length < this.cursor) throw new Error('History protocol cannot rewind');
    while (this.cursor < protocol.length) this.ingest(protocol[this.cursor++]);
    return this.snapshot();
  }

  snapshot() {
    return {schema_version: HISTORY_CONTRACT.schema_version, side: this.side,
      total_events: this.totalEvents, events: structuredClone(this.events),
      public_move_counts: this.moveCounts.map(counts => [...counts].sort((a, b) => a[0] - b[0]))};
  }
}
