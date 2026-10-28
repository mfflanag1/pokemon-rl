/** Public-only dynamic ordinary types, with static species metadata injected.
 * No engine objects, private requests or IO. Null means no dynamic override;
 * ??? is a known neutral typeless component. Tera is stored only to interpret
 * type-copy/removal messages; the parent applies Tera defense separately.
 * Switch-out resets temporary types and addedType, while Tera persists.
 * Pinned setType clears addedType. Transform copies pre-Tera/pre-Roost types;
 * Reflect Type copies current defense (including Tera) and the added type.
 * Unknown copy targets fail closed. Illusion remains a public-appearance
 * limitation until revealed; no hidden information resolves it here.
 * Roost expires at public upkeep/next turn, the observable end-turn boundary.
 */

const TYPE_NAMES = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison',
  'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy'];
const KNOWN_TYPES = new Set([...TYPE_NAMES, '???']);
const MAX_TYPES = 3; // two base types plus one added type

const clone = value => structuredClone(value);
const toID = text => (text ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const effectId = text => toID((text ?? '').replace(/^(move|ability|item): /, ''));
const speciesOf = details => {
  const species = (details ?? '').split(',')[0].trim();
  return species || null;
};

/** `p1a: Nick` -> {seat:'p1', position:'a', name:'Nick', key:'p1:Nick'}; `p1: Nick` also accepted. */
export const canonicalIdent = ident => {
  if (typeof ident !== 'string') return null;
  const match = /^(p[1-4])([a-d]?)(?:: ?(.*))?$/s.exec(ident.trim());
  if (!match) return null;
  const name = (match[3] ?? '').trim();
  return {seat: match[1], position: match[2] || null, name, key: `${match[1]}:${name}`};
};

const parseTypes = (text, line) => {
  if (typeof text !== 'string' || text === '') throw new Error(`Malformed explicit types: ${line}`);
  const types = text.split('/');
  if (types.length > MAX_TYPES || types.some(type => !KNOWN_TYPES.has(type))) {
    throw new Error(`Malformed explicit types "${text}": ${line}`);
  }
  return types;
};

const parseAdded = (text, line) => {
  if (typeof text !== 'string' || !KNOWN_TYPES.has(text) || text === '???') {
    throw new Error(`Malformed typeadd type "${text}": ${line}`);
  }
  return text;
};

const parseLine = line => {
  if (typeof line !== 'string' || !line.startsWith('|')) return null;
  const parts = line.split('|');
  parts.shift();
  const kind = parts.shift() ?? '';
  const args = [], kwargs = {};
  for (const part of parts) {
    const kw = /^\[(\w+)\](?: (.*))?$/s.exec(part);
    if (kw) kwargs[kw[1]] = kw[2] ?? '';
    else args.push(part);
  }
  return {kind, args, kwargs};
};

const fresh = species => ({species, forme: null, transformSpecies: null,
  base: null, added: null, roost: false, transformed: false, tera: null});

export class PublicTypes {
  constructor(speciesTypes) {
    if (typeof speciesTypes !== 'function') throw new TypeError('PublicTypes requires a speciesTypes(species) function');
    this.speciesTypes = speciesTypes;
    this.records = new Map(); // key -> record
    this.active = new Map(); // `${seat}${position}` -> key currently shown in that slot
    this.entries = new Map(); // slot -> {key, prior} snapshot taken at switch-in (Illusion repair)
  }

  record(key) {
    if (!this.records.has(key)) this.records.set(key, fresh(null));
    return this.records.get(key);
  }

  ingest(line) {
    const msg = parseLine(line);
    if (!msg) return;
    const {kind, args, kwargs} = msg;
    if (kind === 'request' || kind === 'split') throw new Error(`Private/unprojected line rejected in public type tracker: ${kind}`);
    if (kind === 'turn' || kind === 'upkeep') {
      for (const rec of this.records.values()) rec.roost = false;
      return;
    }
    const who = canonicalIdent(args[0]);
    if (!who) return;
    const slot = `${who.seat}${who.position ?? 'a'}`;

    switch (kind) {
      case 'switch':
      case 'drag': {
        const outgoing = this.active.get(slot);
        if (outgoing && outgoing !== who.key) {
          // Outgoing bench state: clearVolatile -> setSpecies(baseSpecies) resets ordinary types.
          const old = this.records.get(outgoing);
          this.records.set(outgoing, {...fresh(old?.species ?? null), tera: old?.tera ?? null});
        }
        const prior = this.records.has(who.key) ? clone(this.records.get(who.key)) : null;
        this.entries.set(slot, {key: who.key, prior});
        // getFullDetails appends tera for the actual entrant, even under Illusion.
        const tera = /(?:^|, )tera:([^,]+)/.exec(args[1] ?? '')?.[1] ?? null;
        this.records.set(who.key, {...fresh(speciesOf(args[1]) ?? prior?.species ?? null), tera});
        this.active.set(slot, who.key);
        return;
      }
      case 'replace': {
        const disguiseKey = this.active.get(slot) ?? null;
        const entry = this.entries.get(slot);
        const carried = disguiseKey ? this.records.get(disguiseKey) : null;
        if (disguiseKey && disguiseKey !== who.key) {
          if (entry?.key === disguiseKey && entry.prior) this.records.set(disguiseKey, clone(entry.prior));
          else this.records.delete(disguiseKey);
        }
        this.records.set(who.key, {...(carried ? clone(carried) : fresh(null)),
          species: speciesOf(args[1]) ?? carried?.species ?? null});
        this.active.set(slot, who.key);
        this.entries.set(slot, {key: who.key, prior: null});
        return;
      }
      case 'detailschange':
      case '-formechange': {
        const rec = this.record(who.key);
        const species = speciesOf(args[1]);
        if (kind === 'detailschange') { rec.species = species ?? rec.species; rec.forme = null; } else rec.forme = species ?? rec.forme;
        rec.base = null; rec.added = null; rec.transformSpecies = null; rec.transformed = false;
        return;
      }
      case '-terastallize':
        this.record(who.key).tera = args[1];
        return;
      case '-start':
      case '-end': {
        const what = args[1];
        if (what !== 'typechange' && what !== 'typeadd') return;
        const rec = this.record(who.key);
        if (kind === '-end') {
          if (what === 'typechange') {
            // Pinned Mimicry restores baseSpecies types through setType, which
            // also clears addedType. A temporary forme need not be baseSpecies.
            rec.base = rec.species ? this.speciesTypes(rec.species) : null;
            rec.added = null; rec.transformSpecies = null; rec.transformed = false;
          } else rec.added = null;
          return;
        }
        if (what === 'typeadd') { rec.added = parseAdded(args[2], line); return; }
        // These two moves emit a typechange even when Tera makes setType refuse.
        if (rec.tera && ['doubleshock','burnup'].includes(effectId(kwargs.from))) return;
        if (args[2] !== undefined && args[2] !== '') {
          rec.base = parseTypes(args[2], line);
          rec.added = null;
          return;
        }
        if (effectId(kwargs.from) === 'reflecttype') { this.reflectType(rec, kwargs.of, line); return; }
        throw new Error(`Malformed typechange without explicit types: ${line}`);
      }
      case '-transform': {
        const target = canonicalIdent(args[1]);
        if (!target) throw new Error(`Malformed -transform target: ${line}`);
        const rec = this.record(who.key);
        const targetRec = this.records.get(target.key) ?? null;
        // pokemon.ts:1294-1296 — copy pre-Tera base types, Roost's typeWas, and addedType.
        const base = this.currentBase(targetRec, null, false);
        if (!base) throw new Error('Unknown public Transform target');
        rec.transformSpecies = targetRec?.transformSpecies ?? targetRec?.forme ?? targetRec?.species ?? null;
        rec.base = base; rec.added = targetRec?.added ?? null; rec.transformed = true;
        return;
      }
      case '-singleturn':
        if (effectId(args[1]) === 'roost') this.record(who.key).roost = true;
        return;
      default:
        return;
    }
  }

  reflectType(rec, ofIdent, line) {
    const target = canonicalIdent(ofIdent);
    if (!target) throw new Error(`Reflect Type typechange without [of] target: ${line}`);
    const targetRec = this.records.get(target.key) ?? null;
    const targetBase = targetRec?.tera && targetRec.tera !== 'Stellar' ? [targetRec.tera] :
      this.currentBase(targetRec, null, true);
    if (!targetBase) throw new Error('Unknown public Reflect Type target');
    let types = targetBase.filter(type => type !== '???');
    if (!types.length) {
      if (!targetRec?.added) return; // engine returns false: move fails, no -start would be emitted
      types = ['Normal'];
    }
    rec.base = types;
    rec.added = targetRec?.added ?? null;
  }

  /** Base types (without added type) for a record; null when not publicly derivable. */
  currentBase(rec, fallbackSpecies, applyRoost) {
    const species = rec?.transformSpecies ?? rec?.forme ?? fallbackSpecies ?? rec?.species ?? null;
    let base = rec?.base ?? (species ? this.speciesTypes(species) : null);
    if (!Array.isArray(base) || !base.length) return null;
    base = [...base];
    if (applyRoost && rec?.roost) {
      base = base.filter(type => type !== 'Flying');
      if (!base.length) base = ['Normal'];
    }
    return base;
  }

  /** null = no dynamic override known; otherwise the current ordinary type array. */
  typesFor(ident, fallbackSpecies = null) {
    const who = canonicalIdent(ident);
    if (!who) return null;
    const rec = this.records.get(who.key);
    if (!rec || !(rec.base || rec.added || rec.roost || rec.transformed || rec.forme)) return null;
    const base = this.currentBase(rec, fallbackSpecies, true);
    if (!base) return null;
    if (rec.added && !base.includes(rec.added)) base.push(rec.added);
    return base;
  }
}
