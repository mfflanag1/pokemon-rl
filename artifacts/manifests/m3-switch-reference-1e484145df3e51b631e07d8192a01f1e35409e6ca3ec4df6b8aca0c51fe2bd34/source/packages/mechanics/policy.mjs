// R6 inventories a conservative envelope. These are implementation obligations,
// never assertions that the accelerated engine already implements a callback.
export const FAMILIES = {
  'move-resolution': ['ENG-MOVE', 'accuracy, immunity, PP, damage, secondary effects, fail/no-target, RNG draw order'],
  'ability-events': ['ENG-ABILITY', 'activation, suppression, event priority, source/target roles, switching'],
  'item-events': ['ENG-ITEM', 'held/absent, removal, consumption, restoration, transfer, unremovable items'],
  'species-state': ['ENG-FORM', 'stats, types, HP, identity, cosmetic forms, permanent and temporary changes'],
  'conditions': ['ENG-CONDITION', 'installation, refresh, duration, residual order, removal, faint/switch cleanup'],
  'callback-dispatch': ['ENG-EVENT', 'handler collection, ordering, cancellation, return-value semantics, nested events'],
  'move-calling': ['ENG-CALL', 'selection pool, recursion guards, copied PP, external calls, failure branches'],
  'identity-copy': ['ENG-COPY', 'Transform, ability copy/swap, item identity, persistence after switching'],
  'tera': ['ENG-TERA', 'once-per-side, request legality, type/STAB, Stellar, Ogerpon and Terapagos forms'],
  'battle-core': ['ENG-CORE', 'turn order, phases, switches, simultaneous fainting, targeting, PRNG, terminal rules'],
};

export const DOMAIN_FAMILY = {moves: 'move-resolution', abilities: 'ability-events', items: 'item-events',
  forms: 'species-state', conditions: 'conditions', core: 'battle-core'};

// Unknown sink classes fail generation. A dynamic expression is bounded by the
// entire pinned domain, not incorrectly treated as a literal or as unreachable.
export const PATHWAY_RULES = {
  'source-owned-call': {bound: 'pinned-runtime', family: 'battle-core'},
  'source-owned-construction': {bound: 'pinned-runtime', family: 'battle-core'},
  'move-execution': {bound: 'moves', family: 'move-calling'},
  'active-move-construction': {bound: 'moves', family: 'move-calling'},
  'transform-copy': {bound: 'forms+moves+abilities', family: 'identity-copy'},
  'form-change': {bound: 'forms', family: 'species-state'},
  'ability-change': {bound: 'abilities', family: 'identity-copy'},
  'item-transfer': {bound: 'items', family: 'item-events'},
  'item-consumption': {bound: 'items', family: 'item-events'},
  'condition-install': {bound: 'conditions+moves+abilities+items+core', family: 'conditions'},
  'event-dispatch': {bound: 'pinned-source-callbacks', family: 'callback-dispatch'},
  'callback-invocation': {bound: 'pinned-source-callbacks', family: 'callback-dispatch'},
  'computed-call': {bound: 'pinned-source-callbacks', family: 'callback-dispatch'},
  'dex-resolution': {bound: 'moves+abilities+items+forms+conditions', family: 'callback-dispatch'},
  'identity-state-write': {bound: 'moves+abilities+items+forms', family: 'identity-copy'},
};

export function classifyPathway(p) {
  const rule = Object.hasOwn(PATHWAY_RULES, p.kind) ? PATHWAY_RULES[p.kind] : null;
  if (!rule) throw new Error(`Unclassified pathway kind: ${p.kind}`);
  return {...p, classification: p.arguments[0]?.literal !== null && p.arguments[0]?.literal !== undefined ?
    'literal-with-conservative-envelope' : 'dynamic-conservative-envelope', ...rule};
}

export const INDIRECT_AUDIT = [
  ['moves', 'sleeptalk', 'Selects eligible current move slots; excludes banned callers and charging moves.'],
  ['moves', 'transform', 'Copies target species, move slots, boosts and ability with special exclusions; HP/level are retained.'],
  ['abilities', 'imposter', 'Calls Transform on the opposing active Pokemon at entry.'],
  ['abilities', 'trace', 'Copies an eligible opposing ability; blacklist and later retry matter.'],
  ['abilities', 'dancer', 'Core runMove repeats the current successful dance with external-call guard.'],
  ['abilities', 'magicbounce', 'Reflects the current eligible move with hasBounced guard.'],
  ['moves', 'struggle', 'Core request fallback when no move is usable; not an initial set move.'],
  ...['metronome', 'assist', 'copycat', 'mimic', 'sketch', 'instruct', 'mefirst', 'mirrormove', 'naturepower',
    'skillswap', 'entrainment', 'roleplay', 'simplebeam', 'worryseed'].map(id =>
    ['moves', id, 'Not an observed/declarative initial move. Retained in the conservative envelope; no global unreachability proof is claimed.']),
  ['moves', 'recycle', 'Restores the previously consumed item, with restrictions.'],
  ['moves', 'trick', 'Transfers existing held items subject to removal/transfer restrictions.'],
  ['moves', 'switcheroo', 'Transfers existing held items subject to removal/transfer restrictions.'],
  ['abilities', 'harvest', 'Restores an eligible consumed berry with chance/weather rules.'],
  ['abilities', 'zerotohero', 'Permanent Palafin form change after switching out.'],
  ['abilities', 'disguise', 'Form and damage handling; not ordinary permanent ability replacement.'],
  ['abilities', 'iceface', 'Form, physical damage, and weather restoration rules.'],
  ['abilities', 'terashift', 'Terapagos entry form; Tera adds a further core form/ability transition.'],
];
