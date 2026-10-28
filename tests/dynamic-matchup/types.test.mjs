import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PublicTypes, canonicalIdent} from '../../packages/dynamic-matchup/types.mjs';

// Minimal injected static catalog: the tracker must never reach for engine data itself.
const CATALOG = {
  Pikachu: ['Electric'], 'Rotom-Wash': ['Electric', 'Water'], Garchomp: ['Dragon', 'Ground'],
  Zoroark: ['Dark'], Ditto: ['Normal'], Corviknight: ['Flying', 'Steel'], Latias: ['Dragon', 'Psychic'],
  Tornadus: ['Flying'], Aegislash: ['Steel', 'Ghost'], 'Aegislash-Blade': ['Steel', 'Ghost'],
  Charizard: ['Fire', 'Flying'], 'Charizard-Mega-X': ['Fire', 'Dragon'], Trevenant: ['Ghost', 'Grass'],
};
const speciesTypes = species => CATALOG[species] ?? null;

const tracker = (...lines) => {
  const t = new PublicTypes(speciesTypes);
  for (const line of lines) t.ingest(line);
  return t;
};
const feed = (t, ...lines) => { for (const line of lines) t.ingest(line); return t; };

const SWITCH_PIKA = '|switch|p1a: Pikachu|Pikachu, L50|100/100';
const SWITCH_ROTOM = '|switch|p2a: Rotom-Wash|Rotom-Wash, L50|100/100';
const SWITCH_CHOMP = '|switch|p2a: Garchomp|Garchomp, L80, M|100/100';
const DOUBLE_SHOCK = (ident, printed) => `|-start|${ident}|typechange|${printed}|[from] move: Double Shock`;
const FORESTS_CURSE = ident => `|-start|${ident}|typeadd|Grass|[from] move: Forest's Curse`;
const TRICK_OR_TREAT = ident => `|-start|${ident}|typeadd|Ghost|[from] move: Trick-or-Treat`;

test('constructor requires an injected speciesTypes function', () => {
  assert.throws(() => new PublicTypes(), TypeError);
  assert.throws(() => new PublicTypes({}), TypeError);
});

test('ident canonicalization strips the position letter and keys by seat + nickname', () => {
  assert.deepEqual(canonicalIdent('p1a: Pikachu'), {seat: 'p1', position: 'a', name: 'Pikachu', key: 'p1:Pikachu'});
  assert.deepEqual(canonicalIdent('p2: Mr. Mime: Jr'), {seat: 'p2', position: null, name: 'Mr. Mime: Jr', key: 'p2:Mr. Mime: Jr'});
  assert.equal(canonicalIdent('garbage'), null);
  const t = tracker(SWITCH_PIKA, DOUBLE_SHOCK('p1a: Pikachu', '???'));
  assert.deepEqual(t.typesFor('p1: Pikachu'), ['???']);
  assert.deepEqual(t.typesFor('p1a: Pikachu'), ['???']);
  assert.equal(t.typesFor('p2a: Pikachu'), null, 'other seat with same nickname is a different mon');
  assert.equal(t.typesFor('garbage'), null);
});

test('Double Shock on both seats yields explicit arrays per seat; ??? is typeless, not unknown', () => {
  const t = tracker(SWITCH_PIKA, SWITCH_ROTOM,
    DOUBLE_SHOCK('p1a: Pikachu', '???'), DOUBLE_SHOCK('p2a: Rotom-Wash', '???/Water'));
  assert.deepEqual(t.typesFor('p1a: Pikachu'), ['???']);
  assert.deepEqual(t.typesFor('p2a: Rotom-Wash'), ['???', 'Water']);
  assert.notEqual(t.typesFor('p1a: Pikachu'), null, 'pure ??? must be distinct from null');
  assert.equal(t.typesFor('p1a: Unseen'), null, 'no override known -> null');
});

test('typesFor returns a fresh copy; caller mutation does not leak into state', () => {
  const t = tracker(SWITCH_PIKA, DOUBLE_SHOCK('p1a: Pikachu', '???'));
  t.typesFor('p1a: Pikachu').push('Fire');
  assert.deepEqual(t.typesFor('p1a: Pikachu'), ['???']);
});

test('typeadd appends a third type; repeated typeadd replaces it; -end typeadd removes it', () => {
  const t = tracker(SWITCH_ROTOM, FORESTS_CURSE('p2a: Rotom-Wash'));
  assert.deepEqual(t.typesFor('p2a: Rotom-Wash'), ['Electric', 'Water', 'Grass']);
  feed(t, TRICK_OR_TREAT('p2a: Rotom-Wash'));
  assert.deepEqual(t.typesFor('p2a: Rotom-Wash'), ['Electric', 'Water', 'Ghost']);
  feed(t, '|-end|p2a: Rotom-Wash|typeadd|[silent]');
  assert.equal(t.typesFor('p2a: Rotom-Wash'), null);
});

test('typechange clears the added type; -end typechange restores species base', () => {
  const t = tracker(SWITCH_ROTOM, FORESTS_CURSE('p2a: Rotom-Wash'),
    '|-start|p2a: Rotom-Wash|typechange|Water|[from] move: Soak');
  assert.deepEqual(t.typesFor('p2a: Rotom-Wash'), ['Water']);
  feed(t, FORESTS_CURSE('p2a: Rotom-Wash'));
  assert.deepEqual(t.typesFor('p2a: Rotom-Wash'), ['Water', 'Grass']);
  feed(t, '|-end|p2a: Rotom-Wash|typechange|[silent]');
  assert.deepEqual(t.typesFor('p2a: Rotom-Wash'), ['Electric', 'Water'], 'Mimicry type reset clears added type');
  feed(t, '|-end|p2a: Rotom-Wash|typeadd|[silent]');
  assert.deepEqual(t.typesFor('p2a: Rotom-Wash'), ['Electric','Water']);
});

test('switch-out resets the outgoing bench state; switch-in and re-entry are reset too', () => {
  const t = tracker(SWITCH_PIKA, DOUBLE_SHOCK('p1a: Pikachu', '???'));
  assert.deepEqual(t.typesFor('p1a: Pikachu'), ['???']);
  feed(t, '|switch|p1a: Charizard|Charizard, L50|100/100');
  assert.equal(t.typesFor('p1a: Pikachu'), null, 'benched mon reverted to species types');
  feed(t, FORESTS_CURSE('p1a: Charizard'));
  assert.deepEqual(t.typesFor('p1a: Charizard'), ['Fire', 'Flying', 'Grass']);
  feed(t, '|drag|p1a: Pikachu|Pikachu, L50|100/100');
  assert.equal(t.typesFor('p1a: Charizard'), null, 'drag also resets the outgoing mon');
  assert.equal(t.typesFor('p1a: Pikachu'), null, 're-entering mon starts clean');
  feed(t, SWITCH_ROTOM);
  assert.equal(t.typesFor('p1a: Pikachu'), null, 'other seat switching does not touch p1');
});

test('Terastallize is ignored here: ordinary types are preserved for the parent to combine', () => {
  const t = tracker(SWITCH_PIKA, DOUBLE_SHOCK('p1a: Pikachu', '???'), '|-terastallize|p1a: Pikachu|Water');
  assert.deepEqual(t.typesFor('p1a: Pikachu'), ['???']);
  const u = tracker(SWITCH_ROTOM, '|-terastallize|p2a: Rotom-Wash|Fairy');
  assert.equal(u.typesFor('p2a: Rotom-Wash'), null, 'Tera alone never becomes an ordinary override');
});

test('forme changes clear overrides and added type; later typeadd uses the new forme species', () => {
  const t = tracker('|switch|p1a: Aegislash|Aegislash, L50|100/100',
    '|-start|p1a: Aegislash|typechange|Water|[from] move: Soak', FORESTS_CURSE('p1a: Aegislash'));
  assert.deepEqual(t.typesFor('p1a: Aegislash'), ['Water', 'Grass']);
  feed(t, '|-formechange|p1a: Aegislash|Aegislash-Blade|[from] ability: Stance Change');
  assert.deepEqual(t.typesFor('p1a: Aegislash'), ['Steel','Ghost']);
  feed(t, FORESTS_CURSE('p1a: Aegislash'));
  assert.deepEqual(t.typesFor('p1a: Aegislash'), ['Steel', 'Ghost', 'Grass']);

  const u = tracker('|switch|p1a: Charizard|Charizard, L100, M|100/100',
    '|-start|p1a: Charizard|typechange|Water|[from] move: Soak',
    '|detailschange|p1a: Charizard|Charizard-Mega-X, L100, M');
  assert.equal(u.typesFor('p1a: Charizard'), null);
  feed(u, FORESTS_CURSE('p1a: Charizard'));
  assert.deepEqual(u.typesFor('p1a: Charizard'), ['Fire', 'Dragon', 'Grass']);
});

test('Transform copies the public target types, its added type, and ignores target Roost/Tera', () => {
  const t = tracker('|switch|p1a: Ditto|Ditto, L50|100/100', SWITCH_CHOMP,
    FORESTS_CURSE('p2a: Garchomp'), '|-terastallize|p2a: Garchomp|Steel',
    '|-transform|p1a: Ditto|p2a: Garchomp|[from] ability: Imposter');
  assert.deepEqual(t.typesFor('p1a: Ditto'), ['Dragon', 'Ground', 'Grass']);
  feed(t, '|switch|p1a: Pikachu|Pikachu, L50|100/100');
  assert.equal(t.typesFor('p1a: Ditto'), null, 'transform reverts on switch-out');

  const u = tracker('|switch|p1a: Ditto|Ditto, L50|100/100', '|switch|p2a: Corviknight|Corviknight, L50|100/100',
    '|-singleturn|p2a: Corviknight|move: Roost', '|-transform|p1a: Ditto|p2a: Corviknight');
  assert.deepEqual(u.typesFor('p2a: Corviknight'), ['Steel']);
  assert.deepEqual(u.typesFor('p1a: Ditto'), ['Flying', 'Steel'], 'copies typeWas, not the roosted view');

  const v = tracker('|switch|p1a: Ditto|Ditto, L50|100/100');
  assert.throws(() => v.ingest('|-transform|p1a: Ditto|p2a: Mystery'), /Unknown public Transform target/);
  assert.throws(() => v.ingest('|-transform|p1a: Ditto|'), /Malformed -transform/);
});

test('Reflect Type copies public base types, drops ???, and uses Normal when only an added type remains', () => {
  const t = tracker('|switch|p1a: Latias|Latias, L50|100/100', SWITCH_CHOMP,
    '|-start|p1a: Latias|typechange|[from] move: Reflect Type|[of] p2a: Garchomp');
  assert.deepEqual(t.typesFor('p1a: Latias'), ['Dragon', 'Ground']);

  const u = tracker('|switch|p1a: Latias|Latias, L50|100/100', '|switch|p2a: Pikachu|Pikachu, L50|100/100',
    DOUBLE_SHOCK('p2a: Pikachu', '???'), FORESTS_CURSE('p2a: Pikachu'),
    '|-start|p1a: Latias|typechange|[from] move: Reflect Type|[of] p2a: Pikachu');
  assert.deepEqual(u.typesFor('p2a: Pikachu'), ['???', 'Grass']);
  assert.deepEqual(u.typesFor('p1a: Latias'), ['Normal', 'Grass']);

  const v = tracker('|switch|p1a: Latias|Latias, L50|100/100', '|switch|p2a: Corviknight|Corviknight, L50|100/100',
    '|-singleturn|p2a: Corviknight|move: Roost',
    '|-start|p1a: Latias|typechange|[from] move: Reflect Type|[of] p2a: Corviknight');
  assert.deepEqual(v.typesFor('p1a: Latias'), ['Steel'], 'getTypes(true) on a roosted target excludes Flying');
  feed(v, '|turn|2');
  assert.deepEqual(v.typesFor('p1a: Latias'), ['Steel'], 'copied types persist after target Roost expiry');

  const w = tracker('|switch|p1a: Latias|Latias, L50|100/100');
  assert.throws(() => w.ingest('|-start|p1a: Latias|typechange|[from] move: Reflect Type|[of] p2a: Mystery'), /Unknown public Reflect Type target/);
  assert.throws(() => w.ingest('|-start|p1a: Latias|typechange|[from] move: Reflect Type'), /without \[of\]/);
});

test('Reflect Type on a disguised Illusion user can only copy the public appearance (documented limit)', () => {
  const t = tracker('|switch|p1a: Latias|Latias, L50|100/100', SWITCH_CHOMP,
    '|-start|p1a: Latias|typechange|[from] move: Reflect Type|[of] p2a: Garchomp',
    '|replace|p2a: Zoroark|Zoroark, L80, M');
  assert.deepEqual(t.typesFor('p1a: Latias'), ['Dragon', 'Ground'], 'public evidence never reveals Dark');
});

test('Illusion replace carries appearance overrides to the revealed identity and restores the disguise', () => {
  const t = tracker(SWITCH_PIKA, SWITCH_CHOMP, FORESTS_CURSE('p2a: Garchomp'));
  assert.deepEqual(t.typesFor('p2a: Garchomp'), ['Dragon', 'Ground', 'Grass']);
  feed(t, '|replace|p2a: Zoroark|Zoroark, L80, M');
  assert.deepEqual(t.typesFor('p2a: Zoroark'), ['Dark', 'Grass']);
  assert.equal(t.typesFor('p2a: Garchomp'), null, 'disguise restored to its prior (clean) state');
  feed(t, '|-start|p2a: Zoroark|typechange|Water|[from] move: Soak');
  assert.deepEqual(t.typesFor('p2a: Zoroark'), ['Water']);
  feed(t, '|switch|p2a: Garchomp|Garchomp, L80, M|100/100');
  assert.equal(t.typesFor('p2a: Zoroark'), null, 'revealed identity resets on switch-out');
  assert.equal(t.typesFor('p2a: Garchomp'), null);
});

test('Illusion replace keeps the real disguise species so later events resolve without fallback', () => {
  const t = tracker(SWITCH_CHOMP, '|switch|p2a: Zoroark|Zoroark, L80, M|100/100',
    SWITCH_CHOMP, '|replace|p2a: Zoroark|Zoroark, L80, M', '|switch|p2a: Garchomp|Garchomp, L80, M|100/100',
    FORESTS_CURSE('p2a: Garchomp'));
  assert.deepEqual(t.typesFor('p2a: Garchomp'), ['Dragon', 'Ground', 'Grass']);
});

test('Roost removes Flying until the next turn; pure Flying becomes Normal; expiry on switch-out', () => {
  const t = tracker('|switch|p1a: Corviknight|Corviknight, L50|100/100', '|-singleturn|p1a: Corviknight|move: Roost');
  assert.deepEqual(t.typesFor('p1a: Corviknight'), ['Steel']);
  feed(t, '|upkeep', '|turn|3');
  assert.equal(t.typesFor('p1a: Corviknight'), null);

  const u = tracker('|switch|p1a: Tornadus|Tornadus, L50|100/100', FORESTS_CURSE('p1a: Tornadus'),
    '|-singleturn|p1a: Tornadus|move: Roost');
  assert.deepEqual(u.typesFor('p1a: Tornadus'), ['Normal', 'Grass']);
  feed(u, '|switch|p1a: Pikachu|Pikachu, L50|100/100');
  assert.equal(u.typesFor('p1a: Tornadus'), null);
});

test('fallbackSpecies fills in a species never seen publicly; without it the array is unknown', () => {
  const t = tracker(FORESTS_CURSE('p1a: Mystery'));
  assert.equal(t.typesFor('p1a: Mystery'), null);
  assert.deepEqual(t.typesFor('p1a: Mystery', 'Garchomp'), ['Dragon', 'Ground', 'Grass']);
  assert.equal(t.typesFor('p1a: Mystery', 'NotInCatalog'), null);
  assert.equal(t.typesFor('p1a: Nobody', 'Garchomp'), null, 'fallback never fabricates an override');
});

test('rejects private/unprojected lines and malformed explicit types', () => {
  const t = tracker(SWITCH_PIKA);
  assert.throws(() => t.ingest('|request|{"side":{}}'), /Private\/unprojected/);
  assert.throws(() => t.ingest('|split|p1'), /Private\/unprojected/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typechange|Electric/Bogus'), /Malformed explicit types/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typechange|Electric/'), /Malformed explicit types/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typechange|Stellar'), /Malformed explicit types/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typechange|Fire/Water/Grass/Ice'), /Malformed explicit types/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typechange'), /without explicit types/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typechange||[from] move: Soak'), /without explicit types/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typeadd|???'), /Malformed typeadd/);
  assert.throws(() => t.ingest('|-start|p1a: Pikachu|typeadd|'), /Malformed typeadd/);
  assert.equal(t.typesFor('p1a: Pikachu'), null, 'rejected lines leave state untouched');
});

test('unrelated and non-protocol lines are ignored without side effects', () => {
  const t = tracker(SWITCH_PIKA, DOUBLE_SHOCK('p1a: Pikachu', '???'));
  feed(t, '>battle-gen9ou-1', '|move|p1a: Pikachu|Thunderbolt|p2a: Garchomp', '|-damage|p2a: Garchomp|50/100',
    '|-start|p1a: Pikachu|Substitute', '|-end|p1a: Pikachu|Substitute', '|faint|p2a: Garchomp', '', 'plain text');
  assert.deepEqual(t.typesFor('p1a: Pikachu'), ['???']);
});


test('Reflect Type copies a Tera target while Transform copies its ordinary types', () => {
  const t = tracker('|switch|p1a: Latias|Latias, L50|100/100', SWITCH_CHOMP,
    '|-terastallize|p2a: Garchomp|Steel',
    '|-start|p1a: Latias|typechange|[from] move: Reflect Type|[of] p2a: Garchomp');
  assert.deepEqual(t.typesFor('p1a: Latias'), ['Steel']);
});

test('Tera refusal messages never replace ordinary types and Tera survives reentry', () => {
  const t = tracker(SWITCH_PIKA, '|-terastallize|p1a: Pikachu|Electric', DOUBLE_SHOCK('p1a: Pikachu','Electric'));
  assert.equal(t.typesFor('p1a: Pikachu'), null);
  feed(t, '|switch|p1a: Ditto|Ditto, L50|100/100', SWITCH_PIKA.replace('Pikachu, L50','Pikachu, L50, tera:Electric'), DOUBLE_SHOCK('p1a: Pikachu','Electric'));
  assert.equal(t.typesFor('p1a: Pikachu'), null);
});
