import assert from 'node:assert/strict';
import {readFileSync, readdirSync} from 'node:fs';
import path from 'node:path';
import {Dex, Teams, seedFor, describe} from '../reference-node/engine.mjs';
import {ROOT, UPSTREAM, pin, fileHash, hashJSON} from '../reference-node/provenance.mjs';
import {sourceCatalog} from '../corpus/catalog.mjs';
import {parseSource, tableEntries, methodNodes, stringLiterals, callbacks, pathways} from './source-index.mjs';
import {DOMAIN_FAMILY, FAMILIES, INDIRECT_AUDIT, classifyPathway} from './policy.mjs';

export const INPUTS = {
  frequencies: 'artifacts/runs/r3-gen9-random-100k-v2/inventory-final/frequencies.json',
  r3_report: 'artifacts/reports/r3-corpus.json',
  r4_index: 'artifacts/runs/r4-fixtures-v1/index.json',
  r5_probes: 'artifacts/runs/r5-features-v1/live-probes.json',
};
const FREQUENCIES_HASH = '869d59aa23be70d0c87376f56744ef62824d58e80310e56a37adad52ebd909c3';
const TABLES = {moves: ['data/moves.ts', 'Moves', 'Moves'], abilities: ['data/abilities.ts', 'Abilities', 'Abilities'],
  items: ['data/items.ts', 'Items', 'Items'], forms: ['data/pokedex.ts', 'Pokedex', 'Pokedex'],
  conditions: ['data/conditions.ts', 'Conditions', 'Conditions']};
const sorted = values => [...new Set(values)].sort();
const read = file => JSON.parse(readFileSync(path.join(ROOT, file), 'utf8'));

export function runtimeCallbacks(value, prefix = [], result = []) {
  for (const [key, child] of Object.entries(value ?? {})) {
    const route = [...prefix, key];
    if (typeof child === 'function') result.push(route.join('.'));
    else if (child && typeof child === 'object') runtimeCallbacks(child, route, result);
  }
  return result.sort();
}

export function enumerateInitialForms(catalog, generator = Teams.getGenerator(pin.format_id, seedFor('r6-generator-alternatives-v1'))) {
  const originalSample = generator.sample;
  const originalPrng = generator.prng;
  const randomMethods = Object.fromEntries(['random', 'randomChance', 'sampleNoReplace'].map(name => [name, generator[name]]));
  const forbiddenRandom = () => { throw new Error('getForme used randomness outside its enumerated sample call'); };
  generator.prng = new Proxy(originalPrng, {get: forbiddenRandom});
  for (const name of Object.keys(randomMethods)) generator[name] = forbiddenRandom;
  const result = {};
  try {
    for (const id of Object.keys(catalog.entries).sort()) {
      let options, calls = 0;
      generator.sample = pool => { calls++; options = [...pool]; return pool[0]; };
      const first = generator.getForme(Dex.species.get(id));
      assert.ok(calls <= 1, 'getForme gained multiple sampling sites: update the enumeration');
      const values = [first];
      for (let i = 1; i < (options?.length ?? 0); i++) {
        calls = 0;
        generator.sample = pool => { calls++; assert.deepEqual(pool, options); return pool[i]; };
        values.push(generator.getForme(Dex.species.get(id)));
        assert.equal(calls, 1);
      }
      result[id] = sorted(values.map(name => { const s = Dex.species.get(name); assert.ok(s.exists); return s.id; }));
    }
  } finally {
    generator.sample = originalSample; generator.prng = originalPrng;
    Object.assign(generator, randomMethods);
  }
  return result;
}

export function buildManifest() {
  assert.equal(fileHash(path.join(ROOT, INPUTS.frequencies)), FREQUENCIES_HASH, 'R3 inventory changed');
  const engine = describe(), catalog = sourceCatalog(), frequencies = read(INPUTS.frequencies);
  assert.equal(hashJSON(catalog), '429570e57cb4e9f6442283d4eb2d929ed5f52f63d70b5290cdc61f0454c22b46');
  const dex = Dex.forFormat(pin.format_id);
  assert.equal(dex.currentMod, 'base'); assert.equal(dex.gen, 9);
  const generator = parseSource('data/random-battles/gen9/teams.ts'), methods = methodNodes(generator);
  const initialForms = enumerateInitialForms(catalog);
  const declared = {moves: new Set(), abilities: new Set(), items: new Set(['']), forms: new Set(Object.values(initialForms).flat()), conditions: new Set()};
  for (const e of Object.values(catalog.entries)) for (const set of e.sets) {
    set.moves.forEach(id => declared.moves.add(id)); set.abilities.forEach(id => declared.abilities.add(id));
  }
  // This item set is a candidate set, not a complete branch-feasibility proof.
  // The full item domain is retained below, so an overlooked expression cannot
  // silently remove a required mechanic from the support gate.
  for (const name of ['getPriorityItem', 'getItem']) for (const literal of stringLiterals(methods.get(name))) {
    const item = dex.items.get(literal); if (item.exists) declared.items.add(item.id);
  }
  for (const id of Object.keys(catalog.entries)) for (const name of dex.species.get(id).requiredItems ?? []) declared.items.add(dex.items.get(name).id);
  for (const berry of ['Aguav', 'Figy', 'Iapapa', 'Mago', 'Wiki']) declared.items.add(dex.items.get(`${berry} Berry`).id);
  const observed = Object.fromEntries(Object.keys(TABLES).map(domain => [domain,
    Object.fromEntries(Object.entries(frequencies[domain] ?? {}).map(([id, count]) => [id === '<none>' ? '' : id, count]))]));
  for (const domain of ['moves', 'abilities', 'forms']) for (const id of Object.keys(observed[domain])) assert.ok(declared[domain].has(id), `Unexplained initial ${domain}:${id}`);
  const sources = {}, entries = [], dispatch = [];
  function addSource(source) { sources[source.file] = source.ref(source.ast).file_sha256; }
  addSource(generator);
  function addPathways(list, owner) {
    const ids = [];
    for (const raw of list) {
      const p = classifyPathway(raw), id = hashJSON([owner, p.source.file, p.source.start_byte, p.source.end_byte, p.kind]);
      ids.push(id); dispatch.push({id, owner, ...p});
    }
    return ids;
  }
  for (const [domain, [file, variable, table]] of Object.entries(TABLES)) {
    const source = parseSource(file); addSource(source);
    for (const [id, node] of tableEntries(source, variable)) {
      const cbs = callbacks(source, node), data = dex.data[table][id];
      assert.deepEqual(cbs.map(c => c.path).sort(), runtimeCallbacks(data), `Unindexed callbacks: ${domain}:${id}`);
      const key = `${domain}:${id}`, paths = addPathways(pathways(source, node), key);
      const families = sorted([DOMAIN_FAMILY[domain], ...(cbs.length ? ['callback-dispatch'] : []), ...(data.condition ? ['conditions'] : []),
        ...pathways(source, node).map(p => classifyPathway(p).family)]);
      if ((domain === 'moves' && ['terablast', 'terastarstorm'].includes(id)) ||
          (domain === 'forms' && /^(ogerpon|terapagos)/.test(id))) families.push('tera');
      entries.push({key, domain, id, source: source.ref(node), callback_paths: cbs.map(c => ({path: c.path, source: c.source})),
        data_sha256: hashJSON(JSON.parse(JSON.stringify(data, (_, v) => typeof v === 'function' ? '<callback>' : v))),
        observed_count: observed[domain][id] ?? 0,
        initial_class: observed[domain][id] ? 'observed-initial' : declared[domain].has(id) ? 'declared-candidate' : 'not-established-initial',
        reachability: observed[domain][id] ? 'witnessed-initial' : 'conservative-envelope',
        support: 'unsupported', handler_id: key, ticket: FAMILIES[DOMAIN_FAMILY[domain]][0],
        fixture_families: sorted(families), pathways: paths,
        embedded_condition: Boolean(data.condition),
      });
    }
  }
  const byKey = new Map(entries.map(e => [e.key, e]));
  // Dex materializes cosmetic forms from their base entry; do not invent a
  // nonexistent TS definition for these aliases.
  for (const name of Object.keys(catalog.resolvers.species)) {
    const species = dex.species.get(name), key = `forms:${species.id}`;
    if (byKey.has(key)) continue;
    const parent = byKey.get(`forms:${dex.species.get(species.baseSpecies).id}`);
    assert.ok(parent, `No base source for ${key}`);
    const entry = {...parent, key, id: species.id, handler_id: key, derived_from: parent.key, pathways: [],
      observed_count: observed.forms[species.id] ?? 0,
      initial_class: observed.forms[species.id] ? 'observed-initial' : declared.forms.has(species.id) ? 'declared-candidate' : 'not-established-initial',
      reachability: observed.forms[species.id] ? 'witnessed-initial' : 'conservative-envelope',
      data_sha256: hashJSON({base: parent.data_sha256, id: species.id, name: species.name})};
    entries.push(entry); byKey.set(key, entry);
  }
  const coreFiles = [...readdirSync(path.join(UPSTREAM, 'sim')).filter(f => f.endsWith('.ts')).map(f => `sim/${f}`),
    'data/scripts.ts', 'data/rulesets.ts', 'data/typechart.ts', 'data/natures.ts', 'data/aliases.ts', 'data/formats-data.ts', 'config/formats.ts'];
  for (const file of coreFiles.sort()) {
    const source = parseSource(file); addSource(source);
    const key = `core:${file}`;
    entries.push({key, domain: 'core', id: file, source: source.ref(source.ast), callback_paths: [],
      data_sha256: sources[file], observed_count: 0, initial_class: 'profile-runtime', reachability: 'conservative-envelope',
      support: 'unsupported', handler_id: key, ticket: 'ENG-CORE', fixture_families: ['battle-core', 'callback-dispatch'],
      pathways: addPathways(pathways(source), key), embedded_condition: false});
  }
  entries.push({key: 'items:', domain: 'items', id: '', source: entries.find(e => e.key === 'core:sim/pokemon.ts').source,
    callback_paths: [], data_sha256: hashJSON('no-held-item'), observed_count: observed.items[''], initial_class: 'observed-initial',
    reachability: 'witnessed-initial', support: 'unsupported', handler_id: 'items:', ticket: 'ENG-ITEM',
    fixture_families: ['item-events'], pathways: [], embedded_condition: false});
  addPathways(pathways(generator), 'generator');
  assert.equal(new Set(dispatch.map(p => p.id)).size, dispatch.length, 'Duplicate pathway identity');
  for (const domain of Object.keys(TABLES)) for (const id of Object.keys(observed[domain])) assert.ok(entries.some(e => e.domain === domain && e.id === id), `Missing ${domain}:${id}`);
  const body = {
    schema_version: 'r6-mechanics-1', artifact_kind: 'mechanics-manifest', profile: pin.format_id, showdown_commit: pin.commit,
    reference_manifest_id: engine.manifest_id,
    scope: {kind: 'conservative-pinned-base-dex-envelope', minimal_reachability_proven: false,
      includes_legacy_and_nonstandard_candidates: true, custom_engine_implemented: false,
      dynamic_analysis: 'Syntactic sink inventory with whole-domain bounds; not a whole-program control-flow or reachability proof.',
      generated_effects: 'Embedded conditions, active-move mutations and transient callback objects inherit their enclosing source and fixture family. Core modules own effect dispatch and fallback conditions.',
      unknown_identity_policy: 'reject', external_plugins: false, custom_formats: false, server_config: false},
    inputs: Object.fromEntries(Object.entries(INPUTS).map(([k, file]) => [k, {file, sha256: fileHash(path.join(ROOT, file))}])),
    parser: {version: '5.9.3', lock_sha256: fileHash(path.join(ROOT, 'tools/mechanics-deps/package-lock.json'))},
    sources,
    generator_audit: {source: generator.ref(generator.ast), initial_forms: initialForms,
      declared_moves: sorted(declared.moves), declared_abilities: sorted(declared.abilities), item_candidates: sorted(declared.items),
      item_candidate_scope: 'Resolved literals, requiredItems and five Gluttony berries; conservative and not a branch-feasibility proof. All pinned items remain required by this envelope.',
      methods: Object.fromEntries(['randomTeam', 'randomSet', 'randomMoveset', 'getPokemonPool', 'getAbility', 'getPriorityItem', 'getItem', 'getForme'].map(n => [n, generator.ref(methods.get(n))]))},
    indirect_audit: INDIRECT_AUDIT.map(([domain, id, reason]) => ({entry: `${domain}:${id}`, reason,
      initial: observed[domain][id] ? 'observed' : declared[domain].has(id) ? 'declared-candidate' : 'not-established',
      reachability: observed[domain][id] ? 'witnessed-initial' : 'conservative-envelope'})),
    entries: entries.sort((a, b) => a.key.localeCompare(b.key, 'en')), pathways: dispatch.sort((a, b) => a.id.localeCompare(b.id, 'en')),
    unclassified_pathways: [],
  };
  return {...body, content_hash: hashJSON(body)};
}
