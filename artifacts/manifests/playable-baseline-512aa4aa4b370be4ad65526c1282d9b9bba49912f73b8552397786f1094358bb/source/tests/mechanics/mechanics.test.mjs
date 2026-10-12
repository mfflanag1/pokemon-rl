import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {buildManifest, runtimeCallbacks, enumerateInitialForms} from '../../packages/mechanics/manifest.mjs';
import {captureFixtures} from '../../packages/mechanics/fixtures.mjs';
import {classifyPathway} from '../../packages/mechanics/policy.mjs';
import {callbacks, pathways, tableEntries, ts} from '../../packages/mechanics/source-index.mjs';

let manifest;
test('all R3 roots have sources; conservative inventory contains unobserved generated effects', () => {
  manifest = buildManifest();
  const entries = new Map(manifest.entries.map(e => [e.key, e]));
  for (const key of ['moves:struggle', 'forms:ogerponwellspringtera', 'forms:terapagosstellar', 'abilities:embodyaspectwellspring', 'moves:metronome']) {
    assert.ok(entries.has(key)); assert.equal(entries.get(key).support, 'unsupported');
    assert.equal(entries.get(key).observed_count, 0);
  }
  const f = JSON.parse(readFileSync('artifacts/runs/r3-gen9-random-100k-v2/inventory-final/frequencies.json'));
  for (const domain of ['moves', 'abilities', 'items', 'forms']) for (const [id, n] of Object.entries(f[domain])) {
    assert.equal(entries.get(`${domain}:${id === '<none>' ? '' : id}`).observed_count, n);
  }
  assert.equal(manifest.scope.minimal_reachability_proven, false);
  assert.ok(manifest.pathways.some(p => p.kind === 'computed-call'));
  assert.ok(manifest.pathways.some(p => p.kind === 'identity-state-write'));
  assert.equal(new Set(manifest.pathways.map(p => p.id)).size, manifest.pathways.length);
  assert.ok(manifest.generator_audit.initial_forms.pikachu.length > 1);
});

test('form enumeration rejects hidden RNG channels and restores generator state', () => {
  const original = {}, generator = {prng: original, sample: () => 0, getForme() { this.prng.next(); }};
  assert.throws(() => enumerateInitialForms({entries: {pikachu: {}}}, generator), /randomness outside/);
  assert.equal(generator.prng, original);
});

test('AST indexes array callbacks and computed calls; ordinary inherited names are not sinks', () => {
  const text = 'const Moves = {probe: {secondary: [{onHit() { this.actions[which](x); p.ability = v; x.toString(); }}]}};';
  const ast = ts.createSourceFile('probe.ts', text, ts.ScriptTarget.Latest, true);
  const s = {ast, ref: n => ({start_byte: n.getStart(ast)})};
  const entry = tableEntries(s, 'Moves').get('probe');
  assert.deepEqual(callbacks(s, entry).map(c => c.path), ['secondary.0.onHit']);
  assert.deepEqual(pathways(s, entry).map(p => p.kind), ['computed-call', 'identity-state-write', 'source-owned-call']);
  assert.deepEqual(runtimeCallbacks({secondary: [{onHit() {}}]}), ['secondary.0.onHit']);
  assert.throws(() => classifyPathway({kind: 'new-sink'}), /Unclassified/);
  assert.throws(() => classifyPathway({kind: 'toString'}), /Unclassified/);
});

test('sixteen scripted reference cases satisfy semantic assertions and replay in both seats', async () => {
  const fixtures = await captureFixtures();
  assert.equal(fixtures.cases.length, 16);
  assert.ok(fixtures.cases.every(c => c.replay_verified && c.assertions_passed && !c.official_generator_witness));
  assert.equal(new Set(fixtures.cases.map(c => c.id)).size, 16);
});
