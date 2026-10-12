import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {canonical, parseJSON, hashJSON} from '../../packages/battle-contract/canonical.mjs';
import {validateSchema, contractURL} from '../../packages/battle-contract/schema.mjs';
import {makeArtifact, validateArtifact, loadPlayerView, vocabularyIndex} from '../../packages/battle-contract/artifacts.mjs';
import {actionIndex, actionFromIndex, admissibleActions, actionMask, toProtocol, fromProtocol} from '../../packages/battle-contract/actions.mjs';

const golden = parseJSON(readFileSync(new URL('../fixtures/contract-golden.json', import.meta.url)));
const view = golden.artifacts.find(a => a.artifact_kind === 'player_view');
export function pythonBridge(requests) {
  const result = spawnSync(process.env.CONTRACT_PYTHON ?? '.venv/bin/python', ['tests/contracts/bridge.py'], {
    cwd: new URL('../../', import.meta.url), env: {...process.env, PYTHONPATH: 'python'},
    input: requests.map(r => canonical(r)).join('\n') + '\n', encoding: 'utf8', maxBuffer: 32 * 1024 * 1024, timeout: 60000,
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim().split('\n').map(parseJSON);
}

test('canonical bytes are portable, including Unicode scalar ordering and integer boundaries', () => {
  for (const vector of golden.canonical_vectors) {
    const value = parseJSON(vector.wire);
    assert.equal(canonical(value), vector.canonical);
    assert.equal(hashJSON(value), vector.sha256);
  }
  const results = pythonBridge(golden.canonical_vectors.map(v => ({op: 'wire', value: v.wire})));
  results.forEach((r, i) => assert.deepEqual(r, {ok: true, result: {canonical: golden.canonical_vectors[i].canonical, hash: golden.canonical_vectors[i].sha256}}));
});

test('strict wire decoders reject ambiguous numbers, duplicate keys, malformed Unicode and trailing data', () => {
  const wires = [...golden.invalid_wire, '['.repeat(101) + '0' + ']'.repeat(101)];
  for (const wire of wires) assert.throws(() => parseJSON(wire));
  assert.throws(() => parseJSON(Buffer.from([34, 255, 34])));
  for (const value of [undefined, NaN, Infinity, 1.5, 9007199254740992, [undefined], new Date()]) assert.throws(() => canonical(value));
  for (const r of pythonBridge(wires.map(value => ({op: 'wire', value})))) assert.equal(r.ok, false);
});

test('shared artifact goldens validate independently and preserve all bytes through Python hashing', () => {
  const results = pythonBridge(golden.artifacts.map(value => ({op: 'artifact', value})));
  golden.artifacts.forEach((artifact, i) => {
    validateArtifact(artifact);
    assert.deepEqual(makeArtifact(artifact.artifact_kind, artifact.manifest_id, artifact.payload), artifact);
    assert.deepEqual(results[i], {ok: true, result: hashJSON(artifact)});
  });
});

test('all 14 policy indices are bijective; pass and wait have no trainable index', () => {
  for (let i = 0; i < 14; i++) {
    const expected = i < 8 ? {kind: 'move', slot: i % 4, terastallize: i >= 4} : {kind: 'switch', slot: i - 8};
    assert.deepEqual(actionFromIndex(i), expected);
    assert.equal(actionIndex(expected), i);
  }
  for (const i of [-1, 14, true, '0', 0.5, NaN]) assert.throws(() => actionFromIndex(i));
  assert.throws(() => actionIndex({kind: 'pass'}));
  const results = pythonBridge(Array.from({length: 14}, (_, value) => ({op: 'index', value})));
  results.forEach((r, i) => assert.deepEqual(r, {ok: true, result: {action: actionFromIndex(i), index: i}}));
});

test('captured and constructed requests agree across semantic, protocol and tensor codecs', () => {
  const results = pythonBridge(golden.cases.map(c => ({op: 'request', value: c.request})));
  golden.cases.forEach((c, i) => {
    assert.deepEqual(admissibleActions(c.request), c.actions, c.name);
    assert.deepEqual(actionMask(c.request), c.mask, c.name);
    assert.deepEqual(results[i], {ok: true, result: {actions: c.actions, mask: c.mask}}, c.name);
    c.actions.forEach((a, j) => {
      assert.equal(toProtocol(a, c.request), c.commands[j]);
      assert.deepEqual(fromProtocol(c.commands[j], c.request), a);
      if (a.kind !== 'pass') assert.deepEqual(actionFromIndex(actionIndex(a)), a);
    });
  });
  const struggle = golden.cases.find(c => c.name === 'struggle');
  assert.deepEqual(struggle.actions, [{kind: 'move', slot: 0, terastallize: false}]);
  assert.deepEqual(struggle.mask, [true, ...Array(13).fill(false)]);
  for (const name of ['wait', 'forced-pass', 'no-bench-pass']) assert.ok(golden.cases.find(c => c.name === name).mask.every(v => !v));
  assert.deepEqual(golden.cases.find(c => c.name === 'revival-targets').actions.map(a => a.slot), [1, 2, 3, 4, 5]);
  assert.deepEqual(golden.cases.find(c => c.name === 'revival-active-target').actions.map(a => a.slot), [0, 1, 2, 3, 4, 5]);
});

test('known zero, absent item and unknown knowledge are distinct', () => {
  validateSchema({state: 'known', value: 0}, 'known_integer');
  validateSchema({state: 'unknown'}, 'known_integer');
  const bad = [0, null, {state: 'unknown', value: 0}, {state: 'known'}, {state: 'known', value: true}];
  bad.forEach(v => assert.throws(() => validateSchema(v, 'known_integer')));
  pythonBridge(bad.map(value => ({op: 'fragment', fragment: 'known_integer', value}))).forEach(r => assert.equal(r.ok, false));
  const vocab = parseJSON(readFileSync(new URL('vocabulary.json', contractURL)));
  assert.equal(vocabularyIndex(vocab, 'items', {state: 'unknown'}), 0);
  assert.equal(vocabularyIndex(vocab, 'items', {state: 'known', value: ''}), 1);
  assert.ok(vocabularyIndex(vocab, 'items', {state: 'known', value: 'leftovers'}) > 1);
  assert.throws(() => vocabularyIndex(vocab, 'items', {state: 'known', value: 'invaliditem'}));
});

test('malformed choices, unknown versions and unsupported phases fail in both languages', () => {
  const badActions = [{kind: 'move', slot: true, terastallize: false}, {kind: 'switch', slot: 6}, {kind: 'move', slot: 0}, {kind: 'wait'}, {kind: 'pass', slot: 0}];
  badActions.forEach(a => assert.throws(() => validateSchema(a, 'action')));
  pythonBridge(badActions.map(value => ({op: 'fragment', fragment: 'action', value}))).forEach(r => assert.equal(r.ok, false));
  const badRequests = [
    {...view.payload.request, teamPreview: true},
    {...view.payload.request, active: [view.payload.request.active[0], view.payload.request.active[0]]},
    {...view.payload.request, opponentTeam: []},
  ];
  for (const field of ['canMegaEvo', 'canZMove', 'canDynamax', 'maxMoves']) {
    const req = structuredClone(view.payload.request); req.active[0][field] = true; badRequests.push(req);
  }
  const disabled = structuredClone(view.payload.request);
  disabled.active[0].moves.forEach(m => { m.disabled = true; }); badRequests.push(disabled);
  const commanding = structuredClone(view.payload.request); commanding.side.pokemon[0].commanding = true; badRequests.push(commanding);
  badRequests.forEach(r => assert.throws(() => admissibleActions(r)));
  pythonBridge(badRequests.map(value => ({op: 'request', value}))).forEach(r => assert.equal(r.ok, false));
  const badVersions = ['2.0.0', '1.1.0'].map(schema_version => ({...view, schema_version}));
  badVersions.forEach(a => assert.throws(() => validateArtifact(a)));
  pythonBridge(badVersions.map(value => ({op: 'artifact', value}))).forEach(r => assert.equal(r.ok, false));
  for (const wire of ['move 0', 'move 5', 'switch 7', 'move 1\n', 'move 1\n>eval x', 'move 1 mega', 'default', 'undo', 'move thunderbolt']) assert.throws(() => fromProtocol(wire, view.payload.request));
  assert.throws(() => toProtocol({kind: 'switch', slot: 0}, view.payload.request));
  const struggle = golden.cases.find(c => c.name === 'struggle').request;
  assert.throws(() => toProtocol({kind: 'move', slot: 0, terastallize: true}, struggle));
  assert.throws(() => validateSchema({kind: 'showdown-sodium', hex: 9007199254740992}, 'seed'));
  const badSeeds = [9007199254740992, 'a'.repeat(64) + '\n', 'A'.repeat(64), '0'.repeat(16)];
  for (const hex of badSeeds) {
    const value = {kind: 'showdown-sodium', hex};
    assert.throws(() => validateSchema(value, 'seed'));
    // Unsafe numeric wire values are rejected before schema validation in the bridge.
    if (typeof hex === 'string') assert.equal(pythonBridge([{op: 'fragment', fragment: 'seed', value}])[0].ok, false);
  }
});

test('actor loader rejects privileged artifacts, wrong manifest, wrong viewer and inconsistent masks', () => {
  for (const artifact of golden.artifacts) {
    if (artifact.artifact_kind === 'player_view') assert.equal(loadPlayerView(canonical(artifact), artifact.manifest_id).viewer, artifact.payload.viewer);
    else assert.throws(() => loadPlayerView(canonical(artifact)));
  }
  assert.throws(() => loadPlayerView(canonical(view), '0'.repeat(64)));
  for (const [key, value] of [['content_hash', '0'.repeat(64)], ['artifact_id', 'player_view:' + '0'.repeat(64)]]) assert.throws(() => validateArtifact({...view, [key]: value}));
  for (const [key, value] of [['viewer', 'p2'], ['action_mask', Array(14).fill(false)], ['opponent_private_state', {state: 'unknown', value: 0}], ['opponent_team', []]]) {
    assert.throws(() => makeArtifact('player_view', view.manifest_id, {...view.payload, [key]: value}));
  }
});
