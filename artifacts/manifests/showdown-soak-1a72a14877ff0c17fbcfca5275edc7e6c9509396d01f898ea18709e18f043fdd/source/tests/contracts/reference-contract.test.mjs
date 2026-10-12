import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultSpec, ReferenceSession, runEpisode} from '../../packages/reference-node/runner.mjs';
import {contractManifest, episodeArtifacts, playerViewArtifact, submitAction} from '../../packages/reference-node/contract-v1.mjs';
import {admissibleActions, toProtocol} from '../../packages/battle-contract/actions.mjs';
import {makeArtifact} from '../../packages/battle-contract/artifacts.mjs';
import {canonical} from '../../packages/battle-contract/canonical.mjs';
import {RandomPlayerAI, seedFor} from '../../packages/reference-node/engine.mjs';
import {hashJSON as r1Hash} from '../../packages/reference-node/provenance.mjs';

const manifest = contractManifest();

test('natural hidden-trapping attempts and re-requests preserve semantic choices and history boundaries', async () => {
  const source = await runEpisode(defaultSpec('r1-smoke-8'));
  const bundle = episodeArtifacts(source, manifest);
  assert.equal(bundle.decisions.length, 118);
  assert.equal(bundle.decisions.filter(d => !d.payload.accepted).length, 2);
  bundle.decisions.forEach((artifact, i) => {
    const decision = artifact.payload, view = bundle.views[i];
    assert.equal(decision.view_id, view.artifact_id);
    assert.equal(toProtocol(decision.submitted_action, view.payload.request), source.decisions[i].choice);
    assert.deepEqual(view.payload.admissible_actions.map(a => toProtocol(a, view.payload.request)), source.decisions[i].admissible_choices);
    assert.equal(view.payload.visible_protocol.length, source.decisions[i].protocol_length);
    if (!decision.accepted) {
      const next = bundle.views[i + 1];
      assert.equal(next.payload.request_id, view.payload.request_id + 1);
      assert.equal(next.payload.turn, view.payload.turn);
      assert.equal(next.payload.request.active[0].trapped, true);
      assert.ok(next.payload.admissible_actions.every(a => a.kind !== 'switch'));
      assert.notEqual(next.artifact_id, view.artifact_id);
    }
  });
  // Envelope hashes cannot hide inconsistent history/request boundaries.
  const first = bundle.views[0];
  for (const patch of [{request_id: 999}, {turn: 999}, {visible_protocol: []}]) {
    assert.throws(() => makeArtifact('player_view', manifest.manifest_id, {...first.payload, ...patch}));
  }
  assert.ok(canonical(bundle.episode).includes('mechanics_seed'));
  assert.ok(bundle.views.every(v => !Object.hasOwn(v.payload, 'mechanics_seed')));
  const wrongPhase = structuredClone(source);
  wrongPhase.decisions[0].phase = 'switch';
  const {raw_output, raw_output_hash, content_hash, ...body} = wrongPhase;
  wrongPhase.content_hash = r1Hash(body);
  assert.throws(() => episodeArtifacts(wrongPhase, manifest), /decision boundary/);
});

test('live semantic submission executes Tera, rejects stale requests and invalid visible actions before engine input', async () => {
  const session = new ReferenceSession(defaultSpec('r1-smoke-0'));
  const submit = async (side, action) => {
    const view = playerViewArtifact(session.projection.view(side), manifest.manifest_id);
    const input = {schema_version: '1.0.0', side, request_id: view.payload.request_id, view_id: view.artifact_id, action};
    return {input, record: await submitAction(session, input, manifest.manifest_id)};
  };
  try {
    await session.start();
    const view = playerViewArtifact(session.projection.view('p1'), manifest.manifest_id);
    const invalid = {schema_version: '1.0.0', side: 'p1', request_id: view.payload.request_id, view_id: view.artifact_id, action: {kind: 'switch', slot: 0}};
    const count = session.input_tape.length;
    await assert.rejects(() => submitAction(session, invalid, manifest.manifest_id), /forbidden/);
    assert.equal(session.input_tape.length, count);
    const tera = admissibleActions(view.payload.request).find(a => a.terastallize);
    assert.ok(tera);
    const {input, record} = await submit('p1', tera);
    assert.equal(record.accepted, true);
    await assert.rejects(() => submitAction(session, input, manifest.manifest_id), /Stale/);
    await submit('p2', admissibleActions(session.projection.view('p2').request).find(a => a.kind === 'move' && !a.terastallize));
    assert.ok(session.projection.view('p1').protocol.some(line => line.startsWith('|-terastallize|p1a:')));
    assert.ok(admissibleActions(session.projection.view('p1').request).every(a => !a.terastallize));
  } finally { session.close(); }
});

test('switch slot is the current request order after a live switch', async () => {
  const session = new ReferenceSession(defaultSpec('r1-smoke-0'));
  try {
    await session.start();
    const first = session.projection.view('p1');
    await session.submit('p1', first.request_id, toProtocol({kind: 'switch', slot: 1}, first.request));
    const other = session.projection.view('p2');
    await session.submit('p2', other.request_id, toProtocol(admissibleActions(other.request)[0], other.request));
    const next = session.projection.view('p1');
    assert.equal(next.request.side.pokemon[0].ident, first.request.side.pokemon[1].ident);
    assert.equal(next.request.side.pokemon[1].ident, first.request.side.pokemon[0].ident);
    assert.equal(next.request.side.pokemon[0].active, true);
  } finally { session.close(); }
});

test('constructed hidden-disabled live state resolves to Struggle before any re-request', async () => {
  const session = await new ReferenceSession(defaultSpec('r1-smoke-0')).start();
  try {
    const side = session.stream.battle.sides[0], mon = side.active[0];
    mon.moveSlots.forEach((move, i) => { move.disabled = i === mon.moveSlots.length - 1 ? 'hidden' : true; });
    const visible = mon.getMoveRequestData();
    assert.deepEqual(visible.moves.map(m => m.disabled), [true, true, true, false]);
    assert.equal(mon.getMoves().length, 0);
    const before = session.projection.view('p1').request_id;
    session.stream.battle.choose('p1', 'move 4');
    assert.equal(side.choice.actions[0].moveid, 'struggle');
    assert.equal(side.choice.error, '');
    assert.equal(session.projection.view('p1').request_id, before);
  } finally { session.close(); }
});

test('pinned upstream AI includes slot zero in constructed revival target requests', async () => {
  const session = await new ReferenceSession(defaultSpec('r1-smoke-0')).start();
  try {
    const side = session.projection.view('p1').request.side;
    side.pokemon.forEach(mon => { mon.condition = '0 fnt'; });
    side.pokemon[0].reviving = true;
    const request = {forceSwitch: [true], side};
    let options;
    const ai = new RandomPlayerAI({write() {}}, {seed: seedFor('r2-revival-audit')});
    ai.chooseSwitch = (_, choices) => { options = choices.map(c => c.slot - 1); return choices[0].slot; };
    ai.receiveRequest(request);
    assert.deepEqual(options, [0, 1, 2, 3, 4, 5]);
    assert.deepEqual(admissibleActions(request).map(a => a.slot), options);
  } finally { session.close(); }
});
