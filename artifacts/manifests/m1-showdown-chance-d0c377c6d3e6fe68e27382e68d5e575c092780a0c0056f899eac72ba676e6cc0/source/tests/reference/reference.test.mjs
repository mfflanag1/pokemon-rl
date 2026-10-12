import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {readFileSync, writeFileSync, unlinkSync} from 'node:fs';
import {describe, generateTeam, seedFor, BattleStream, getPlayerStreams} from '../../packages/reference-node/engine.mjs';
import {ROOT, UPSTREAM, hashJSON, pin, verifySource, verifyBuild} from '../../packages/reference-node/provenance.mjs';
import {defaultSpec, ReferenceSession, runEpisode, replayEpisode} from '../../packages/reference-node/runner.mjs';
import {PlayerProjection, normalizeProtocol, admissibleChoices} from '../../packages/reference-node/observations.mjs';

const spec = defaultSpec();
const episode = await runEpisode(spec);

test('fixed reference fingerprints remain stable across future harness changes', async () => {
  const golden = JSON.parse(readFileSync(new URL('../fixtures/reference-golden.json', import.meta.url), 'utf8'));
  assert.equal(golden.showdown_commit, pin.commit);
  for (const fixture of golden.fixtures) {
    const result = await runEpisode(defaultSpec(fixture.label));
    assert.deepEqual(result.initial_team_hashes, fixture.initial_team_hashes);
    assert.equal(result.normalized_output_hash, fixture.normalized_output_hash);
    assert.deepEqual(result.outcome, fixture.outcome);
    assert.equal(result.decisions.length, fixture.decision_count);
    assert.equal(result.decisions.filter(row => !row.accepted).length, fixture.unavailable_rejections);
  }
});

test('manifest identifies the exact source, lockfile and resolved rules', () => {
  const manifest = describe();
  assert.equal(manifest.commit, pin.commit);
  assert.equal(manifest.lock_sha256, pin.package_lock_sha256);
  assert.equal(manifest.source_dirty, false);
  assert.equal(manifest.format.game_type, 'singles');
  assert.ok(manifest.format.resolved_rules.some(([rule]) => rule === 'hppercentagemod'));
  assert.ok(manifest.format.resolved_rules.some(([rule]) => rule === 'sleepclausemod'));
  assert.equal(manifest.configuration.potd, '');
});

test('provenance refuses changed source and a tampered compiled tree', () => {
  const name = `.r1-provenance-test-${process.pid}`;
  for (const [file, check, message] of [
    [`${UPSTREAM}/${name}`, verifySource, /source is dirty/],
    [`${UPSTREAM}/dist/${name}`, verifyBuild, /Compiled Showdown tree changed/],
  ]) {
    writeFileSync(file, 'test-only integrity sentinel', {flag: 'wx'});
    try { assert.throws(check, message); } finally { unlinkSync(file); }
  }
});

test('official whole teams regenerate exactly with their own random stream', () => {
  for (const team of spec.teams) assert.deepEqual(generateTeam(team.seed), team);
  assert.notEqual(spec.teams[0].team_hash, spec.teams[1].team_hash);
  assert.throws(() => generateTeam('sodium,xyz'), /canonical sodium seed/);
  assert.throws(() => generateTeam([1, 2, 3, 4]), /canonical sodium seed/);
});

test('a completed seeded battle and its input tape reproduce exactly', async () => {
  assert.ok(episode.outcome.turns > 1);
  assert.ok(episode.decisions.some(row => row.phase === 'switch'));
  assert.ok(episode.decisions.every(row => !row.choice.includes('terastallize')));
  const second = await runEpisode(spec);
  assert.equal(second.content_hash, episode.content_hash);
  assert.equal((await replayEpisode(episode)).content_hash, episode.content_hash);
  assert.deepEqual(second.protocol, episode.protocol);
});

test('replay is independent of wall-clock time while raw evidence is retained', async () => {
  const originalNow = Date.now;
  try {
    Date.now = () => 1000000000000;
    const first = await runEpisode(spec);
    Date.now = () => 1500000000000;
    const second = await replayEpisode(first);
    assert.equal(first.content_hash, second.content_hash);
    assert.notEqual(first.raw_output_hash, second.raw_output_hash);
  } finally { Date.now = originalNow; }
});

test('natural hidden-trapping rejections retain the action and updated request on replay', async () => {
  const trapped = await runEpisode(defaultSpec('r1-smoke-8'));
  const rejected = trapped.decisions.filter(row => !row.accepted);
  assert.equal(rejected.length, 2);
  for (const row of rejected) {
    assert.match(row.errors[0].message, /\[Unavailable choice\].*trapped/);
    const next = trapped.decisions[trapped.decisions.indexOf(row) + 1];
    assert.equal(next.side, row.side);
    assert.equal(next.request_id, row.request_id + 1);
    assert.equal(next.turn, row.turn);
    assert.notEqual(next.request_hash, row.request_hash);
  }
  assert.equal((await replayEpisode(trapped)).content_hash, trapped.content_hash);
});

test('explicit teams retain identity when seats and mechanics seed change', async () => {
  const swapped = {...structuredClone(spec), teams: [...spec.teams].reverse(), mechanics_seed: seedFor('swapped/mechanics')};
  const session = new ReferenceSession(swapped);
  try {
    await session.start();
    assert.deepEqual(session.initial_team_hashes, swapped.teams.map(team => team.team_hash));
    assert.equal(session.projection.view('p1').request.side.id, 'p1');
  } finally { session.close(); }
});

test('projection agrees with upstream getPlayerStreams for an entire recorded battle', async () => {
  const stream = new BattleStream({noCatch: true});
  const streams = getPlayerStreams(stream);
  const collected = {};
  const readers = ['p1', 'p2', 'spectator'].map(async side => {
    collected[side] = [];
    for await (const chunk of streams[side]) collected[side].push(...normalizeProtocol(chunk).split('\n'));
  });
  for (const command of episode.input_tape) await stream.write(command);
  await Promise.all(readers);
  assert.deepEqual(collected, episode.protocol);
  stream.battle.destroy();
});

test('private requests, split HP and terminal private teams do not cross player channels', () => {
  const projection = new PlayerProjection();
  projection.ingest('update', '|split|p2\n|-damage|p2a: Example|137/301\n|-damage|p2a: Example|46/100');
  projection.ingest('sideupdate', 'p2\n|request|{"side":{"secret":"private-item"},"wait":true}');
  projection.ingest('end', '{"p2team":"whole-secret-team"}');
  const p1 = JSON.stringify(projection.view('p1'));
  assert.ok(p1.includes('46/100'));
  for (const secret of ['137/301', 'private-item', 'whole-secret-team']) assert.ok(!p1.includes(secret));
  assert.ok(JSON.stringify(projection.view('p2')).includes('137/301'));
});

test('real initial opponent-private mutation leaves the other player view unchanged', async () => {
  async function initialView(mutate) {
    const stream = new BattleStream({noCatch: true});
    const streams = getPlayerStreams(stream);
    const out = {p1: [], p2: []};
    const readers = ['p1', 'p2'].map(async side => {
      for await (const chunk of streams[side]) out[side].push(normalizeProtocol(chunk));
    });
    const teams = spec.teams.map(entry => structuredClone(entry.team));
    if (mutate) teams[1][5].item = teams[1][5].item === 'Leftovers' ? 'Choice Band' : 'Leftovers';
    await stream.write(`>start ${JSON.stringify({formatid: pin.format_id, seed: spec.mechanics_seed, p1: {name: 'A', team: teams[0]}, p2: {name: 'B', team: teams[1]}})}`);
    await stream.writeEnd();
    await Promise.all(readers);
    return out;
  }
  // Deliberately constructed hidden-state test, not an official generated-team corpus sample.
  const before = await initialView(false);
  const after = await initialView(true);
  assert.deepEqual(before.p1, after.p1);
  assert.notDeepEqual(before.p2, after.p2);
});

test('stale requests and injected control commands are rejected before simulation', async () => {
  const session = new ReferenceSession(spec);
  try {
    await session.start();
    const view = session.projection.view('p1');
    const before = session.input_tape.length;
    await assert.rejects(session.submit('p1', view.request_id + 1, 'move 1'), /Stale/);
    await assert.rejects(session.submit('p1', view.request_id, 'move 1\n>eval 1'), /syntax/);
    await assert.rejects(session.submit('p1', view.request_id, 'switch 1'), /visible request/);
    assert.equal(session.input_tape.length, before);
  } finally { session.close(); }
});

test('observable choices exclude known-disabled moves without inspecting hidden legality', () => {
  const request = {side: {pokemon: [
    {active: true, condition: '100/100'}, {active: false, condition: '100/100'},
  ]}, active: [{moves: [{disabled: true}, {disabled: false}], maybeTrapped: true, canTerastallize: 'Fire'}]};
  assert.deepEqual(admissibleChoices(request), ['move 2', 'move 2 terastallize', 'switch 2']);
  request.active[0].trapped = true;
  assert.deepEqual(admissibleChoices(request), ['move 2', 'move 2 terastallize']);
});

test('timestamps alone normalize; choice errors, HP and mechanics remain intact', () => {
  assert.equal(normalizeProtocol('|t:|123\n|turn|12\n|-damage|p1a: Test|12/100'), '|t:|<timestamp>\n|turn|12\n|-damage|p1a: Test|12/100');
  assert.equal(normalizeProtocol('|error|test'), '|error|test');
});

test('tampered episode, raw trace and incompatible schemas fail replay', async () => {
  const damaged = structuredClone(episode);
  damaged.decisions[0].choice = 'move 4';
  // Force a definite content difference even if the original selected move 4.
  damaged.outcome.turns++;
  await assert.rejects(replayEpisode(damaged), /content hash mismatch/);
  const raw = structuredClone(episode);
  raw.raw_output[0].data += '\n|invalid|tamper';
  await assert.rejects(replayEpisode(raw), /Raw output hash mismatch/);
  await assert.rejects(replayEpisode({...episode, schema_version: 'future'}), /Unsupported episode schema/);
});

test('live replay detects divergent requests and results even with a valid envelope hash', async () => {
  function resign(changed) {
    const {raw_output, raw_output_hash, content_hash, ...body} = changed;
    return {...body, content_hash: hashJSON(body), raw_output, raw_output_hash};
  }
  const request = structuredClone(episode);
  request.decisions[0].request_hash = '0'.repeat(64);
  await assert.rejects(replayEpisode(resign(request)), /observation or action result diverged/);
  const order = structuredClone(episode);
  order.decisions[0].request_id++;
  await assert.rejects(replayEpisode(resign(order)), /decision order diverged/);
  const outcome = structuredClone(episode);
  outcome.outcome.turns++;
  await assert.rejects(replayEpisode(resign(outcome)), /transcript\/result mismatch/);
});

test('a safety cap cannot be reported as a completed draw', async () => {
  await assert.rejects(runEpisode({...spec, max_submissions: 1}), /incomplete, not a draw/);
});

test('JSONL worker processes multiple requests and reports malformed commands', () => {
  const result = spawnSync(process.execPath, ['packages/reference-node/cli.mjs', 'worker'], {
    cwd: ROOT, encoding: 'utf8', input: [
      JSON.stringify({id: 'a', command: 'describe'}),
      JSON.stringify({id: 'b', command: 'generate_teams', seeds: [spec.teams[0].seed]}),
      JSON.stringify({id: 'c', command: 'not-supported'}),
    ].join('\n'),
  });
  assert.equal(result.status, 0, result.stderr);
  const rows = result.stdout.trim().split('\n').map(JSON.parse);
  assert.deepEqual(rows.map(row => row.id), ['a', 'b', 'c']);
  assert.deepEqual(rows.map(row => row.ok), [true, true, false]);
  assert.equal(rows[1].result[0].team_hash, spec.teams[0].team_hash);
});

test('artifact hash is independent of object key insertion order', () => {
  assert.equal(hashJSON({a: 1, b: [2, 3]}), hashJSON({b: [2, 3], a: 1}));
  assert.throws(() => hashJSON({bad: NaN}), /finite JSON/);
});
