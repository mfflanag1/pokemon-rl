import test from 'node:test';
import assert from 'node:assert/strict';
import {ACTIONS, choiceFor, requestActions} from '../../packages/full-showdown/actions.mjs';
import {RosterView} from '../../packages/full-showdown/view.mjs';
import {FullShowdownSession, officialPlan} from '../../packages/full-showdown/session.mjs';
import {teamArtifact} from '../../packages/reference-node/contract-v1.mjs';
import {hashJSON} from '../../packages/reference-node/provenance.mjs';
import {partition, teamKey} from '../../packages/dataset/identity.mjs';
import {trappedFrames} from './trapped-fixture.mjs';
import {rechargeFrames} from './recharge-fixture.mjs';

function request(side = 'p1') {
  return {side: {id: side, pokemon: Array.from({length: 6}, (_, i) => ({ident: `${side}: Test${i}`,
    details: 'Pikachu, L80', condition: i === 5 ? '0 fnt' : '100/200', active: i === 0,
    stats: {atk: 100, def: 100, spa: 100, spd: 100, spe: 100}, moves: ['tackle'], ability: 'static', item: 'lightball', teraType: 'Water'}))},
  active: [{moves: [{id: 'tackle', move: 'Tackle', pp: 10, maxpp: 35}, {id: 'protect', move: 'Protect', disabled: true}], canTerastallize: 'Water'}]};
}

test('14 semantic actions: normal/Tera, switching, wait and stale legality', () => {
  const req = request(), {mask} = requestActions(req);
  assert.equal(ACTIONS.length, 14);
  assert.deepEqual(mask.map((live, i) => live ? i : -1).filter(i => i >= 0), [0, 4, 9, 10, 11, 12]);
  assert.equal(choiceFor(4, req), 'move 1 terastallize');
  assert.throws(() => choiceFor(1, req), /Illegal/);
  assert.ok(requestActions({wait: true}).mask.every(x => !x));
  req.active[0].trapped = true;
  assert.equal(requestActions(req).mask.filter(Boolean).length, 2);
});

test('forced replacements, Revival Blessing, pass and Struggle use request slots', () => {
  const req = request(); delete req.active; req.forceSwitch = [true];
  assert.deepEqual(requestActions(req).mask.slice(8), [false, true, true, true, true, false]);
  req.side.pokemon[0].reviving = true;
  assert.deepEqual(requestActions(req).mask.slice(8), [false, false, false, false, false, true]);
  req.forceSwitch = [false]; assert.equal(requestActions(req).pass, true);
  delete req.forceSwitch; req.active = [{moves: [{id: 'struggle', move: 'Struggle', pp: 1, maxpp: 1}], trapped: true}];
  assert.deepEqual(requestActions(req).mask, ACTIONS.map((_, i) => i === 0));
});

test('public roster never imports opponent private state and keeps six slots', () => {
  const view = new RosterView('p1');
  view.ingest('|switch|p2a: Charizard|Charizard, L80|100/100');
  const before = view.observe(request(), 1);
  assert.equal(before.own.length, 6); assert.equal(before.opponent.length, 6);
  assert.equal(before.opponent.filter(Boolean).length, 1);
  assert.equal(before.opponent[0].ability, null); assert.equal(before.opponent[0].item, null);
  assert.deepEqual(before.opponent[0].moves, []);
  assert.throws(() => view.observe(request('p2'), 2), /Foreign/);
  assert.throws(() => view.ingest('|request|{"side":{"id":"p2"}}'), /Private/);
  assert.throws(() => view.ingest('|split|p2'), /Private/);
  const req = request(); req.side.pokemon.forEach(p => { p.opponent_private = 'secret'; });
  assert.deepEqual(view.observe(req, 1), before);
  view.ingest('|move|p2a: Charizard|Flamethrower|p1a: Test0');
  view.ingest('|-damage|p2a: Charizard|47/100 brn');
  assert.deepEqual(view.observe(request(), 2).opponent[0].moves, ['flamethrower']);
  assert.equal(view.observe(request(), 2).opponent[0].condition, '47/100 brn');
});

test('Illusion correction transfers current reveals and restores prior disguise knowledge', () => {
  const view = new RosterView('p1');
  view.ingest('|switch|p2a: Pikachu|Pikachu, L80|80/100');
  view.ingest('|move|p2a: Pikachu|Thunderbolt|p1a: Test0');
  view.ingest('|switch|p2a: Gengar|Gengar, L80|100/100');
  view.ingest('|switch|p2a: Pikachu|Pikachu, L80|100/100');
  view.ingest('|move|p2a: Pikachu|Night Daze|p1a: Test0');
  view.ingest('|replace|p2a: Zoroark|Zoroark, L80');
  const roster = view.observe(request(), 1).opponent;
  assert.deepEqual(roster.find(p => p?.species === 'Pikachu').moves, ['thunderbolt']);
  assert.deepEqual(roster.find(p => p?.species === 'Zoroark').moves, ['nightdaze']);
  assert.equal(roster.filter(p => p?.active).length, 1);
});

test('same-player public reserve HP survives switching', () => {
  const view = new RosterView('p1');
  view.ingest('|switch|p2a: Pikachu|Pikachu, L80|100/100');
  view.ingest('|-damage|p2a: Pikachu|20/100');
  view.ingest('|switch|p2a: Gengar|Gengar, L80|100/100');
  const roster = view.observe(request(), 1).opponent;
  assert.equal(roster[0].condition, '20/100'); assert.equal(roster[0].active, false);
});

test('pinned session train admission, fresh joint frame and stale-request rejection', async () => {
  const session = new FullShowdownSession();
  try {
    const frame = await session.start({episodeIndex: 0, runSeed: 7781});
    assert.equal(frame.players.p1.phase, 'move'); assert.equal(frame.players.p2.phase, 'move');
    assert.ok(frame.provenance.teams.every(t => t.partition === 'train'));
    assert.equal(frame.players.p1.opponent.filter(Boolean).length, 1);
    assert.equal(JSON.stringify(frame.players).includes('team_key'), false);
    await assert.rejects(session.step({actions: {p1: 0, p2: 0}, request_ids: {p1: -1, p2: -1}}), /Stale/);
    assert.equal(session.submissions, 0);
    const cut = session.cutoff();
    assert.equal(cut.outcome.kind, 'administrative_cutoff');
    assert.deepEqual(cut.outcome.rewards, {p1: 0, p2: 0});
    assert.ok(cut.players.p1.mask.some(Boolean));
  } finally { session.close(); }
});

test('coordinator rejects forged nontraining plans', async () => {
  const plan = officialPlan(1, 7781); plan.teams[0].partition = 'final';
  const session = new FullShowdownSession();
  try { await assert.rejects(session.start({episodeIndex: 1, runSeed: 7781, plan}), /Nontraining/); }
  finally { session.close(); }
});

test('real hidden Arena Trap rejects an advertised switch, refreshes masks and continues', async () => {
  const {initial, rejected, continued, cutoff} = await trappedFrames();
  assert.ok(initial.provenance.teams.every(t => t.partition === 'train'));
  assert.equal(initial.players.p1.opponent[0].ability, null);
  assert.equal(initial.players.p1.mask[9], true);
  assert.equal(rejected.submitted[0].side, 'p1');
  assert.equal(rejected.submitted[0].accepted, false);
  assert.match(rejected.submitted[0].feedback[0], /\[Unavailable choice\].*trapped/);
  assert.equal(rejected.submitted[1].accepted, true);
  assert.ok(rejected.players.p1.request_id > initial.players.p1.request_id);
  assert.ok(rejected.players.p1.mask.slice(8).every(live => !live));
  assert.equal(rejected.players.p1.mask[0], true);
  assert.ok(rejected.players.p2.mask.every(live => !live));
  assert.equal(rejected.outcome, null);
  assert.equal(rejected.players.p1.turn, 1);
  assert.deepEqual(continued.submitted.map(row => [row.side, row.accepted]), [['p1', true]]);
  assert.equal(continued.outcome, null);
  assert.equal(continued.players.p1.turn, 2);
  assert.ok(continued.players.p1.mask.some(Boolean) && continued.players.p2.mask.some(Boolean));
  assert.equal(continued.submissions, 3);
  assert.equal(cutoff.outcome.kind, 'administrative_cutoff');
  assert.deepEqual(cutoff.outcome.rewards, {p1: 0, p2: 0});
});

test('real Hyper Beam recharge request has explicit finite metadata and resumes normally', async () => {
  const {initial, recharge, rechargeRequest, resumed, publicEvidence} = await rechargeFrames();
  assert.ok(initial.provenance.teams.every(t => t.partition === 'train'));
  assert.deepEqual(rechargeRequest.moves, [{move: 'Recharge', id: 'recharge'}]);
  assert.equal(rechargeRequest.trapped, true);
  assert.deepEqual(recharge.players.p1.active_moves, [{id: 'recharge', pp: null, maxpp: null,
    disabled: false, base_power: 0, type: '???', category: 'Status'}]);
  assert.deepEqual(recharge.players.p1.mask, ACTIONS.map((_, i) => i === 0));
  assert.equal(recharge.players.p1.turn, 2);
  assert.equal(recharge.outcome, null);
  assert.equal(resumed.players.p1.active_moves[0].id, 'hyperbeam');
  assert.equal(resumed.players.p1.active_moves[0].category, 'Special');
  assert.equal(resumed.players.p1.turn, 3);
  assert.equal(resumed.outcome, null);
  assert.ok(resumed.submitted.every(row => row.accepted));
  assert.ok(publicEvidence.includes('|-mustrecharge|p1a: Machamp'));
  assert.ok(publicEvidence.includes('|cant|p1a: Machamp|recharge'));
});

test('unknown and Gen1-only request controls fail explicitly; Struggle retains Dex metadata', () => {
  const view = new RosterView('p1'), req = request();
  for (const id of ['unknownrequestmove', 'fight']) {
    req.active[0].moves = [{id, move: id}];
    assert.throws(() => view.observe(req, 1), new RegExp(`Unknown Gen9 request move: ${id}`));
  }
  req.active[0].moves = [{id: 'struggle', move: 'Struggle', disabled: false}];
  assert.deepEqual(view.observe(req, 1).active_moves, [{id: 'struggle', pp: null, maxpp: null,
    disabled: false, base_power: 50, type: 'Normal', category: 'Physical'}]);
});

test('changing unrevealed opponent reserve order leaves the other actor byte-identical', async () => {
  const plan = officialPlan(2, 7781);
  let alternative = null;
  for (let index = 2; index <= 4 && !alternative; index++) {
    const candidate = structuredClone(plan), team = candidate.teams[1];
    [team.team[1], team.team[index]] = [team.team[index], team.team[1]];
    team.team_hash = hashJSON(team.team);
    const payload = teamArtifact(team, '0'.repeat(64)).payload;
    if (partition(payload) === 'train') { team.team_key = teamKey(payload); alternative = candidate; }
  }
  assert.ok(alternative, 'Deterministic paired fixture must remain train admitted');
  const one = new FullShowdownSession(), two = new FullShowdownSession();
  try {
    const a = await one.start({episodeIndex: 2, runSeed: 7781, plan});
    const b = await two.start({episodeIndex: 2, runSeed: 7781, plan: alternative});
    assert.deepEqual(a.players.p1, b.players.p1);
    assert.notDeepEqual(a.players.p2.own, b.players.p2.own);
  } finally { one.close(); two.close(); }
});
