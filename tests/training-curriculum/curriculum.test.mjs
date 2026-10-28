import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {BattleStream, generateTeam} from '../../packages/reference-node/engine.mjs';
import {UPSTREAM} from '../../packages/reference-node/provenance.mjs';
import {PlayerProjection} from '../../packages/reference-node/observations.mjs';
import {teamArtifact} from '../../packages/reference-node/contract-v1.mjs';
import {hashJSON} from '../../packages/battle-contract/canonical.mjs';
import {partition, teamKey} from '../../packages/dataset/identity.mjs';
import {legalChoices} from '../../packages/showdown-bot/protocol.mjs';
import {createActor} from '../../packages/showdown-panel/policy.mjs';
import {CATALOG, PLAN, curriculumSlot, curriculumEpisode, admitTrainingTeam, verifyPublicEvidence} from '../../packages/training-curriculum/index.mjs';
import {captureCurriculumEvidence} from '../../packages/training-curriculum/audit.mjs';
const {TeamValidator} = createRequire(import.meta.url)(`${UPSTREAM}/dist/sim/index.js`);
const RUN = 'bounded-c1-fixtures-v1';

async function startFixture(plan) {
  const projection = new PlayerProjection(); let ended = false;
  const stream = new class extends BattleStream {
    constructor() { super({noCatch: true}); }
    pushMessage(type, data) { projection.ingest(type, data); if (type === 'end') ended = true; }
  }();
  async function submitGroup(group) {
    const pending = ['p1', 'p2'].filter(side => projection.players[side].pending);
    assert.deepEqual(Object.keys(group).sort(), pending, 'Prefix/actor must submit exactly the current request group');
    for (const side of pending) assert(legalChoices(projection.players[side].request).includes(group[side]));
    const errorsBefore = projection.errors.length;
    for (const side of pending) projection.players[side].pending = false;
    await stream.write(pending.map(side => `>${side} ${group[side]}`).join('\n'));
    assert.equal(projection.errors.length, errorsBefore, 'Fixed fixtures must not produce unavailable/rejected choices');
  }
  const close = () => stream.battle?.destroy();
  try {
    await stream.write(`>start ${JSON.stringify({formatid: 'gen9randombattle', seed: plan.seed})}\n` +
      plan.teams.map((entry, i) => `>player p${i + 1} ${JSON.stringify({name: `Curriculum ${i + 1}`, team: entry.team})}`).join('\n'));
    for (const row of plan.prefix) { assert(!ended); await submitGroup(row); }
    assert(!ended, 'Prefix must hand off a live battle');
    return {projection, submitGroup, ended: () => ended, close};
  } catch (error) { close(); throw error; }
}

test('the complete frozen schedule has an exact roster and no implicit extension', () => {
  const count = {targeted: 0, official: 0}, scenarios = {}, families = {}, stages = {};
  const byId = new Map(CATALOG.scenarios.map(s => [s.id, s]));
  for (let i = 0; i < PLAN.episode_budget; i++) {
    const slot = curriculumSlot(i); count[slot.source]++;
    stages[slot.stage] ??= {targeted: 0, official: 0}; stages[slot.stage][slot.source]++;
    if (slot.scenario_id) { scenarios[slot.scenario_id] = (scenarios[slot.scenario_id] ?? 0) + 1;
      const family = byId.get(slot.scenario_id).family; families[family] = (families[family] ?? 0) + 1; }
  }
  assert.deepEqual(count, {targeted: 850, official: 2150});
  assert.deepEqual(stages, {'stage-0': {targeted: 500, official: 500}, 'stage-1': {targeted: 250, official: 750}, 'stage-2': {targeted: 100, official: 900}});
  assert.deepEqual(Object.keys(scenarios), PLAN.scenario_ids); assert(Object.values(scenarios).every(n => n === 85));
  assert.equal(Object.keys(families).length, 5); assert(Object.values(families).every(n => n === 170));
  for (const invalid of [-1, 3000, 1.5, NaN, '0']) assert.throws(() => curriculumSlot(invalid), /budget/);
});

test('all frozen scenarios have two train-admitted legal complete teams and stable family groups', () => {
  const validator = new TeamValidator('gen9anythinggoes');
  for (const scenario of CATALOG.scenarios) {
    assert.equal(scenario.teams.length, 2);
    for (const entry of scenario.teams) {
      assert.equal(entry.team.length, 6); assert.equal(new Set(entry.team.map(m => m.species)).size, 6);
      assert.equal(validator.validateTeam(structuredClone(entry.team)), null, scenario.id);
      const payload = teamArtifact(entry, '0'.repeat(64)).payload;
      assert.equal(partition(payload), 'train'); assert.equal(teamKey(payload), entry.team_key);
      assert(entry.admission_candidate >= 0 && entry.admission_candidate < CATALOG.team_admission_candidate_budget);
    }
    assert(scenario.leakage_group.startsWith('training-curriculum-v1/family/'));
  }
  for (let i = 0; i < 8; i += 2) {
    assert.equal(CATALOG.scenarios[i].leakage_group, CATALOG.scenarios[i + 1].leakage_group);
    assert.deepEqual(CATALOG.scenarios[i].teams.map(t => t.team_key), CATALOG.scenarios[i + 1].teams.map(t => t.team_key).reverse());
  }
  assert.equal(CATALOG.scenarios[8].counterfactual_group, CATALOG.scenarios[9].counterfactual_group);
  assert.deepEqual(CATALOG.scenarios[8].teams[0].team, CATALOG.scenarios[9].teams[0].team);
});

test('R7 rejection cannot be bypassed by forged metadata; never execute rejected candidate teams', () => {
  const seen = new Set();
  for (let n = 0; n < 128 && seen.size < 2; n++) {
    const entry = structuredClone(CATALOG.scenarios[0].teams[0]); entry.team[5].evs.hp = n;
    entry.team_hash = hashJSON(entry.team); entry.partition = 'train'; entry.team_key = '0'.repeat(64);
    const split = partition(teamArtifact(entry, '0'.repeat(64)).payload);
    if (split !== 'train') { assert.equal(admitTrainingTeam(entry), null); seen.add(split); }
  }
  assert.deepEqual([...seen].sort(), ['development', 'final']);
  const wrong = structuredClone(CATALOG.scenarios[0].teams[0]); wrong.team.pop();
  assert.throws(() => admitTrainingTeam(wrong), /six/);
  const corrupt = structuredClone(CATALOG.scenarios[0].teams[0]); corrupt.team[0].level--;
  assert.throws(() => admitTrainingTeam(corrupt), /hash/);
});

test('episode plans are deterministic, seed-separated, bounded and coordinator-only', () => {
  const a = curriculumEpisode({episodeIndex: 0, runSeed: RUN});
  assert.deepEqual(a, curriculumEpisode({episodeIndex: 0, runSeed: RUN}));
  const b = curriculumEpisode({episodeIndex: 0, runSeed: `${RUN}-other`});
  assert.notEqual(a.seed, b.seed); assert.deepEqual(a.teams, b.teams); assert.deepEqual(a.prefix, b.prefix);
  assert.equal(a.provenance.prefix_actions_trainable, false); assert.equal(a.provenance.outcome_selected, false);
  assert.equal(a.provenance.split, 'train'); assert.equal(a.provenance.complete_battle_continuation, true);
  assert.throws(() => { a.teams[0].team[0].level = 1; }, TypeError);
  assert.throws(() => curriculumEpisode({episodeIndex: 0, runSeed: ''}), /runSeed/);
  const official = curriculumEpisode({episodeIndex: 10, runSeed: RUN});
  assert.deepEqual(official, curriculumEpisode({episodeIndex: 10, runSeed: RUN}));
  assert.equal(official.provenance.source, 'official'); assert.deepEqual(official.prefix, []);
  assert.notEqual(official.teams[0].team_key, official.teams[1].team_key);
  for (const entry of official.teams) {
    assert.equal(entry.source, 'official'); assert.equal(entry.team.length, 6);
    assert.deepEqual(entry.team, generateTeam(entry.seed).team);
    assert.equal(partition(teamArtifact(entry, '0'.repeat(64)).payload), 'train');
    assert(entry.admission_attempt < PLAN.official_team_candidate_budget);
  }
});

test('every target is realized by public events from its legal scripted prefix in pinned Showdown', async () => {
  for (let episodeIndex = 0; episodeIndex < 10; episodeIndex++) {
    const plan = curriculumEpisode({episodeIndex, runSeed: RUN}), fixture = await startFixture(plan);
    try {
      const evidence = verifyPublicEvidence(fixture.projection.spectator, plan.evidence_requirements);
      assert.equal(evidence.length, plan.evidence_requirements.length);
      for (const side of ['p1', 'p2']) {
        const view = fixture.projection.view(side);
        assert.equal(view.request.side.pokemon.length, 6);
        assert(legalChoices(view.request).length > 0);
        assert(!JSON.stringify(view).includes(plan.provenance.scenario_id));
      }
      if (plan.provenance.family_id === 'status') {
        const sleeper = episodeIndex === 4 ? 'p2' : 'p1';
        assert(fixture.projection.players[sleeper].request.side.pokemon[0].condition.endsWith(' slp'));
      }
      assert.throws(() => verifyPublicEvidence([], plan.evidence_requirements), /Missing|Expected/);
    } finally { fixture.close(); }
  }
});

test('hidden reserve counterfactuals have identical pre-reveal actor inputs/actions, then differ only after public revelation', async () => {
  const a = curriculumEpisode({episodeIndex: 8, runSeed: RUN});
  const b = structuredClone(curriculumEpisode({episodeIndex: 9, runSeed: RUN})); b.seed = a.seed;
  const x = await startFixture(a), y = await startFixture(b);
  try {
    assert.notDeepEqual(a.teams[1].team, b.teams[1].team);
    assert.deepEqual(x.projection.view('p1'), y.projection.view('p1'));
    const choices = [x, y].map(fixture => {
      const actor = createActor('damage-v1'); fixture.projection.players.p1.protocol.forEach(line => actor.state.ingest(line));
      return actor.choose(structuredClone(fixture.projection.players.p1.request), structuredClone(actor.state));
    });
    assert.equal(choices[0], choices[1]);
    for (const fixture of [x, y]) {
      await fixture.submitGroup({p1: 'move 4', p2: 'switch 6'});
      await fixture.submitGroup({p1: 'move 2', p2: 'move 4'});
    }
    assert.notDeepEqual(x.projection.view('p1'), y.projection.view('p1'));
    assert(x.projection.spectator.some(line => line.includes('|move|p2a: Corviknight|Protect|')));
    assert(y.projection.spectator.some(line => line.includes('|move|p2a: Corviknight|Iron Defense|')));
  } finally { x.close(); y.close(); }
});

test('the prefix hands off to independent actors for a complete six-member battle', async () => {
  const fixture = await startFixture(curriculumEpisode({episodeIndex: 0, runSeed: RUN}));
  const actors = [createActor('damage-v1'), createActor('damage-v1')], cursors = [0, 0]; let groups = 0;
  try {
    while (!fixture.ended() && groups < 1000) {
      const choices = {};
      for (let i = 0; i < 2; i++) {
        const side = `p${i + 1}`, player = fixture.projection.players[side];
        while (cursors[i] < player.protocol.length) actors[i].state.ingest(player.protocol[cursors[i]++]);
        if (player.pending) choices[side] = actors[i].choose(structuredClone(player.request), structuredClone(actors[i].state));
      }
      await fixture.submitGroup(choices); groups++;
    }
    assert(fixture.ended(), 'Bounded fixture must complete naturally, without forced terminal reward');
    assert(groups > 1); assert(fixture.projection.spectator.some(line => /^\|(win|tie)\|/.test(line)));
  } finally { fixture.close(); }
});

test('evidence checker rejects private channels and missing target events', () => {
  assert.throws(() => verifyPublicEvidence(['|split|p1'], []), /private/);
  assert.throws(() => verifyPublicEvidence(['|request|{}'], []), /private/);
  assert.throws(() => verifyPublicEvidence(['|move|p1a: X|Protect|p1a: X'], [{kind: 'event', event: '-terastallize', side: 'p1', value: 'Fire'}]), /Missing/);
});

test('the fixed public-event evidence fixture replays exactly without outcome search', async () => {
  const expected = JSON.parse(readFileSync(new URL('./prefix-evidence.json', import.meta.url)));
  assert.deepEqual(await captureCurriculumEvidence(), expected);
});
