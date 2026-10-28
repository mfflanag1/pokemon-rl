import {BattleStream, generateTeam, seedFor, describe, checkSeed} from '../reference-node/engine.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {partition, teamKey} from '../dataset/identity.mjs';
import {requestActions, choiceFor, ACTIONS} from './actions.mjs';
import {RosterView} from './view.mjs';
import {validatePlayer} from './schema.mjs';

const SIDES = ['p1', 'p2'];
export function officialPlan(episodeIndex, runSeed) {
  const teams = [], seen = new Set();
  for (let attempt = 0; teams.length < 2 && attempt < 100; attempt++) {
    const entry = generateTeam(seedFor(`full-showdown-f1/train/${runSeed}/${episodeIndex}/team/${attempt}`));
    const payload = teamArtifact(entry, '0'.repeat(64)).payload;
    const key = teamKey(payload);
    if (partition(payload) !== 'train' || seen.has(key)) continue;
    seen.add(key); teams.push({...entry, team_key: key, partition: 'train'});
  }
  if (teams.length !== 2) throw new Error('Train team admission budget exhausted');
  return {teams, seed: seedFor(`full-showdown-f1/train/${runSeed}/${episodeIndex}/mechanics`),
    provenance: {source: 'official', partition: 'train', namespace: 'full-showdown-f1/train', episode_index: episodeIndex}};
}

export class FullShowdownSession {
  constructor() {
    this.projection = new PlayerProjection();
    this.views = Object.fromEntries(SIDES.map(side => [side, new RosterView(side)]));
    this.cursors = {p1: 0, p2: 0}; this.outcome = null; this.submissions = 0; this.started = false;
    const self = this;
    this.stream = new class extends BattleStream {
      constructor() { super({noCatch: true}); }
      pushMessage(type, data) {
        self.projection.ingest(type, data);
        if (type === 'end') {
          const {winner, turns} = JSON.parse(data);
          if (winner && !['Full p1', 'Full p2'].includes(winner)) throw new Error('Unknown winner');
          self.outcome = {kind: 'terminal', winner: winner || null, turns,
            rewards: Object.fromEntries(SIDES.map(side => [side, !winner ? .5 : Number(winner === `Full ${side}`)]))};
        }
      }
    }();
  }

  async start({episodeIndex, runSeed, plan = null}) {
    if (this.started) throw new Error('Session already started');
    if (!Number.isSafeInteger(episodeIndex) || episodeIndex < 0 || !Number.isInteger(runSeed) || runSeed < 0 || runSeed >= 2 ** 32) {
      throw new Error('Invalid episode/run seed');
    }
    this.started = true; this.episodeId = `f1:${runSeed}:${episodeIndex}`;
    const admitted = plan ?? officialPlan(episodeIndex, runSeed);
    checkSeed(admitted.seed);
    if (!Array.isArray(admitted.teams) || admitted.teams.length !== 2) throw new Error('Expected two admitted teams');
    for (const entry of admitted.teams) {
      const payload = teamArtifact(entry, '0'.repeat(64)).payload;
      if (entry.team.length !== 6 || entry.partition !== 'train' || partition(payload) !== 'train' || teamKey(payload) !== entry.team_key) {
        throw new Error('Nontraining/mismatched team plan rejected');
      }
    }
    this.provenance = {episode_id: this.episodeId, reference_commit: describe().commit, seed: admitted.seed,
      teams: admitted.teams.map(t => ({seed: t.seed, team_key: t.team_key, partition: t.partition})),
      scenario: structuredClone(admitted.provenance ?? {source: 'official'})};
    await this.stream.write(`>start ${JSON.stringify({formatid: 'gen9randombattle', seed: admitted.seed})}\n` +
      admitted.teams.map((t, i) => `>player p${i + 1} ${JSON.stringify({name: `Full p${i + 1}`, team: t.team})}`).join('\n'));
    await this.passControls();
    for (const row of admitted.prefix ?? []) {
      const snap = this.snapshot(), actions = {}, request_ids = {};
      for (const side of SIDES) if (snap.players[side].mask.some(Boolean)) {
        if (typeof row[side] !== 'string') throw new Error('Prefix omitted pending choice');
        actions[side] = ACTIONS.indexOf(row[side]); request_ids[side] = snap.players[side].request_id;
      }
      if (Object.keys(row).some(s => !Object.hasOwn(actions, s))) throw new Error('Prefix includes nonpending side');
      const result = await this.step({actions, request_ids});
      if (result.submitted.some(row => !row.accepted)) throw new Error('Curriculum prefix choice unavailable');
      if (this.outcome) throw new Error('Curriculum prefix reached terminal state');
    }
    return this.snapshot();
  }

  drain() {
    for (const side of SIDES) {
      const lines = this.projection.players[side].protocol;
      while (this.cursors[side] < lines.length) {
        const line = lines[this.cursors[side]++];
        if (!line.startsWith('|request|')) this.views[side].ingest(line);
      }
    }
  }

  async passControls() {
    for (let round = 0; round < 4 && !this.outcome; round++) {
      let passed = false;
      for (const side of SIDES) {
        const p = this.projection.players[side];
        if (p.pending && requestActions(p.request).pass) {
          p.pending = false; await this.stream.write(`>${side} pass`); passed = true;
        }
      }
      if (!passed) return;
    }
  }

  snapshot() {
    this.drain();
    const players = Object.fromEntries(SIDES.map(side => {
      const p = this.projection.players[side];
      // Preserve own initial request before consuming wait/terminal snapshots.
      if (!this.views[side].lastOwn && p.request?.side) this.views[side].observe(p.request, p.request_id);
      return [side, validatePlayer(this.views[side].observe(!this.outcome && p.pending ? p.request : null, p.request_id))];
    }));
    if (!this.outcome && !SIDES.some(s => players[s].mask.some(Boolean))) throw new Error('Stalled full-Showdown session');
    return {schema_version: 'full-showdown-frame-v1', episode_id: this.episodeId,
      players, outcome: this.outcome, submissions: this.submissions, provenance: this.provenance};
  }

  async step({actions, request_ids}) {
    if (this.outcome) throw new Error('Cannot step a completed session');
    const pending = SIDES.filter(s => this.projection.players[s].pending);
    if (!actions || !request_ids || Object.keys(actions).sort().join() !== pending.join() ||
      Object.keys(request_ids).sort().join() !== pending.join()) throw new Error('Submit exactly the currently pending sides');
    // Validate every selection before submitting either simultaneous action.
    const choices = pending.map(side => {
      const p = this.projection.players[side];
      if (request_ids[side] !== p.request_id) throw new Error('Stale player request');
      return [side, choiceFor(actions[side], p.request)];
    });
    const submitted = [];
    for (const [side, choice] of choices) {
      const p = this.projection.players[side], before = this.projection.errors.length;
      p.pending = false; await this.stream.write(`>${side} ${choice}`); this.submissions++;
      const errors = this.projection.errors.slice(before);
      if (errors.some(e => e.side !== side || !e.message.startsWith('[Unavailable choice]'))) {
        throw new Error(`Reference rejected choice: ${JSON.stringify(errors)}`);
      }
      if (errors.length && !p.pending) throw new Error('Unavailable action without refreshed request');
      // Rejected advertised choices are still actor decisions with observable
      // feedback; retaining them avoids conditioning away sampled action mass.
      submitted.push({side, action: actions[side], accepted: errors.length === 0, feedback: errors.map(e => e.message)});
    }
    await this.passControls();
    return {...this.snapshot(), submitted};
  }

  cutoff() {
    const snap = this.snapshot();
    if (this.outcome || !SIDES.every(s => snap.players[s].phase === 'move' && snap.players[s].mask.some(Boolean))) {
      throw new Error('Cutoff requires a fresh joint move boundary');
    }
    this.outcome = {kind: 'administrative_cutoff', winner: null, turns: snap.players.p1.turn, rewards: {p1: 0, p2: 0}};
    // The collecting policy bootstraps these live observations before close.
    return {...snap, outcome: this.outcome};
  }

  close() { this.stream.battle?.destroy(); }
}
