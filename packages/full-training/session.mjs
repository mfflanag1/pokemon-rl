import {BattleStream, generateTeam, seedFor, describe, checkSeed} from '../reference-node/engine.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {partition, teamKey, pairPartition} from '../dataset/identity.mjs';
import {requestActions, choiceFor} from '../full-showdown/actions.mjs';
import {RosterView} from '../full-showdown/view.mjs';
import {validatePlayer} from '../full-showdown/schema.mjs';

const SIDES = ['p1', 'p2'];
const ZERO_COMMITMENT = '0'.repeat(64);
// F2 episodes are a pure function of (run_seed, episode_index, split, swap).
// There is no plan argument, no curriculum provider and no held-out split:
// `final` stays unused by construction, not by convention.
export const SPLITS = Object.freeze(['train', 'development']);
export const MAX_TEAM_ATTEMPTS = 1000;
const START_OPTIONS = Object.freeze(['episodeIndex', 'runSeed', 'split', 'swap']);

export function episodeStem(episodeIndex, runSeed, split) {
  return `full-showdown-f2/${split}/${runSeed}/${episodeIndex}`;
}

/** Strict admission of the four episode coordinates. Anything else - `final`,
 * an unknown split, a nonboolean swap, a fractional/negative/oversized seed -
 * is refused before a single team is sampled or an engine is created. */
export function validateEpisodeArgs(episodeIndex, runSeed, split, swap) {
  if (!Number.isSafeInteger(episodeIndex) || episodeIndex < 0) throw new Error('Invalid episode index');
  if (!Number.isInteger(runSeed) || runSeed < 0 || runSeed >= 2 ** 32) throw new Error('Invalid run seed');
  if (split === 'final') throw new Error("Split 'final' is never admitted by full-training F2");
  if (typeof split !== 'string' || !SPLITS.includes(split)) {
    throw new Error(`Unknown split rejected: ${JSON.stringify(split ?? null)}`);
  }
  if (typeof swap !== 'boolean') throw new Error('Swap must be a boolean');
  return {episodeIndex, runSeed, split, swap};
}

/** Rejection samples two distinct teams whose frozen R7 partition digest lands
 * in `split`. The attempt stream is namespaced per split, so the train and
 * development pools are disjoint by partition and by sampling namespace.
 * `swap` reverses the very same admitted pair for seat-paired evaluation;
 * the mechanics seed is untouched, so both orientations share one rollout. */
export function officialPlan(episodeIndex, runSeed, split = 'train', swap = false) {
  validateEpisodeArgs(episodeIndex, runSeed, split, swap);
  const namespace = `full-showdown-f2/${split}`, stem = episodeStem(episodeIndex, runSeed, split);
  const teams = [], seen = new Set();
  let attempts = 0;
  for (; teams.length < 2 && attempts < MAX_TEAM_ATTEMPTS; attempts++) {
    const entry = generateTeam(seedFor(`${stem}/team/${attempts}`));
    const payload = teamArtifact(entry, ZERO_COMMITMENT).payload;
    const key = teamKey(payload);
    if (partition(payload) !== split || seen.has(key)) continue;
    seen.add(key); teams.push({...entry, team_key: key, partition: split});
  }
  if (teams.length !== 2) throw new Error(`${split} team admission budget exhausted`);
  return {teams: swap ? [teams[1], teams[0]] : teams, seed: seedFor(`${stem}/mechanics`), split, swap,
    provenance: {source: 'official', partition: split, namespace, episode_index: episodeIndex,
      run_seed: runSeed, swap, attempts, attempt_budget: MAX_TEAM_ATTEMPTS}};
}

/** Identical actor boundary to F1: the same projected roster view, the same 14
 * semantic actions, the same forced-pass control, the same unavailable-choice
 * feedback and the same validated `full-showdown-player-v1` player payload.
 * Only episode admission (two splits, deterministic seat swap) is new. */
export class FullTrainingSession {
  constructor() {
    this.projection = new PlayerProjection();
    this.views = Object.fromEntries(SIDES.map(side => [side, new RosterView(side)]));
    this.cursors = {p1: 0, p2: 0}; this.outcome = null; this.submissions = 0; this.started = false;
    this.split = null; this.swap = null;
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

  // Coordinator-internal seam, overridden only by tests that probe revalidation.
  // Actor input can never reach it: `start` exposes no plan or curriculum field.
  planFor(episodeIndex, runSeed, split, swap) { return officialPlan(episodeIndex, runSeed, split, swap); }

  async start(options = {}) {
    if (this.started) throw new Error('Session already started');
    if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('Expected start options object');
    const unexpected = Object.keys(options).filter(key => !START_OPTIONS.includes(key));
    if (unexpected.length) throw new Error(`Unsupported start option(s): ${unexpected.sort().join(', ')}`);
    const {episodeIndex, runSeed, split = 'train', swap = false} = options;
    validateEpisodeArgs(episodeIndex, runSeed, split, swap);
    this.started = true; this.split = split; this.swap = swap;
    this.episodeId = `f2:${split}:${runSeed}:${episodeIndex}:${swap ? 'swap' : 'base'}`;
    const admitted = this.planFor(episodeIndex, runSeed, split, swap);
    // Revalidate independently of the generator: provenance labels are claims,
    // the partition digest and the team key are the evidence.
    checkSeed(admitted.seed);
    if (!Array.isArray(admitted.teams) || admitted.teams.length !== 2) throw new Error('Expected two admitted teams');
    if (admitted.split !== split || admitted.swap !== swap) throw new Error('Plan split/swap disagrees with the request');
    // Seed representation is the reference node's business, so compare
    // structurally rather than assuming a scalar PRNG seed.
    if (JSON.stringify(admitted.seed) !== JSON.stringify(seedFor(`${episodeStem(episodeIndex, runSeed, split)}/mechanics`))) {
      throw new Error('Nondeterministic mechanics seed rejected');
    }
    const payloads = admitted.teams.map(entry => {
      if (!Array.isArray(entry?.team) || entry.team.length !== 6) throw new Error('Expected six-member admitted team');
      return teamArtifact(entry, ZERO_COMMITMENT).payload;
    });
    admitted.teams.forEach((entry, index) => {
      if (entry.partition !== split || partition(payloads[index]) !== split) {
        throw new Error(`Team outside the admitted ${split} partition rejected`);
      }
      if (teamKey(payloads[index]) !== entry.team_key) throw new Error('Mismatched team key rejected');
    });
    if (pairPartition(payloads[0], payloads[1]) !== split) throw new Error('Cross-partition matchup rejected');
    if (admitted.teams[0].team_key === admitted.teams[1].team_key) throw new Error('Duplicate admitted teams rejected');
    this.provenance = {episode_id: this.episodeId, reference_commit: describe().commit, seed: admitted.seed,
      split, swap, teams: admitted.teams.map(t => ({seed: t.seed, team_key: t.team_key, partition: t.partition})),
      scenario: structuredClone(admitted.provenance ?? {source: 'official'})};
    await this.stream.write(`>start ${JSON.stringify({formatid: 'gen9randombattle', seed: admitted.seed})}\n` +
      admitted.teams.map((t, i) => `>player p${i + 1} ${JSON.stringify({name: `Full p${i + 1}`, team: t.team})}`).join('\n'));
    await this.passControls();
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
      // Public replace events may omit HP. An absent update must not erase
      // previously known private HP when rendering terminal observations.
      if (this.outcome) for (const event of this.views[side].ownEvents.values()) {
        if (event.condition === undefined) delete event.condition;
      }
      // Wait requests can contain authoritative own-team updates (including
      // Illusion replacement). Keep that private side payload while suppressing
      // action availability; a nonpending player must never act on an old request.
      const request = this.outcome ? null : p.pending ? p.request :
        p.request?.side ? {wait: true, side: p.request.side} : null;
      const player = this.views[side].observe(request, p.request_id);
      // A fainted member's request may clear terastallized; retain its reveal.
      for (const own of player.own) {
        const revealed = this.views[side].ownEvents.get(own.ident)?.tera;
        if (revealed) own.tera = revealed;
      }
      return [side, validatePlayer(player)];
    }));
    if (!this.outcome && !SIDES.some(s => players[s].mask.some(Boolean))) throw new Error('Stalled full-training session');
    // Frames carry projected players only; engine state is never serialised out.
    return {schema_version: 'full-training-frame-v1', episode_id: this.episodeId,
      split: this.split, swap: this.swap, players, outcome: this.outcome,
      submissions: this.submissions, provenance: this.provenance};
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
