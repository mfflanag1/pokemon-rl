import {BattleStream, RandomPlayerAI, describe, generateTeam, checkSeed, seedFor} from './engine.mjs';
import {pin, hashJSON} from './provenance.mjs';
import {PlayerProjection, normalizeProtocol, admissibleChoices} from './observations.mjs';

export function defaultSpec(label = 'r1-smoke') {
  return {
    format_id: pin.format_id,
    teams: [generateTeam(seedFor(`${label}/team/0`)), generateTeam(seedFor(`${label}/team/1`))],
    mechanics_seed: seedFor(`${label}/mechanics`),
    policy_seeds: [seedFor(`${label}/policy/0`), seedFor(`${label}/policy/1`)],
    policy_id: 'upstream-random-move-0.9-no-tera-r1',
    max_submissions: 20000,
  };
}

function validateSpec(spec) {
  if (spec.format_id !== pin.format_id) throw new Error('Only pinned gen9randombattle is supported');
  checkSeed(spec.mechanics_seed);
  if (!Array.isArray(spec.policy_seeds) || spec.policy_seeds.length !== 2) throw new Error('Need two independent policy seeds');
  spec.policy_seeds.forEach(checkSeed);
  if (spec.policy_id !== 'upstream-random-move-0.9-no-tera-r1') throw new Error('Unknown baseline policy');
  if (!Number.isInteger(spec.max_submissions) || spec.max_submissions < 1 || spec.max_submissions > 100000) throw new Error('Invalid submission safety cap');
  if (!Array.isArray(spec.teams) || spec.teams.length !== 2) throw new Error('Need two teams');
  for (const entry of spec.teams) {
    checkSeed(entry.seed);
    if (!Array.isArray(entry.team) || entry.team.length !== 6 || hashJSON(entry.team) !== entry.team_hash) throw new Error('Team hash/shape mismatch');
    // Fixed teams must really be products of the pinned generator, not arbitrary injected sets.
    if (generateTeam(entry.seed).team_hash !== entry.team_hash) throw new Error('Team does not match its generator seed');
  }
}

export class ReferenceSession {
  constructor(spec) {
    validateSpec(spec);
    this.spec = structuredClone(spec);
    this.projection = new PlayerProjection();
    this.raw = [];
    this.decisions = [];
    this.input_tape = [];
    this.result = null;
    const session = this;
    this.stream = new class extends BattleStream {
      constructor() { super({noCatch: true}); }
      pushMessage(type, data) {
        session.raw.push({type, data});
        session.projection.ingest(type, data);
        if (type === 'end') session.result = JSON.parse(data);
      }
    }();
  }

  async start() {
    const initial = [
      `>start ${JSON.stringify({formatid: this.spec.format_id, seed: this.spec.mechanics_seed})}`,
      ...this.spec.teams.map((entry, i) => `>player p${i + 1} ${JSON.stringify({name: `Player ${i + 1}`, team: entry.team})}`),
    ].join('\n');
    await this.write(initial);
    this.initial_team_hashes = this.stream.battle.sides.map(side => hashJSON(side.team));
    if (hashJSON(this.initial_team_hashes) !== hashJSON(this.spec.teams.map(entry => entry.team_hash))) throw new Error('Simulator changed supplied team data');
    return this;
  }

  async write(command) {
    this.input_tape.push(command);
    await this.stream.write(command);
  }

  async submit(side, requestId, choice) {
    if (!['p1', 'p2'].includes(side)) throw new Error('Unknown side');
    const player = this.projection.players[side];
    if (this.result || !player.pending || player.request_id !== requestId) throw new Error('Stale or inactive request');
    // Never expose BattleStream's eval/edit/reseed control commands through a choice tape.
    if (typeof choice !== 'string' || !/^(move [1-4]( terastallize)?|switch [1-6]|pass)$/.test(choice)) throw new Error('Unsupported choice syntax');
    const choices = admissibleChoices(player.request);
    if (!choices.includes(choice)) throw new Error('Choice is forbidden by the visible request');
    const record = {
      side, request_id: requestId, turn: player.turn,
      phase: player.request.forceSwitch ? 'switch' : 'move',
      protocol_length: player.protocol.length,
      request_hash: hashJSON(player.request), admissible_choices: choices, choice,
    };
    player.pending = false;
    const errorStart = this.projection.errors.length;
    await this.write(`>${side} ${choice}`);
    record.errors = this.projection.errors.slice(errorStart);
    if (record.errors.some(error => error.side !== side)) throw new Error('Unexpected error for a different player during submission');
    record.accepted = record.errors.length === 0;
    this.decisions.push(record);
    const invalid = record.errors.find(error => !error.message.startsWith('[Unavailable choice]'));
    if (invalid) throw new Error(`Reference rejected invalid policy choice: ${invalid.message}`);
    if (!record.accepted && !player.pending) throw new Error('Unavailable choice without a replacement request');
    return record;
  }

  artifact(manifest) {
    if (!this.result) throw new Error('Incomplete battle; safety caps are not draws');
    const normalized = this.raw.map(({type, data}) => ({type, data: type === 'end' ? data : normalizeProtocol(data)}));
    const body = {
      schema_version: 'r1-episode-1', manifest_id: manifest.manifest_id,
      spec: this.spec, initial_team_hashes: this.initial_team_hashes,
      input_tape: this.input_tape, decisions: this.decisions,
      protocol: {
        p1: this.projection.players.p1.protocol,
        p2: this.projection.players.p2.protocol,
        spectator: this.projection.spectator,
      },
      outcome: {winner: this.result.winner, turns: this.result.turns, score: this.result.score},
      normalized_output_hash: hashJSON(normalized),
    };
    return {...body, content_hash: hashJSON(body), raw_output: this.raw, raw_output_hash: hashJSON(this.raw)};
  }

  close() {
    // No reader is attached to our recording stream; EOF already closes its queue.
    this.stream.battle?.destroy();
    this.stream.battle = null;
  }
}

export async function runEpisode(spec = defaultSpec(), replay = null) {
  const manifest = describe();
  if (replay && replay.manifest_id !== manifest.manifest_id) throw new Error('Replay manifest differs from this runtime/build');
  const session = new ReferenceSession(spec);
  const agents = spec.policy_seeds.map(seed => {
    let selected;
    const ai = new RandomPlayerAI({write(choice) { selected = choice; }}, {seed, move: 0.9, mega: 0});
    return request => { selected = undefined; ai.receiveRequest(structuredClone(request)); return selected; };
  });
  try {
    await session.start();
    while (!session.result) {
      if (session.decisions.length >= spec.max_submissions) throw new Error('Submission cap reached; battle is incomplete, not a draw');
      const side = ['p1', 'p2'].find(id => session.projection.players[id].pending);
      if (!side) throw new Error('Battle stalled without a player request');
      const view = session.projection.view(side);
      const expected = replay?.decisions[session.decisions.length];
      if (replay && (!expected || expected.side !== side || expected.request_id !== view.request_id)) throw new Error('Replay decision order diverged');
      const choice = replay ? expected.choice : agents[Number(side[1]) - 1](view.request);
      const record = await session.submit(side, view.request_id, choice);
      if (replay && hashJSON(record) !== hashJSON(expected)) throw new Error('Replay observation or action result diverged');
    }
    const artifact = session.artifact(manifest);
    if (replay && artifact.content_hash !== replay.content_hash) throw new Error('Replay transcript/result mismatch');
    return artifact;
  } finally { session.close(); }
}

export async function replayEpisode(episode) {
  if (episode.schema_version !== 'r1-episode-1') throw new Error('Unsupported episode schema');
  const {raw_output, raw_output_hash, content_hash, ...body} = episode;
  if (hashJSON(raw_output) !== raw_output_hash) throw new Error('Raw output hash mismatch');
  const normalized = raw_output.map(({type, data}) => ({type, data: type === 'end' ? data : normalizeProtocol(data)}));
  if (hashJSON(normalized) !== body.normalized_output_hash) throw new Error('Raw trace disagrees with normalized output');
  if (hashJSON(body) !== content_hash) throw new Error('Episode content hash mismatch');
  return runEpisode(episode.spec, episode);
}
