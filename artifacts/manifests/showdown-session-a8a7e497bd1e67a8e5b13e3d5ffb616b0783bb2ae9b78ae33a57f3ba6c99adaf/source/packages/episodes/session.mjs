import {ReferenceSession, defaultSpec} from '../reference-node/runner.mjs';
import {createHash} from 'node:crypto';
import {RandomPlayerAI} from '../reference-node/engine.mjs';
import {normalizeProtocol} from '../reference-node/observations.mjs';
import {canonical, hashJSON, parseJSON} from '../battle-contract/canonical.mjs';
import {actionMask, admissibleActions, fromProtocol, requestPhase, toProtocol} from '../battle-contract/actions.mjs';
import {episodeManifest} from './manifest.mjs';
import {SIDES, captureFrame} from './phases.mjs';
import {applyFixture, FIXTURES} from './fixtures.mjs';
import {validateEpisode} from './contract.mjs';

export const normalizedOutput = raw => raw.map(({type, data}) => ({type, data: type === 'end' ? data : normalizeProtocol(data)}));
export const decisionId = (key, body) => hashJSON(['r4-decision-1', key, body]);

export function observation(view, from, prefixHash = hashJSON(view.protocol)) {
  const body = {viewer: view.side, request_id: view.request_id, turn: view.turn,
    phase: requestPhase(view.request), request: structuredClone(view.request),
    protocol_start: from, protocol_end: view.protocol.length,
    protocol_delta: view.protocol.slice(from), protocol_prefix_hash: prefixHash,
    admissible_actions: admissibleActions(view.request), action_mask: actionMask(view.request)};
  return {...body, view_id: hashJSON(body)};
}

export class EpisodeSession {
  constructor(spec, {fixture = null, manifest = null, controller = 'scripted-semantic-choices-r4', order = 'external'} = {}) {
    if (fixture !== null && !FIXTURES.includes(fixture)) throw new Error('Unknown fixture');
    this.fixture = fixture;
    if (![spec.policy_id, 'scripted-semantic-choices-r4'].includes(controller)) throw new Error('Unknown controller');
    this.controller = controller;
    if (!['p1-first', 'p2-first', 'external'].includes(order)) throw new Error('Unknown submission order');
    this.order = order;
    const current = episodeManifest(fixture === null ? 'official' : 'constructed');
    if (manifest && canonical(manifest) !== canonical(current)) throw new Error('R4 manifest differs from current runtime');
    this.manifest = current;
    if (this.manifest.profile !== (fixture === null ? 'official' : 'constructed')) throw new Error('Fixture/profile mismatch');
    this.reference = new ReferenceSession(parseJSON(canonical(spec)));
    this.key = hashJSON({spec, fixture, controller, submission_order: order, manifest_id: this.manifest.manifest_id});
    this.decisions = []; this.frames = []; this.receipts = new Map();
    this.cursors = {p1: 0, p2: 0}; this.truncated = false;
    this.history = Object.fromEntries(SIDES.map(s => [s, {length: 0, hash: createHash('sha256').update('[')}]));
    this.queue = Promise.resolve(); this.queued = 0; this.failed = false; this.started = false; this.closed = false;
  }

  async start() {
    if (this.started || this.closed) throw new Error('Session already started/closed');
    this.started = true;
    await this.reference.start();
    applyFixture(this.reference, this.fixture, 'initial');
    this.frames.push(captureFrame(this.reference, 0, 'initial', {p1: null, p2: null}));
    return this;
  }

  view(side) {
    if (!SIDES.includes(side)) throw new TypeError('Unknown side');
    const player = this.reference.projection.players[side], history = this.history[side];
    while (history.length < player.protocol.length) {
      if (history.length) history.hash.update(',');
      history.hash.update(canonical(player.protocol[history.length++]));
    }
    return observation(player, this.cursors[side], history.hash.copy().update(']').digest('hex'));
  }

  submit(submission) {
    const copy = structuredClone(submission);
    this.queued++;
    const operation = this.queue.then(async () => {
      const writes = this.reference.input_tape.length;
      try { return await this.submitOnce(copy); }
      catch (error) {
        if (this.reference.input_tape.length !== writes) this.failed = true;
        throw error;
      }
    });
    this.queue = operation.catch(() => {});
    return operation.finally(() => { this.queued--; });
  }

  async submitOnce(submission) {
    if (!submission || typeof submission.submission_id !== 'string' || !submission.submission_id || submission.submission_id.length > 200) throw new TypeError('Need bounded submission ID');
    if (Object.keys(submission).sort().join(',') !== 'action,request_id,side,submission_id,view_id') throw new TypeError('Unexpected submission fields');
    const fingerprint = hashJSON(submission), previous = this.receipts.get(submission.submission_id);
    if (previous) {
      if (previous.fingerprint !== fingerprint) throw new Error('Submission ID reused for a different action');
      return structuredClone(previous.record); // A transport retry is not another decision.
    }
    if (this.failed || !this.started || this.closed || this.truncated || this.reference.result) throw new Error('Episode is inactive or failed');
    const before = this.frames.at(-1), side = submission.side;
    if (!before.pending_sides.includes(side)) throw new Error('Stale or inactive request');
    if (this.decisions.length >= this.reference.spec.max_submissions) throw new Error('Submission safety cap; incomplete episode');
    const view = this.view(side);
    if (submission.request_id !== view.request_id || submission.view_id !== view.view_id) throw new Error('Stale or mismatched view');
    const command = toProtocol(submission.action, view.request);
    const sequence = this.decisions.length;
    const result = await this.reference.submit(side, view.request_id, command);
    if (sequence === 0) {
      if (this.fixture === 'tie-after-first-choice' && !result.accepted) throw new Error('Constructed tie requires an accepted first submission');
      applyFixture(this.reference, this.fixture, 'after-first-choice');
    }
    const changed = SIDES.filter(s => this.reference.projection.players[s].request_id !== before.requests[s].request_id);
    const terminal = Boolean(this.reference.result), nextGroup = result.accepted && changed.length > 0;
    const commitments = terminal || nextGroup ? {p1: null, p2: null} : {...before.commitments};
    if (result.accepted && !terminal && !nextGroup) commitments[side] = sequence;
    const event = terminal ? 'terminal' : !result.accepted ? 'rerequest' : nextGroup ? 'resolved' : 'awaiting_choice';
    const after = captureFrame(this.reference, before.group + Number(nextGroup), event, commitments);
    if (!result.accepted) {
      const other = side === 'p1' ? 'p2' : 'p1';
      if (canonical(changed) !== canonical([side]) || before.turn !== after.turn ||
          before.audit.mechanics_hash !== after.audit.mechanics_hash ||
          before.audit.choices[other] !== after.audit.choices[other] ||
          before.requests[other].request_id !== after.requests[other].request_id) throw new Error('Re-request advanced mechanics or changed opponent commitment');
    }
    const body = {sequence, submission_id: submission.submission_id,
      side, frame_before: sequence, frame_after: sequence + 1, view,
      action: structuredClone(submission.action), command, accepted: result.accepted, feedback: result.errors};
    const record = {...body, decision_id: decisionId(this.key, body)};
    this.cursors[side] = view.protocol_end;
    this.decisions.push(record); this.frames.push(after);
    this.receipts.set(submission.submission_id, {fingerprint, record: structuredClone(record)});
    return structuredClone(record);
  }

  canTruncate() {
    const frame = this.frames.at(-1);
    return !this.queued && !this.failed && !this.closed && !this.reference.result && frame?.phase === 'joint_move' &&
      ['initial', 'resolved'].includes(frame.event) && frame.pending_sides.length === 2 &&
      SIDES.every(side => frame.commitments[side] === null);
  }

  truncate() {
    if (!this.canTruncate()) throw new Error('Truncation requires a fresh uncommitted joint boundary');
    this.truncated = true;
  }

  artifact() {
    if (this.queued || this.failed || this.closed) throw new Error('In-flight, failed or closed episode');
    if (!this.reference.result && !this.truncated) throw new Error('Incomplete episode');
    const source = this.reference, final = this.frames.length - 1;
    const winner = source.result ? ({'Player 1': 'p1', 'Player 2': 'p2', '': null})[source.result.winner] : null;
    if (winner === undefined) throw new Error('Unknown winner');
    const outcome = {status: this.truncated ? 'truncated' : 'terminal', reason: this.truncated ? 'decision_budget' : winner === null ? 'tie' : 'win',
      winner, turn: source.stream.battle.turn};
    const bootstrap_views = this.truncated ? Object.fromEntries(SIDES.map(s => [s, this.view(s)])) : null;
    const next = {p1: null, p2: null};
    const transitions = this.decisions.map(() => null);
    for (let i = this.decisions.length - 1; i >= 0; i--) {
      const d = this.decisions[i], following = next[d.side], terminal = following === null && !this.truncated;
      const target = following !== null ? {kind: 'decision', sequence: following, id: this.decisions[following].decision_id} :
        this.truncated ? {kind: 'bootstrap', frame: final, id: bootstrap_views[d.side].view_id} : {kind: 'terminal', frame: final};
      transitions[i] = {decision_id: d.decision_id, side: d.side, target, terminal,
        truncated: following === null && this.truncated, reward_numerator: terminal ? winner === null ? 1 : winner === d.side ? 2 : 0 : 0,
        reward_denominator: 2};
      next[d.side] = i;
    }
    const payload = {episode_key: this.key, spec: source.spec, fixture: this.fixture, controller: this.controller, submission_order: this.order, split: 'unassigned',
      initial_team_hashes: source.initial_team_hashes, input_tape: source.input_tape,
      frames: this.frames, decisions: this.decisions, transitions, bootstrap_views, outcome,
      protocol: Object.fromEntries(SIDES.map(s => [s, source.projection.players[s].protocol])),
      spectator: source.projection.spectator, normalized_output_hash: hashJSON(normalizedOutput(source.raw))};
    const body = {schema_version: '1.0.0', artifact_kind: 'phase_episode', manifest_id: this.manifest.manifest_id, payload};
    const content_hash = hashJSON(body);
    const episode = {...body, content_hash, artifact_id: `phase_episode:${content_hash}`};
    const bundle = {episode, raw_output: source.raw, raw_output_hash: hashJSON(source.raw)};
    return validateEpisode(bundle, this.manifest);
  }

  close() {
    if (this.queued) throw new Error('Cannot close with submissions in flight');
    this.closed = true; this.reference.close();
  }
}

export async function runPhaseEpisode(spec = defaultSpec('r4-demo'), {manifest = null, fixture = null, truncateAfter = null, order = 'p1-first', replay = null} = {}) {
  if (truncateAfter !== null && (!Number.isSafeInteger(truncateAfter) || truncateAfter < 0)) throw new Error('Invalid truncation budget');
  if (!['p1-first', 'p2-first'].includes(order)) throw new Error('Invalid submission order');
  const session = new EpisodeSession(spec, {fixture, manifest, controller: replay?.episode.payload.controller ?? spec.policy_id,
    order: replay?.episode.payload.submission_order ?? order});
  const agents = spec.policy_seeds.map(seed => {
    let selected;
    const ai = new RandomPlayerAI({write(choice) { selected = choice; }}, {seed, move: 0.9, mega: 0});
    return request => { selected = undefined; ai.receiveRequest(structuredClone(request)); return selected; };
  });
  try {
    await session.start();
    while (!session.reference.result) {
      if (truncateAfter !== null && session.decisions.length >= truncateAfter && session.canTruncate()) { session.truncate(); break; }
      const expected = replay?.episode.payload.decisions[session.decisions.length];
      const pending = session.frames.at(-1).pending_sides;
      const side = replay ? expected?.side : (order === 'p1-first' ? SIDES : [...SIDES].reverse()).find(s => pending.includes(s));
      if (!side || !pending.includes(side)) throw new Error('Missing or stale replay decision / stalled episode');
      const view = session.view(side);
      const action = replay ? expected.action : fromProtocol(agents[Number(side[1]) - 1](view.request), view.request);
      const record = await session.submit({submission_id: expected?.submission_id ?? `${session.key}/${session.decisions.length}`,
        side, request_id: view.request_id, view_id: view.view_id, action});
      if (replay && canonical(record) !== canonical(expected)) throw new Error('Replay decision diverged');
    }
    const artifact = session.artifact();
    if (replay && artifact.episode.content_hash !== replay.episode.content_hash) throw new Error('Replay phases, mechanics, protocol or outcome diverged');
    return artifact;
  } finally { session.close(); }
}

export async function replayPhaseEpisode(bundle, manifest = episodeManifest(bundle.episode.payload.fixture === null ? 'official' : 'constructed')) {
  validateEpisode(bundle, manifest);
  const p = bundle.episode.payload;
  return runPhaseEpisode(p.spec, {manifest, fixture: p.fixture, replay: bundle,
    truncateAfter: p.outcome.status === 'truncated' ? p.decisions.length : null});
}
