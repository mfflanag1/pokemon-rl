import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import Ajv2020 from 'ajv/dist/2020.js';
import {schema as battleSchema} from '../battle-contract/schema.mjs';
import {validateArtifact} from '../battle-contract/artifacts.mjs';
import {canonical, hashJSON, parseJSON} from '../battle-contract/canonical.mjs';
import {actionMask, admissibleActions, requestPhase, toProtocol} from '../battle-contract/actions.mjs';
import {PlayerProjection, normalizeProtocol} from '../reference-node/observations.mjs';
import {phaseFor, SIDES} from './phases.mjs';

const schema = parseJSON(readFileSync(new URL('../../contracts/episode-v1/schema.json', import.meta.url)));
const ajv = new Ajv2020({strict: true});
ajv.addSchema(battleSchema);
const validate = ajv.compile(schema);
const equal = (a, b, message) => { if (canonical(a) !== canonical(b)) throw new TypeError(message); };
const check = (condition, message) => { if (!condition) throw new TypeError(message); };

export function validateManifest(manifest) {
  const {manifest_id, ...body} = manifest;
  check(hashJSON(body) === manifest_id && body.version === 'r4-manifest-1', 'R4 manifest identity/version');
  check(['official', 'constructed'].includes(body.profile), 'R4 profile');
  check(body.phase_version === 'r4-phases-1' && body.audit_version === 'r4-selected-mechanics-1', 'R4 semantics version');
  validateArtifact(body.reference_contract, {kind: 'manifest'});
  return manifest;
}

function historyIndex(lines, boundaries, side) {
  let request = null, request_id = 0, turn = 0;
  const wanted = new Set(boundaries), states = new Map(), hash = createHash('sha256').update('[');
  for (let i = 0; i <= lines.length; i++) {
    if (wanted.has(i)) states.set(i, {request, request_id, turn, protocol_hash: hash.copy().update(']').digest('hex')});
    if (i === lines.length) break;
    const line = lines[i];
    if (i) hash.update(',');
    hash.update(canonical(line));
    if (line.startsWith('|request|')) {
      request = parseJSON(line.slice(9)); request_id++;
      check(request?.side?.id === side, 'Foreign private request in history');
    }
    if (line.startsWith('|turn|')) { check(/^\|turn\|[0-9]+$/.test(line), 'Malformed turn'); turn = Number(line.slice(6)); }
  }
  return states;
}

function validateView(view, side, boundary, cursor, protocol, states) {
  const state = states.get(boundary.protocol_length);
  check(view.viewer === side && view.request.side.id === side && view.request.side.pokemon.every(p => p.ident.startsWith(`${side}: `)), 'View owner');
  check(view.protocol_start === cursor && view.protocol_end === boundary.protocol_length && view.protocol_end <= protocol.length, 'View delta boundary');
  equal(view.protocol_delta, protocol.slice(cursor, boundary.protocol_length), 'View delta differs');
  check(view.protocol_prefix_hash === state.protocol_hash, 'View prefix hash');
  equal(view.request, state.request, 'View request differs from protocol');
  check(view.request_id === state.request_id && view.turn === state.turn && view.phase === requestPhase(view.request), 'View request metadata');
  equal(view.admissible_actions, admissibleActions(view.request), 'Visible action set');
  equal(view.action_mask, actionMask(view.request), 'Visible action mask');
  const {view_id, ...body} = view;
  check(view_id === hashJSON(body), 'View identity');
}

export function validateEpisode(bundle, manifest) {
  canonical(bundle);
  if (!validate(bundle)) throw new TypeError(`R4 schema: ${ajv.errorsText(validate.errors)}`);
  validateManifest(manifest);
  const e = bundle.episode, p = e.payload;
  check(e.manifest_id === manifest.manifest_id, 'Episode manifest mismatch');
  check(manifest.profile === (p.fixture === null ? 'official' : 'constructed'), 'Fixture profile mismatch');
  const {content_hash, artifact_id, ...body} = e;
  check(content_hash === hashJSON(body) && artifact_id === `phase_episode:${content_hash}`, 'Episode identity');
  check(p.episode_key === hashJSON({spec: p.spec, fixture: p.fixture, controller: p.controller, submission_order: p.submission_order, manifest_id: e.manifest_id}), 'Episode key');
  p.spec.teams.forEach((entry, i) => check(entry.team_hash === hashJSON(entry.team) && entry.team_hash === p.initial_team_hashes[i], 'Initial team identity'));
  check(bundle.raw_output_hash === hashJSON(bundle.raw_output), 'Raw evidence hash');
  const normalized = bundle.raw_output.map(({type, data}) => ({type, data: type === 'end' ? data : normalizeProtocol(data)}));
  check(p.normalized_output_hash === hashJSON(normalized), 'Normalized output hash');
  const projection = new PlayerProjection();
  const ends = bundle.raw_output.filter(row => row.type === 'end');
  for (const row of bundle.raw_output) projection.ingest(row.type, row.data);
  SIDES.forEach(s => equal(p.protocol[s], projection.players[s].protocol, 'Protocol/raw disagreement'));
  equal(p.spectator, projection.spectator, 'Spectator/raw disagreement');
  const terminal = p.outcome.status === 'terminal';
  check(ends.length === Number(terminal), 'Terminal/raw disagreement');
  if (terminal) {
    const end = parseJSON(ends[0].data), winner = {'Player 1': 'p1', 'Player 2': 'p2', '': null}[end.winner];
    check(winner !== undefined && p.outcome.winner === winner && p.outcome.reason === (winner === null ? 'tie' : 'win') && end.turns === p.outcome.turn, 'Terminal outcome disagreement');
    check(p.bootstrap_views === null, 'Terminal has bootstrap views');
  } else check(p.outcome.reason === 'decision_budget' && p.outcome.winner === null && p.bootstrap_views !== null, 'Truncation outcome');
  check(p.frames.length === p.decisions.length + 1 && p.transitions.length === p.decisions.length, 'Decision/frame/transition count');
  check(p.input_tape.length === p.decisions.length + 1 && p.decisions.length <= p.spec.max_submissions, 'Input tape count/cap');
  const sortedSpec = parseJSON(canonical(p.spec));
  const setup = [`>start ${JSON.stringify({formatid: sortedSpec.format_id, seed: sortedSpec.mechanics_seed})}`,
    ...sortedSpec.teams.map((entry, i) => `>player p${i + 1} ${JSON.stringify({name: `Player ${i + 1}`, team: entry.team})}`)].join('\n');
  check(p.input_tape[0] === setup, 'Initial input tape differs');
  const cursors = {p1: 0, p2: 0}, submissionIds = new Set();
  const histories = Object.fromEntries(SIDES.map(s => [s, historyIndex(p.protocol[s], p.frames.map(f => f.requests[s].protocol_length), s)]));
  for (const [i, frame] of p.frames.entries()) {
    const requests = {};
    for (const side of SIDES) {
      const boundary = frame.requests[side], prior = p.frames[i - 1]?.requests[side];
      check(boundary.protocol_length <= p.protocol[side].length && (!prior || boundary.protocol_length >= prior.protocol_length), 'Frame protocol boundary');
      const state = histories[side].get(boundary.protocol_length);
      check(state.request_id === boundary.request_id && hashJSON(state.request) === boundary.request_hash && state.turn === frame.turn, 'Frame request boundary');
      requests[side] = state.request;
    }
    const isTerminal = terminal && i === p.frames.length - 1;
    const classified = phaseFor(requests, isTerminal);
    for (const key of ['phase', 'side_phases', 'requested_sides']) equal(frame[key], classified[key], 'Frame phase mismatch');
    check(frame.audit.mechanics_hash === hashJSON(frame.audit.mechanics) && frame.audit.mechanics.turn === frame.turn, 'Audit state/hash');
    equal(frame.audit.mechanics.sides.map(s => s.id), SIDES, 'Audit side order');
    equal(frame.pending_sides, frame.requested_sides.filter(s => frame.commitments[s] === null), 'Pending/commitment disagreement');
    if (i === 0) {
      check(frame.group === 0 && frame.event === 'initial', 'Initial frame');
      equal(frame.commitments, {p1: null, p2: null}, 'Initial commitments');
    } else {
      const before = p.frames[i - 1], d = p.decisions[i - 1];
      const changed = SIDES.filter(s => frame.requests[s].request_id !== before.requests[s].request_id);
      const nextGroup = d.accepted && changed.length > 0;
      const event = isTerminal ? 'terminal' : !d.accepted ? 'rerequest' : nextGroup ? 'resolved' : 'awaiting_choice';
      check(frame.group === before.group + Number(nextGroup) && frame.event === event, 'Group/event linkage');
      const commitments = isTerminal || nextGroup ? {p1: null, p2: null} : {...before.commitments};
      if (d.accepted && !isTerminal && !nextGroup) commitments[d.side] = i - 1;
      equal(frame.commitments, commitments, 'Commitment linkage');
      if (!d.accepted) {
        const other = d.side === 'p1' ? 'p2' : 'p1';
        equal(changed, [d.side], 'Re-request side');
        check(frame.turn === before.turn && frame.audit.mechanics_hash === before.audit.mechanics_hash &&
          frame.audit.choices[other] === before.audit.choices[other], 'Rejection changed mechanics/opponent choice');
      }
    }
  }
  for (const [i, d] of p.decisions.entries()) {
    const before = p.frames[i], after = p.frames[i + 1];
    check(d.sequence === i && d.frame_before === i && d.frame_after === i + 1 && before.pending_sides.includes(d.side), 'Decision boundary/order');
    const {decision_id, ...decisionBody} = d;
    check(decision_id === hashJSON(['r4-decision-1', p.episode_key, decisionBody]), 'Decision identity');
    check(!submissionIds.has(d.submission_id), 'Repeated transport submission'); submissionIds.add(d.submission_id);
    validateView(d.view, d.side, before.requests[d.side], cursors[d.side], p.protocol[d.side], histories[d.side]);
    cursors[d.side] = d.view.protocol_end;
    check(d.command === toProtocol(d.action, d.view.request) && p.input_tape[i + 1] === `>${d.side} ${d.command}`, 'Action/input mismatch');
    const feedback = SIDES.flatMap(s => p.protocol[s].slice(before.requests[s].protocol_length, after.requests[s].protocol_length)
      .filter(l => l.startsWith('|error|')).map(l => ({side: s, message: l.slice(7)})));
    equal(d.feedback, feedback, 'Feedback/protocol mismatch');
    check(d.accepted === (d.feedback.length === 0) && d.feedback.every(f => f.side === d.side && f.message.startsWith('[Unavailable choice]')), 'Invalid choice feedback');
  }
  const last = p.frames.at(-1);
  check(last.turn === p.outcome.turn, 'Final turn');
  SIDES.forEach(s => check(last.requests[s].protocol_length === p.protocol[s].length, 'Unaccounted final protocol'));
  if (!terminal) {
    check(last.phase === 'joint_move' && ['initial', 'resolved'].includes(last.event) && last.pending_sides.length === 2, 'Unsafe truncation boundary');
    SIDES.forEach(s => validateView(p.bootstrap_views[s], s, last.requests[s], cursors[s], p.protocol[s], histories[s]));
  }
  const next = {p1: null, p2: null};
  for (let i = p.decisions.length - 1; i >= 0; i--) {
    const d = p.decisions[i], following = next[d.side], end = following === null && terminal;
    const target = following !== null ? {kind: 'decision', sequence: following, id: p.decisions[following].decision_id} :
      terminal ? {kind: 'terminal', frame: p.frames.length - 1} : {kind: 'bootstrap', frame: p.frames.length - 1, id: p.bootstrap_views[d.side].view_id};
    equal(p.transitions[i], {decision_id: d.decision_id, side: d.side, target, terminal: end,
      truncated: following === null && !terminal, reward_numerator: end ? p.outcome.winner === null ? 1 : p.outcome.winner === d.side ? 2 : 0 : 0,
      reward_denominator: 2}, 'Same-player transition/reward linkage');
    if (!d.accepted) check(following !== null, 'Rejected action lacks same-player re-request decision');
    next[d.side] = i;
  }
  return bundle;
}
