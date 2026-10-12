"""Independent validation of R4 phase episode bundles.

This is a pure-Python re-derivation of ``packages/episodes/contract.mjs``. It never
runs the simulator, never spawns Node, and never builds model features; it only
re-checks that a stored bundle is internally consistent with its own raw evidence:

* episode-v1 schema (with the battle-v1 schema resolved from an offline registry),
* manifest / envelope / team / view / decision / audit hashes,
* the raw simulator output projected through the pinned ``|split|`` channel rule,
* exact per-player view deltas, prefix hashes, admissible action sets and masks,
* request-group, commitment and rejection (re-request) bookkeeping,
* same-player next-decision links with terminal / truncation rewards.

``validate_episode`` returns the bundle unchanged on success and raises
``EpisodeValidationError`` on any corruption. Memory use is bounded by one episode.
"""

import argparse
import gzip
from hashlib import sha256
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from .contract import (
    CONTRACT, SCHEMA as BATTLE_SCHEMA, action_mask, admissible_actions, canonical,
    hash_json, parse_json, request_phase, validate_artifact,
)

EPISODE_CONTRACT = CONTRACT.parent / 'episode-v1'
EPISODE_SCHEMA = parse_json((EPISODE_CONTRACT / 'schema.json').read_bytes())
SIDES = ('p1', 'p2')

Draft202012Validator.check_schema(EPISODE_SCHEMA)
# Offline resolution: the episode schema references the battle schema by absolute $id.
_REGISTRY = Registry().with_resources([
    (BATTLE_SCHEMA['$id'], Resource.from_contents(BATTLE_SCHEMA, default_specification=DRAFT202012)),
    (EPISODE_SCHEMA['$id'], Resource.from_contents(EPISODE_SCHEMA, default_specification=DRAFT202012)),
])
_VALIDATOR = Draft202012Validator(EPISODE_SCHEMA, registry=_REGISTRY)


class EpisodeValidationError(ValueError):
    """Raised when an R4 episode bundle or manifest fails validation."""


_MISSING = object()


def _check(condition, message):
    if not condition:
        raise EpisodeValidationError(message)


def _same(a, b):
    """Strict JSON equality (no bool/int or int/str coercion)."""
    return canonical(a) == canonical(b)


def _equal(a, b, message):
    if not _same(a, b):
        raise EpisodeValidationError(message)


# --- Raw protocol channel projection ------------------------------------------------

# Mirror of the pinned Showdown splitter: /^\|split\|p([1234])\n(.*)\n(.*)|.+/gm.
# JavaScript "." excludes \n, \r, \u2028, \u2029 and multiline "^" matches after any of them.
_NOT_EOL = r'[^\n\r\u2028\u2029]'
_SPLIT_LINE = re.compile(
    r'(?:\A|(?<=[\n\r\u2028\u2029]))\|split\|p([1234])\n(' + _NOT_EOL + r'*)\n(' + _NOT_EOL + r'*)'
    r'|' + _NOT_EOL + r'+'
)
_TIMESTAMP = re.compile(r'\|t:\|[0-9]+')
_TURN = re.compile(r'\|turn\|[0-9]+')


def extract_channel_messages(message, channel_ids=(0, 1, 2)):
    """Split one ``update`` payload into spectator (0) and private player (1..4) channels."""
    channels = {cid: [] for cid in (-1, 0, 1, 2, 3, 4)}
    for match in _SPLIT_LINE.finditer(message):
        line, player_match, secret, shared = match.group(0), match.group(1), match.group(2), match.group(3)
        player = int(player_match) if player_match else 0
        for cid in dict.fromkeys(channel_ids):
            chosen = line
            if player:
                chosen = secret if (cid == -1 or player == cid) else shared
                if not chosen:
                    continue
            channels[cid].append(chosen)
    return channels


def normalize_protocol(text):
    """Only protocol timestamp lines are normalized; requests and mechanics are unchanged."""
    return '\n'.join('|t:|<timestamp>' if _TIMESTAMP.fullmatch(line) else line for line in text.split('\n'))


class _Projection:
    """Per-player and spectator protocol lines rebuilt from the raw simulator output."""

    def __init__(self):
        self.players = {side: [] for side in SIDES}
        self.spectator = []

    def ingest(self, kind, data):
        if kind == 'update':
            channels = extract_channel_messages(data, (0, 1, 2))
            # join/split (not extend) is deliberate: an empty channel yields one empty line upstream.
            self.spectator.extend(normalize_protocol('\n'.join(channels[0])).split('\n'))
            for side, channel in (('p1', 1), ('p2', 2)):
                self._receive(side, '\n'.join(channels[channel]))
        elif kind == 'sideupdate':
            newline = data.find('\n')
            side = data[:newline]
            _check(side in self.players, f'Unexpected private channel {side!r}')
            self._receive(side, data[newline + 1:])
        elif kind != 'end':
            raise EpisodeValidationError(f'Unknown simulator message {kind!r}')
        # The end payload contains both private teams and never enters a player view.

    def _receive(self, side, text):
        self.players[side].extend(normalize_protocol(text).split('\n'))


# --- Request / phase helpers --------------------------------------------------------

def _history_index(lines, boundaries, side):
    """Parse each private protocol once; retain state only at captured boundaries."""
    wanted, states = set(boundaries), {}
    request, request_id, turn = None, 0, 0
    digest = sha256(b'[')
    for i in range(len(lines) + 1):
        if i in wanted:
            prefix = digest.copy()
            prefix.update(b']')
            states[i] = (request, request_id, turn, prefix.hexdigest())
        if i == len(lines):
            break
        line = lines[i]
        if i:
            digest.update(b',')
        digest.update(canonical(line).encode())
        if line.startswith('|request|'):
            request = parse_json(line[9:])
            request_id += 1
            _check(request.get('side', {}).get('id') == side, 'Foreign private request in history')
        if line.startswith('|turn|'):
            _check(_TURN.fullmatch(line), 'Malformed turn')
            turn = int(line[6:])
    return states


def phase_for(requests, terminal=False):
    """Cross-player phase label for a request group (diagnostics, never policy features)."""
    if terminal:
        return {'phase': 'terminal', 'side_phases': {'p1': 'terminal', 'p2': 'terminal'}, 'requested_sides': []}
    side_phases = {}
    for side in SIDES:
        request = requests[side]
        kind = request_phase(request)
        if kind != 'switch':
            side_phases[side] = kind
            continue
        _check(request['forceSwitch'][0], 'Unexpected singles pass-only switch group')
        mon = request['side']['pokemon'][0]
        side_phases[side] = ('revival' if mon.get('reviving') else
                             'faint_replacement' if mon['condition'].endswith(' fnt') else 'switch_continuation')
    requested_sides = [s for s in SIDES if side_phases[s] != 'wait']
    phases = [side_phases[s] for s in requested_sides]
    if len(phases) == 2 and all(p == 'move' for p in phases):
        phase = 'joint_move'
    elif phases and all(p == 'faint_replacement' for p in phases):
        phase = 'double_replacement' if len(phases) == 2 else 'single_replacement'
    elif phases and all(p == 'switch_continuation' for p in phases):
        phase = 'switch_continuation'
    elif phases and all(p == 'revival' for p in phases):
        phase = 'revival'
    elif phases and all(p != 'move' for p in phases):
        phase = 'mixed_switch'
    else:
        raise EpisodeValidationError('Unsupported or empty request group')
    return {'phase': phase, 'side_phases': side_phases, 'requested_sides': requested_sides}


def to_protocol(action, request):
    """Showdown choice string for a contract action chosen from ``request``."""
    _check(action in admissible_actions(request), 'Action is not admissible for its request')
    kind = action['kind']
    if kind == 'pass':
        return 'pass'
    command = f"{kind} {action['slot'] + 1}"
    if kind == 'move' and action['terastallize']:
        command += ' terastallize'
    return command


# --- Manifest -------------------------------------------------------------------------

def validate_manifest(manifest):
    _check(isinstance(manifest, dict), 'R4 manifest must be a JSON object')
    body = {k: v for k, v in manifest.items() if k != 'manifest_id'}
    _check(hash_json(body) == manifest.get('manifest_id') and body.get('version') == 'r4-manifest-1',
           'R4 manifest identity/version')
    _check(body.get('profile') in ('official', 'constructed'), 'R4 profile')
    _check(body.get('phase_version') == 'r4-phases-1' and body.get('audit_version') == 'r4-selected-mechanics-1',
           'R4 semantics version')
    _check('reference_contract' in body, 'R4 manifest lacks reference contract')
    validate_artifact(body['reference_contract'], kind='manifest')
    return manifest


# --- Views ----------------------------------------------------------------------------

def _validate_view(view, side, boundary, cursor, protocol, states):
    request, request_id, turn, prefix_hash = states[boundary['protocol_length']]
    _check(view['viewer'] == side and view['request']['side']['id'] == side
           and all(mon['ident'].startswith(f'{side}: ') for mon in view['request']['side']['pokemon']), 'View owner')
    _check(view['protocol_start'] == cursor and view['protocol_end'] == boundary['protocol_length']
           and view['protocol_end'] <= len(protocol), 'View delta boundary')
    _equal(view['protocol_delta'], protocol[cursor:boundary['protocol_length']], 'View delta differs')
    _check(view['protocol_prefix_hash'] == prefix_hash, 'View prefix hash')
    _equal(view['request'], request, 'View request differs from protocol')
    _check(view['request_id'] == request_id and view['turn'] == turn
           and view['phase'] == request_phase(view['request']), 'View request metadata')
    _equal(view['admissible_actions'], admissible_actions(view['request']), 'Visible action set')
    _equal(view['action_mask'], action_mask(view['request']), 'Visible action mask')
    body = {k: v for k, v in view.items() if k != 'view_id'}
    _check(view['view_id'] == hash_json(body), 'View identity')


# --- Episode --------------------------------------------------------------------------

_WINNERS = {'Player 1': 'p1', 'Player 2': 'p2', '': None}


def _validate_envelope(bundle, manifest):
    e, p = bundle['episode'], bundle['episode']['payload']
    _check(e['manifest_id'] == manifest['manifest_id'], 'Episode manifest mismatch')
    _check(manifest['profile'] == ('official' if p['fixture'] is None else 'constructed'), 'Fixture profile mismatch')
    body = {k: v for k, v in e.items() if k not in ('content_hash', 'artifact_id')}
    _check(e['content_hash'] == hash_json(body) and e['artifact_id'] == f"phase_episode:{e['content_hash']}",
           'Episode identity')
    _check(p['episode_key'] == hash_json({'spec': p['spec'], 'fixture': p['fixture'], 'controller': p['controller'],
                                           'submission_order': p['submission_order'], 'manifest_id': e['manifest_id']}),
           'Episode key')
    for i, entry in enumerate(p['spec']['teams']):
        _check(entry['team_hash'] == hash_json(entry['team']) and entry['team_hash'] == p['initial_team_hashes'][i],
               'Initial team identity')


def _validate_raw_evidence(bundle):
    """Raw output hashes, channel projection and the terminal/truncation outcome."""
    p = bundle['episode']['payload']
    _check(bundle['raw_output_hash'] == hash_json(bundle['raw_output']), 'Raw evidence hash')
    normalized = [{'type': row['type'], 'data': row['data'] if row['type'] == 'end' else normalize_protocol(row['data'])}
                  for row in bundle['raw_output']]
    _check(p['normalized_output_hash'] == hash_json(normalized), 'Normalized output hash')
    projection = _Projection()
    ends = [row for row in bundle['raw_output'] if row['type'] == 'end']
    for row in bundle['raw_output']:
        projection.ingest(row['type'], row['data'])
    for side in SIDES:
        _equal(p['protocol'][side], projection.players[side], 'Protocol/raw disagreement')
    _equal(p['spectator'], projection.spectator, 'Spectator/raw disagreement')
    terminal = p['outcome']['status'] == 'terminal'
    _check(len(ends) == int(terminal), 'Terminal/raw disagreement')
    if terminal:
        end = parse_json(ends[0]['data'])
        raw_winner = end.get('winner') if isinstance(end, dict) else None
        winner = _WINNERS.get(raw_winner, _MISSING) if isinstance(raw_winner, str) else _MISSING
        turns = end.get('turns') if isinstance(end, dict) else None
        _check(winner is not _MISSING and _same(p['outcome']['winner'], winner)
               and p['outcome']['reason'] == ('tie' if winner is None else 'win')
               and _same(turns, p['outcome']['turn']), 'Terminal outcome disagreement')
        _check(p['bootstrap_views'] is None, 'Terminal has bootstrap views')
    else:
        _check(p['outcome']['reason'] == 'decision_budget' and p['outcome']['winner'] is None
               and p['bootstrap_views'] is not None, 'Truncation outcome')
    return terminal


def _validate_input_tape_setup(p):
    spec = p['spec']
    # canonical() is the compact, deeply key-sorted JSON used by the reference writer.
    setup = [f">start {canonical({'formatid': spec['format_id'], 'seed': spec['mechanics_seed']})}"]
    for i, entry in enumerate(spec['teams']):
        setup.append(f">player p{i + 1} {canonical({'name': f'Player {i + 1}', 'team': entry['team']})}")
    _check(p['input_tape'][0] == '\n'.join(setup), 'Initial input tape differs')


def _validate_frames(p, terminal, histories):
    """Frame request boundaries, phase labels, audits, groups, commitments and rejections."""
    frames, decisions, protocol = p['frames'], p['decisions'], p['protocol']
    for i, frame in enumerate(frames):
        requests = {}
        for side in SIDES:
            boundary = frame['requests'][side]
            prior = frames[i - 1]['requests'][side] if i > 0 else None
            _check(boundary['protocol_length'] <= len(protocol[side])
                   and (prior is None or boundary['protocol_length'] >= prior['protocol_length']),
                   'Frame protocol boundary')
            request, request_id, turn, _ = histories[side][boundary['protocol_length']]
            _check(request_id == boundary['request_id'] and hash_json(request) == boundary['request_hash']
                   and turn == frame['turn'], 'Frame request boundary')
            requests[side] = request
        is_terminal = terminal and i == len(frames) - 1
        classified = phase_for(requests, is_terminal)
        for key in ('phase', 'side_phases', 'requested_sides'):
            _equal(frame[key], classified[key], 'Frame phase mismatch')
        audit = frame['audit']
        _check(audit['mechanics_hash'] == hash_json(audit['mechanics']) and audit['mechanics']['turn'] == frame['turn'],
               'Audit state/hash')
        _equal([s['id'] for s in audit['mechanics']['sides']], list(SIDES), 'Audit side order')
        _equal(frame['pending_sides'], [s for s in frame['requested_sides'] if frame['commitments'][s] is None],
               'Pending/commitment disagreement')
        if i == 0:
            _check(frame['group'] == 0 and frame['event'] == 'initial', 'Initial frame')
            _equal(frame['commitments'], {'p1': None, 'p2': None}, 'Initial commitments')
            continue
        before, d = frames[i - 1], decisions[i - 1]
        changed = [s for s in SIDES if frame['requests'][s]['request_id'] != before['requests'][s]['request_id']]
        next_group = bool(d['accepted'] and changed)
        event = ('terminal' if is_terminal else 'rerequest' if not d['accepted']
                 else 'resolved' if next_group else 'awaiting_choice')
        _check(frame['group'] == before['group'] + int(next_group) and frame['event'] == event, 'Group/event linkage')
        commitments = {'p1': None, 'p2': None} if is_terminal or next_group else dict(before['commitments'])
        if d['accepted'] and not is_terminal and not next_group:
            commitments[d['side']] = i - 1
        _equal(frame['commitments'], commitments, 'Commitment linkage')
        if not d['accepted']:
            other = 'p2' if d['side'] == 'p1' else 'p1'
            _equal(changed, [d['side']], 'Re-request side')
            _check(frame['turn'] == before['turn'] and audit['mechanics_hash'] == before['audit']['mechanics_hash']
                   and audit['choices'][other] == before['audit']['choices'][other],
                   'Rejection changed mechanics/opponent choice')


def _validate_decisions(p, histories):
    """Decision order/identity, per-player views, action encoding and engine feedback."""
    frames, decisions, protocol = p['frames'], p['decisions'], p['protocol']
    cursors = {'p1': 0, 'p2': 0}
    submission_ids = set()
    for i, d in enumerate(decisions):
        before, after = frames[i], frames[i + 1]
        _check(d['sequence'] == i and d['frame_before'] == i and d['frame_after'] == i + 1
               and d['side'] in before['pending_sides'], 'Decision boundary/order')
        _check(d['decision_id'] == hash_json(['r4-decision-1', p['episode_key'], {k: v for k, v in d.items() if k != 'decision_id'}]), 'Decision identity')
        _check(d['submission_id'] not in submission_ids, 'Repeated transport submission')
        submission_ids.add(d['submission_id'])
        _validate_view(d['view'], d['side'], before['requests'][d['side']], cursors[d['side']], protocol[d['side']], histories[d['side']])
        cursors[d['side']] = d['view']['protocol_end']
        _check(d['command'] == to_protocol(d['action'], d['view']['request'])
               and p['input_tape'][i + 1] == f">{d['side']} {d['command']}", 'Action/input mismatch')
        feedback = [{'side': s, 'message': line[7:]} for s in SIDES
                    for line in protocol[s][before['requests'][s]['protocol_length']:after['requests'][s]['protocol_length']]
                    if line.startswith('|error|')]
        _equal(d['feedback'], feedback, 'Feedback/protocol mismatch')
        _check(d['accepted'] == (len(d['feedback']) == 0)
               and all(f['side'] == d['side'] and f['message'].startswith('[Unavailable choice]') for f in d['feedback']),
               'Invalid choice feedback')
    return cursors


def _validate_final_boundary(p, terminal, cursors, histories):
    last = p['frames'][-1]
    _check(last['turn'] == p['outcome']['turn'], 'Final turn')
    for side in SIDES:
        _check(last['requests'][side]['protocol_length'] == len(p['protocol'][side]), 'Unaccounted final protocol')
    if not terminal:
        _check(last['phase'] == 'joint_move' and last['event'] in ('initial', 'resolved')
               and len(last['pending_sides']) == 2, 'Unsafe truncation boundary')
        for side in SIDES:
            _validate_view(p['bootstrap_views'][side], side, last['requests'][side], cursors[side], p['protocol'][side], histories[side])


def _validate_transitions(p, terminal):
    """Same-player next-decision links with terminal / truncation rewards (denominator 2)."""
    decisions, last_frame = p['decisions'], len(p['frames']) - 1
    next_decision = {'p1': None, 'p2': None}
    for i in range(len(decisions) - 1, -1, -1):
        d = decisions[i]
        following = next_decision[d['side']]
        end = following is None and terminal
        if following is not None:
            target = {'kind': 'decision', 'sequence': following, 'id': decisions[following]['decision_id']}
        elif terminal:
            target = {'kind': 'terminal', 'frame': last_frame}
        else:
            target = {'kind': 'bootstrap', 'frame': last_frame, 'id': p['bootstrap_views'][d['side']]['view_id']}
        winner = p['outcome']['winner']
        reward = (1 if winner is None else 2 if winner == d['side'] else 0) if end else 0
        _equal(p['transitions'][i], {
            'decision_id': d['decision_id'], 'side': d['side'], 'target': target, 'terminal': end,
            'truncated': following is None and not terminal, 'reward_numerator': reward, 'reward_denominator': 2,
        }, 'Same-player transition/reward linkage')
        if not d['accepted']:
            _check(following is not None, 'Rejected action lacks same-player re-request decision')
        next_decision[d['side']] = i


def _validate(bundle, manifest):
    canonical(bundle)  # Reject non-JSON values, floats and excessive nesting up front.
    _VALIDATOR.validate(bundle)
    validate_manifest(manifest)
    p = bundle['episode']['payload']
    _validate_envelope(bundle, manifest)
    terminal = _validate_raw_evidence(bundle)
    _check(len(p['frames']) == len(p['decisions']) + 1 and len(p['transitions']) == len(p['decisions']),
           'Decision/frame/transition count')
    _check(len(p['input_tape']) == len(p['decisions']) + 1 and len(p['decisions']) <= p['spec']['max_submissions'],
           'Input tape count/cap')
    _validate_input_tape_setup(p)
    histories = {side: _history_index(p['protocol'][side],
                 [f['requests'][side]['protocol_length'] for f in p['frames']], side) for side in SIDES}
    _validate_frames(p, terminal, histories)
    cursors = _validate_decisions(p, histories)
    _validate_final_boundary(p, terminal, cursors, histories)
    _validate_transitions(p, terminal)
    return bundle


def validate_episode(bundle, manifest):
    """Validate one R4 episode bundle against its manifest.

    Returns ``bundle`` unchanged on success; raises ``EpisodeValidationError`` otherwise.
    The inputs are never mutated.
    """
    try:
        return _validate(bundle, manifest)
    except EpisodeValidationError:
        raise
    except SchemaValidationError as exc:
        raise EpisodeValidationError(f'R4 schema: {exc.message}') from exc
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise EpisodeValidationError(f'R4 structure: {exc!r}') from exc


# --- CLI ------------------------------------------------------------------------------

def summarize(bundle):
    """Compact counts for a validated bundle."""
    p = bundle['episode']['payload']
    accepted = sum(1 for d in p['decisions'] if d['accepted'])
    return {
        'artifact_id': bundle['episode']['artifact_id'],
        'status': p['outcome']['status'], 'reason': p['outcome']['reason'], 'winner': p['outcome']['winner'],
        'turn': p['outcome']['turn'], 'frames': len(p['frames']), 'decisions': len(p['decisions']),
        'accepted': accepted, 'rejected': len(p['decisions']) - accepted, 'transitions': len(p['transitions']),
        'groups': p['frames'][-1]['group'] + 1, 'raw_output': len(bundle['raw_output']),
        'protocol_p1': len(p['protocol']['p1']), 'protocol_p2': len(p['protocol']['p2']),
        'spectator': len(p['spectator']), 'bootstrap_views': 0 if p['bootstrap_views'] is None else 2,
    }


def _load_json_file(path):
    data = Path(path).read_bytes()
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    return parse_json(data)


def _unwrap(document, key, marker):
    """Accept either a bare object or a golden-style wrapper such as {"bundle": ..., "manifest": ...}."""
    if isinstance(document, dict) and key in document and marker not in document:
        return document[key]
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate an R4 phase episode bundle without Node.')
    parser.add_argument('episode', help='Episode bundle JSON (optionally gzip-compressed)')
    parser.add_argument('--manifest', required=True, help='R4 manifest JSON (optionally gzip-compressed)')
    args = parser.parse_args(argv)
    try:
        bundle = _unwrap(_load_json_file(args.episode), 'bundle', 'episode')
        manifest = _unwrap(_load_json_file(args.manifest), 'manifest', 'manifest_id')
        validate_episode(bundle, manifest)
    except (EpisodeValidationError, ValueError, OSError) as exc:
        print(f'invalid: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(summarize(bundle), separators=(',', ':'), sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
