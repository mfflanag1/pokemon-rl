"""Independent Python implementation of battle contract 1.0.0.

No simulator state or Node subprocess is needed. Protocol command encoding remains
on the Node side. This module does not build model features from battle history.
"""

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

CONTRACT = Path(__file__).resolve().parents[2] / 'contracts' / 'battle-v1'
MAX_INTEGER = 9007199254740991


def _string(value):
    value.encode('utf-8', errors='strict')  # Reject lone surrogates.
    return json.dumps(value, ensure_ascii=False)


def canonical(value, depth=0):
    if value is None:
        return 'null'
    if type(value) is bool:
        return 'true' if value else 'false'
    if type(value) is str:
        return _string(value)
    if type(value) is int:
        if abs(value) > MAX_INTEGER:
            raise ValueError('Expected a safe integer')
        return str(value)
    if depth >= 100:
        raise ValueError('JSON nesting exceeds 100 containers')
    if type(value) is list:
        return '[' + ','.join(canonical(v, depth + 1) for v in value) + ']'
    if type(value) is dict and all(type(k) is str for k in value):
        return '{' + ','.join(_string(k) + ':' + canonical(value[k], depth + 1) for k in sorted(value)) + '}'
    raise TypeError('Non-JSON value (only safe integer numbers are supported)')


def hash_json(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def parse_json(text):
    def reject_number(_):
        raise ValueError('Only integer JSON literals are allowed')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate object key')
            result[key] = value
        return result

    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='strict')
    value = json.loads(text, parse_float=reject_number, parse_constant=reject_number, object_pairs_hook=pairs)
    canonical(value)
    return value


SCHEMA = parse_json((CONTRACT / 'schema.json').read_bytes())
Draft202012Validator.check_schema(SCHEMA)
ACTIONS = parse_json((CONTRACT / 'actions.json').read_bytes())['actions']
ACTION_COUNT = len(ACTIONS)
_VALIDATORS = {None: Draft202012Validator(SCHEMA)}


def validate_schema(value, fragment=None):
    canonical(value)
    if fragment not in _VALIDATORS:
        if fragment not in SCHEMA['$defs']:
            raise ValueError('Unknown schema fragment')
        _VALIDATORS[fragment] = Draft202012Validator({
            '$schema': SCHEMA['$schema'], '$defs': SCHEMA['$defs'], '$ref': f'#/$defs/{fragment}',
        })
    _VALIDATORS[fragment].validate(value)
    return value


def action_index(action):
    validate_schema(action, 'action')
    try:
        return ACTIONS.index(action)
    except ValueError as exc:
        raise ValueError('Control actions have no policy index') from exc


def action_from_index(index):
    if type(index) is not int or not 0 <= index < ACTION_COUNT:
        raise ValueError('Invalid action index')
    return deepcopy(ACTIONS[index])


def request_phase(request):
    validate_schema(request, 'request')
    return 'wait' if request.get('wait') else 'switch' if 'forceSwitch' in request else 'move'


def admissible_actions(request):
    phase = request_phase(request)
    if phase == 'wait':
        return []
    pokemon = request['side']['pokemon']
    fainted = lambda mon: mon['condition'].endswith(' fnt')
    if phase == 'switch':
        if not request['forceSwitch'][0]:
            return [{'kind': 'pass'}]
        reviving = pokemon[0].get('reviving', False)
        choices = [{'kind': 'switch', 'slot': slot} for slot, mon in enumerate(pokemon)
                   if (slot > 0 or reviving) and (fainted(mon) if reviving else not fainted(mon))]
        return choices or [{'kind': 'pass'}]
    active = request['active'][0]
    if all(move.get('disabled') for move in active['moves']):
        raise ValueError('No advertised move; expected a Struggle request')
    actions = []
    for slot, move in enumerate(active['moves']):
        if move.get('disabled'):
            continue
        actions.append({'kind': 'move', 'slot': slot, 'terastallize': False})
        if active.get('canTerastallize'):
            actions.append({'kind': 'move', 'slot': slot, 'terastallize': True})
    if not active.get('trapped'):
        actions.extend({'kind': 'switch', 'slot': slot} for slot, mon in enumerate(pokemon)
                       if not mon['active'] and not fainted(mon))
    return actions


def action_mask(request):
    mask = [False] * ACTION_COUNT
    for action in admissible_actions(request):
        if action['kind'] != 'pass':
            mask[action_index(action)] = True
    return mask


def validate_artifact(artifact, *, kind=None, manifest_id=None):
    validate_schema(artifact)
    artifact_kind = artifact['artifact_kind']
    payload = artifact['payload']
    if kind is not None and artifact_kind != kind:
        raise ValueError(f'Expected {kind} artifact')
    if manifest_id is not None and artifact['manifest_id'] != manifest_id:
        raise ValueError('Manifest mismatch')
    body = {k: artifact[k] for k in ('schema_version', 'artifact_kind', 'manifest_id', 'payload')}
    if hash_json(body) != artifact['content_hash']:
        raise ValueError('Content hash mismatch')
    if artifact['artifact_id'] != artifact_kind + ':' + artifact['content_hash']:
        raise ValueError('Artifact ID mismatch')
    if artifact_kind == 'manifest' and artifact['manifest_id'] != hash_json(payload):
        raise ValueError('Manifest identity mismatch')
    if artifact_kind == 'manifest':
        source = {k: v for k, v in payload['reference_manifest'].items() if k != 'manifest_id'}
        if (hash_json(source) != payload['reference_manifest']['manifest_id']
                or source['commit'] != source['build']['commit']
                or source['lock_sha256'] != source['build']['lock_sha256']):
            raise ValueError('Reference manifest identity mismatch')
    if artifact_kind == 'team' and payload['source_team_hash'] != hash_json(payload['pokemon']):
        raise ValueError('Source team hash mismatch')
    if artifact_kind == 'player_view':
        request = payload['request']
        viewer = payload['viewer']
        if request['side']['id'] != viewer or any(not mon['ident'].startswith(viewer + ': ') for mon in request['side']['pokemon']):
            raise ValueError('Private request belongs to another viewer')
        if (payload['phase'] != request_phase(request)
                or payload['admissible_actions'] != admissible_actions(request)
                or payload['action_mask'] != action_mask(request)):
            raise ValueError('Request/action mask mismatch')
        last_request, request_count, turn = None, 0, 0
        for line in payload['visible_protocol']:
            if line.startswith('|request|'):
                last_request = parse_json(line[9:])
                request_count += 1
                if not isinstance(last_request, dict) or last_request.get('side', {}).get('id') != viewer:
                    raise ValueError('Foreign private request in player history')
            if line.startswith('|turn|'):
                if not re.fullmatch(r'\|turn\|[0-9]+', line):
                    raise ValueError('Malformed turn record')
                turn = int(line[6:])
        if request_count != payload['request_id'] or turn != payload['turn'] or last_request != request:
            raise ValueError('Player history/request boundary mismatch')
    if artifact_kind == 'decision':
        if payload['accepted'] != (len(payload['feedback']) == 0) or any(e['side'] != payload['side'] for e in payload['feedback']):
            raise ValueError('Inconsistent decision feedback')
    if artifact_kind == 'episode' and ((payload['outcome']['winner'] is None) != (payload['outcome']['reason'] == 'tie')):
        raise ValueError('Inconsistent terminal reason')
    return artifact


def make_artifact(kind, manifest_id, payload):
    if kind == 'manifest':
        manifest_id = hash_json(payload)
    body = {'schema_version': '1.0.0', 'artifact_kind': kind, 'manifest_id': manifest_id, 'payload': payload}
    digest = hash_json(body)
    return validate_artifact({**body, 'content_hash': digest, 'artifact_id': kind + ':' + digest})


def load_artifact(text, **kwargs):
    return validate_artifact(parse_json(text), **kwargs)


def load_player_view(text, manifest_id=None):
    return load_artifact(text, kind='player_view', manifest_id=manifest_id)['payload']


def vocabulary_index(vocabulary, domain, knowledge):
    validate_schema(knowledge, 'known_id')
    entries = vocabulary['domains'].get(domain)
    if not entries or entries[:2] != ['<unknown>', '<none>']:
        raise ValueError('Invalid vocabulary domain')
    if knowledge['state'] == 'unknown':
        return 0
    if knowledge['value'] == '':
        return 1
    try:
        return entries.index(knowledge['value'], 2)
    except ValueError as exc:
        raise ValueError('Unknown known ID: ' + knowledge['value']) from exc
