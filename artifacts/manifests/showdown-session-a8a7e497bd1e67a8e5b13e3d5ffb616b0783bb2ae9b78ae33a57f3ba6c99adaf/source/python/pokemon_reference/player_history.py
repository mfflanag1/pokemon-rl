"""Typed player-only history -> fixed actor inputs. Never accepts full episode data."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import re
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from .contract import (SCHEMA, action_mask, admissible_actions, canonical, hash_json,
                       parse_json, request_phase, validate_schema)
from .episodes import EPISODE_SCHEMA
from .feature_spec import CATALOG, DIRECTORY, LAYOUT, LAYOUT_HASH, category
from .actor_actions import ActionCodec
from .public_history import PublicHistory, details, health
from .feature_validation import validate_tensors

_REGISTRY = Registry().with_resources([(s['$id'], Resource.from_contents(s, default_specification=DRAFT202012))
                                      for s in (SCHEMA, EPISODE_SCHEMA)])
_VIEW = Draft202012Validator({'$ref': EPISODE_SCHEMA['$id'] + '#/$defs/view'}, registry=_REGISTRY)
_ACTOR_SCHEMA = parse_json((DIRECTORY / 'schema.json').read_bytes())
_ACTOR = Draft202012Validator(_ACTOR_SCHEMA)


@dataclass(frozen=True)
class ActorInput:
    """Immutable serialized actor features; provenance and targets are excluded."""
    _wire: str

    @property
    def feature_id(self):
        return parse_json(self._wire)['feature_id']

    def to_dict(self):
        return parse_json(self._wire)

    def model_inputs(self):
        return parse_json(self._wire)['tensors']

    def policy_inputs(self):
        return self.model_inputs()

    def value_inputs(self):
        return self.model_inputs()


def load_actor_input(wire):
    body = parse_json(wire)
    _ACTOR.validate(body)
    validate_tensors(body['tensors'])
    if (body['layout_hash'] != LAYOUT_HASH or body['catalog_hash'] != CATALOG['content_hash']
            or body['feature_id'] != hash_json({k: v for k, v in body.items() if k != 'feature_id'})):
        raise ValueError('Actor feature identity/definition mismatch')
    return ActorInput(canonical(body))


def _bytes_rows(lines, rows, width):
    encoded, lengths, overflow = [], [], []
    for text in lines[-rows:]:
        raw = text.encode('utf-8')
        encoded.append([v + 1 for v in raw[:width]] + [0] * max(0, width - len(raw)))
        lengths.append(min(len(raw), width)); overflow.append(int(len(raw) > width))
    pad = rows - len(encoded)
    return encoded + [[0] * width for _ in range(pad)], lengths + [0] * pad, overflow + [0] * pad


def _appearance(row):
    return [row.get(k, 0) for k in LAYOUT['appearance_columns']] if row else [0] * len(LAYOUT['appearance_columns'])


class PlayerHistory:
    """Append verified private deltas transactionally. One instance per player/game."""
    def __init__(self, viewer):
        if viewer not in ('p1', 'p2'):
            raise ValueError('Unknown player')
        self.viewer = viewer
        self._protocol = []
        self._prefix = sha256(b'[')
        self._request = None
        self._request_count = 0
        self._turn = 0
        self._public = PublicHistory(viewer)
        self._own_max = {}
        self._last_wire = None
        self._actor = None

    def append(self, view):
        """Accept only an episode-v1 view. A rejected append leaves all state unchanged."""
        _VIEW.validate(view)
        wire = canonical(view)
        if wire == self._last_wire:
            return self._actor
        if view['viewer'] != self.viewer or view['protocol_start'] != len(self._protocol):
            raise ValueError('Foreign, stale or noncontiguous player delta')
        if view['protocol_end'] != view['protocol_start'] + len(view['protocol_delta']):
            raise ValueError('Delta length mismatch')
        if view['view_id'] != hash_json({k: v for k, v in view.items() if k != 'view_id'}):
            raise ValueError('View identity mismatch')
        prefix = self._prefix.copy()
        request, count, turn = self._request, self._request_count, self._turn
        own_max = dict(self._own_max)
        for i, line in enumerate(view['protocol_delta'], start=len(self._protocol)):
            if i: prefix.update(b',')
            prefix.update(canonical(line).encode())
            if line.startswith('|request|'):
                request = parse_json(line[9:]); count += 1
                validate_schema(request, 'request')
                if (request['side']['id'] != self.viewer
                        or any(not p['ident'].startswith(self.viewer + ': ') for p in request['side']['pokemon'])):
                    raise ValueError('Foreign private request in player history')
                for mon in request['side']['pokemon']:
                    _, maximum, _, known, _, _ = health(mon['condition'], previous_max=own_max.get(mon['ident'], 0))
                    if known: own_max[mon['ident']] = maximum
            if line.startswith('|turn|'):
                if not re.fullmatch(r'\|turn\|[0-9]+', line):
                    raise ValueError('Malformed turn')
                turn = int(line[6:])
        finished = prefix.copy(); finished.update(b']')
        if finished.hexdigest() != view['protocol_prefix_hash']:
            raise ValueError('Private history prefix checksum')
        if (canonical(request) != canonical(view['request']) or count != view['request_id'] or turn != view['turn']
                or view['phase'] != request_phase(request)):
            raise ValueError('Request/history boundary mismatch')
        if (canonical(view['admissible_actions']) != canonical(admissible_actions(request))
                or view['action_mask'] != action_mask(request)):
            raise ValueError('Observable action set mismatch')
        public = deepcopy(self._public)
        for line in view['protocol_delta']:
            if line: public.ingest(line)
        actor = self._encode(request, count, turn, own_max, public)
        # Commit only after every wire, semantic and feature check succeeds.
        self._protocol.extend(view['protocol_delta']); self._prefix = prefix
        self._request = deepcopy(request); self._request_count = count; self._turn = turn
        self._own_max = own_max; self._public = public; self._actor = actor; self._last_wire = wire
        return actor

    @property
    def request(self):
        if self._request is None:
            raise ValueError('No verified request yet')
        return deepcopy(self._request)

    def snapshot(self):
        """Player-visible debug state; callers cannot mutate the live memory."""
        return {'viewer': self.viewer, 'request': self.request, 'protocol': list(self._protocol),
                'opponent_appearances': self._public.appearances('p2' if self.viewer == 'p1' else 'p1'),
                'own_public_appearances': self._public.appearances(self.viewer),
                'public_facts': dict(self._public.facts)}

    def _encode(self, request, count, turn, own_max, public):
        codec = ActionCodec(request)
        active = request.get('active', [{}])[0]
        own = []
        for mon in request['side']['pokemon']:
            species, level, gender = details(mon['details'])
            hp, maximum, hp_known, max_known, fainted, status = health(mon['condition'], previous_max=own_max.get(mon['ident'], 0))
            moves = [category('moves', m) for m in mon['moves']]
            own.append([1, species, level, gender, hp, maximum, hp_known, max_known, fainted, status,
                        int(mon['active']), category('abilities', mon['baseAbility']), category('abilities', mon.get('ability')),
                        category('items', mon['item']), category('types', mon.get('teraType')),
                        category('types', mon.get('terastallized')), int(mon.get('reviving', False)),
                        *[mon['stats'][s] for s in ('atk', 'def', 'spa', 'spd', 'spe')], *moves, *([0] * (4 - len(moves)))])
        advertised = [[1, category('moves', m['id']), m.get('pp', 0), m.get('maxpp', 0), int(m.get('disabled', False)), int('disabled' in m), int('pp' in m), int('maxpp' in m)]
                      for m in active.get('moves', [])]
        other = 'p2' if self.viewer == 'p1' else 'p1'
        encounters = public.encounters[other]
        appearances = [_appearance(r) for r in reversed(encounters[-6:])]
        facts = [key + '|' + value for key, value in sorted(public.facts.items())]
        events, event_lengths, event_overflow = _bytes_rows(public.events, 64, 192)
        fact_bytes, fact_lengths, fact_overflow = _bytes_rows(facts, 32, 128)
        tensors = {
            'global': [turn, int(codec.phase == 'move'), int(codec.phase == 'switch'), int(codec.phase == 'wait'),
                       int(codec.control == 'pass'), count, int(active.get('trapped', False)), int('trapped' in active),
                       int(active.get('maybeTrapped', False)), int('maybeTrapped' in active), int(bool(active.get('canTerastallize'))),
                       len(encounters), max(0, len(encounters) - 6), len(public.events), max(0, len(public.events) - 64),
                       public.unstructured, max(0, len(facts) - 32)],
            'own': own, 'own_public_active': _appearance(public.current(self.viewer)),
            'opponent_appearances': appearances + [[0] * len(LAYOUT['appearance_columns']) for _ in range(6 - len(appearances))],
            'advertised_moves': advertised + [[0] * len(LAYOUT['advertised_move_columns']) for _ in range(4 - len(advertised))],
            'actions': deepcopy(codec.bindings), 'action_mask': list(codec.mask),
            'event_bytes': events, 'event_lengths': event_lengths, 'event_overflow': event_overflow,
            'fact_bytes': fact_bytes, 'fact_lengths': fact_lengths, 'fact_overflow': fact_overflow,
        }
        body = {'schema_version': '1.0.0', 'artifact_kind': 'actor_input', 'feature_version': LAYOUT['version'],
                'layout_hash': LAYOUT_HASH, 'catalog_hash': CATALOG['content_hash'], 'tensors': tensors}
        return ActorInput(canonical({**body, 'feature_id': hash_json(body)}))
