"""Player-only structured input for switch-reference-1, not a JAX tensor layout.

R5's public event reducer is reusable, but its six-member request/schema encoder
is not. This version validates the new four-action request and retains full public
memory plus the latest own private roster, including while waiting.
"""
from dataclasses import dataclass
from copy import deepcopy
import json
from pokemon_reference.contract import canonical, hash_json, parse_json
from pokemon_reference.public_history import PublicHistory
from .contract import mask


@dataclass(frozen=True)
class ActorInput:
    _wire: str

    def policy_inputs(self): return json.loads(self._wire)
    def value_inputs(self): return json.loads(self._wire)
    @property
    def feature_id(self): return hash_json(json.loads(self._wire))


def actor(view):
    if not isinstance(view,dict) or set(view)!={'side','request_id','turn','request','protocol'}:
        raise ValueError('Expected a player-only R1 view, without audit/seed/opponent state')
    if view['side'] not in ('p1','p2') or type(view['turn']) is not int or type(view['request_id']) is not int:
        raise ValueError('Player identity/counters')
    if not isinstance(view['protocol'],list) or not all(isinstance(line,str) for line in view['protocol']):
        raise ValueError('Player protocol')
    public=PublicHistory(view['side']);request=None;own=None;count=turn=0
    for line in view['protocol']:
        if line.startswith('|request|'):
            request=parse_json(line[9:]);count+=1;mask(request)
            if 'side' in request:
                if request['side']['id']!=view['side']:
                    raise ValueError('Foreign private request')
                own=deepcopy(request['side'])
        if line.startswith('|turn|'):
            token=line[6:]
            if not token.isascii() or not token.isdecimal():raise ValueError('Malformed turn')
            turn=int(token)
        if line:public.ingest(line)
    if (request!=view['request'] or count!=view['request_id'] or turn!=view['turn'] or own is None):
        raise ValueError('Request/history boundary mismatch')
    other='p2' if view['side']=='p1' else 'p1'
    body={'version':'switch-player-input-1','turn':turn,
          'phase':'wait' if request.get('wait') else 'replacement' if request.get('forceSwitch') else 'move',
          'own':own,'request':request,'action_mask':mask(request),
          'own_public_appearances':public.appearances(view['side']),
          'opponent_appearances':public.appearances(other),
          'public_events':public.events,'public_facts':public.facts}
    return ActorInput(canonical(body))
