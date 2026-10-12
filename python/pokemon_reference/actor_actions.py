"""Request-relative policy indices; wait/pass never enter the categorical head."""
from .contract import (ACTION_COUNT, action_from_index, action_index, admissible_actions,
                       canonical, request_phase, validate_schema)
from .feature_spec import category


class ActionCodec:
    def __init__(self, request):
        validate_schema(request, 'request')
        self.phase = request_phase(request)
        legal = admissible_actions(request)
        self.control = 'wait' if self.phase == 'wait' else 'pass' if legal == [{'kind': 'pass'}] else 'choose'
        self._legal = {canonical(a) for a in legal}
        self.mask = tuple(canonical(action_from_index(i)) in self._legal for i in range(ACTION_COUNT))
        self.bindings = []
        mons = request['side']['pokemon']
        moves = request.get('active', [{}])[0].get('moves', [])
        for i in range(ACTION_COUNT):
            a = action_from_index(i); slot = a['slot']; row = [0, slot, 0, 0, int(self.mask[i])]
            if self.control == 'choose':
                if a['kind'] == 'switch' and slot < len(mons):
                    row = [3, slot, category('species', mons[slot]['details'].split(',')[0]), 0, int(self.mask[i])]
                elif a['kind'] == 'move' and slot < len(moves):
                    row = [2 if a['terastallize'] else 1, slot, category('moves', moves[slot]['id']), int(a['terastallize']), int(self.mask[i])]
            self.bindings.append(row)

    def encode(self, action):
        validate_schema(action, 'action')
        if canonical(action) not in self._legal:
            raise ValueError('Action forbidden by visible request')
        if action['kind'] == 'pass':
            raise ValueError('Pass is a control action, not a policy index')
        return action_index(action)

    def decode(self, index):
        action = action_from_index(index)
        if not self.mask[index]:
            raise ValueError('Policy index forbidden by visible request')
        return action
