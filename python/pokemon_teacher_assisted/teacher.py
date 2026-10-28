"""Acceptance-aware wrapper around the unchanged, pinned offline Foul Play adapter."""
from copy import deepcopy
import json
import logging
import sys

from pokemon_external_benchmark.foul_play import FoulPlayAdapter


class AcceptedTeacher:
    """Commit a selected-move bookkeeping update only after Showdown accepts it."""
    def __init__(self, side, search_time_ms=100, seed=0, *, adapter=None):
        self.adapter = adapter if adapter is not None else FoulPlayAdapter(side, search_time_ms, seed)
        self.pending = None

    def choose(self, envelope):
        if self.pending is not None:
            raise ValueError('Teacher decision requires submission feedback before next observation')
        self.adapter.ingest(envelope)
        if not self.adapter.pending:
            return {'waiting': True}
        previous = deepcopy(self.adapter.battle.user.last_selected_move)
        decision = self.adapter.choose()
        self.pending = {'request_id': decision['request_id'], 'previous': previous}
        return decision

    def feedback(self, request_id, accepted):
        if type(request_id) is not int or type(accepted) is not bool:
            raise ValueError('Teacher feedback coordinates')
        if self.pending is None or request_id != self.pending['request_id']:
            raise ValueError('Missing, duplicate or stale teacher feedback')
        if not accepted:
            self.adapter.battle.user.last_selected_move = self.pending['previous']
        self.pending = None
        return {'acknowledged': True, 'request_id': request_id, 'accepted': accepted}


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    teacher = None
    fields = {'start': {'op', 'id', 'side', 'search_time_ms', 'seed'},
              'choose': {'op', 'id', 'input'}, 'feedback': {'op', 'id', 'request_id', 'accepted'},
              'close': {'op', 'id'}}
    for raw in sys.stdin:
        command = None
        try:
            command = json.loads(raw)
            if not isinstance(command, dict) or set(command) != fields.get(command.get('op')):
                raise ValueError('Teacher command inventory')
            op = command['op']
            if op == 'start':
                if teacher is not None:
                    raise ValueError('Teacher already started')
                teacher = AcceptedTeacher(command['side'], command['search_time_ms'], command['seed'])
                data = {'started': True}
            elif op == 'choose' and teacher:
                data = teacher.choose(command['input'])
            elif op == 'feedback' and teacher:
                data = teacher.feedback(command['request_id'], command['accepted'])
            elif op == 'close':
                teacher = None
                data = {'closed': True}
            else:
                raise ValueError('Missing teacher or unsupported operation')
            print(json.dumps({'id': command['id'], 'ok': True, 'data': data}), flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'id': command.get('id') if isinstance(command, dict) else None,
                              'ok': False, 'error': str(exc), 'error_type': type(exc).__name__}), flush=True)
            teacher = None


if __name__ == '__main__':
    main()
