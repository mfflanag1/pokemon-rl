"""Constrain the pinned teacher's final root choice to its current legal request.

Native search still samples the same worlds and uses the original time budget.
Only its final root selection is replaced. A request with one legal action needs
no search or random draw. The authoritative mask is the teacher's own projection;
no opponent-private state crosses this boundary.
"""
from __future__ import annotations

from copy import deepcopy
import json
import logging
import random
import sys
import time

from pokemon_external_benchmark.foul_play import FoulPlayAdapter, decision_to_action, validate_input
from pokemon_teacher_assisted.teacher import AcceptedTeacher as OriginalAcceptedTeacher
from . import TEACHER_POLICY
from .selection import select_legal_from_mcts_results


def validate_legality(legal_mask, request_id):
    if (not isinstance(legal_mask, list) or len(legal_mask) != 14
            or any(type(value) is not bool for value in legal_mask)):
        raise ValueError('Authoritative legal mask must contain exactly 14 booleans')
    if type(request_id) is not int or request_id < 0:
        raise ValueError('Authoritative request ID must be a nonnegative integer')
    return tuple(legal_mask)


def singleton_decision(battle, action):
    """Invert one authoritative action through the current own request slots."""
    if action < 8:
        slot = action % 4
        moves = battle.request_json['active'][0]['moves']
        if slot >= len(moves) or not isinstance(moves[slot].get('id'), str):
            raise ValueError('Forced move is absent from current request')
        return moves[slot]['id'] + ('-tera' if action >= 4 else '')
    matches = [pokemon for pokemon in battle.user.reserve if pokemon.index == action - 8 + 1]
    if len(matches) != 1:
        raise ValueError('Forced switch slot is absent or ambiguous in current request')
    return 'switch ' + matches[0].name


class LegalFoulPlayAdapter(FoulPlayAdapter):
    def set_request_legality(self, legal_mask, request_id):
        self.legal_mask = validate_legality(legal_mask, request_id)
        self.legal_request_id = request_id

    def choose(self):
        if not self.pending or not self.battle.started or self.battle.wait:
            raise ValueError('Foul Play cannot choose while waiting')
        if self.last_decided_request == self.request_count:
            raise ValueError('Duplicate Foul Play decision for request')
        if (self.battle.rqid != self.request_count or not isinstance(self.battle.request_json, dict)
                or self.battle.request_json.get('rqid') != self.request_count
                or getattr(self, 'legal_request_id', None) != self.request_count):
            raise ValueError('Foul Play request state or authoritative legality is stale')
        legal = [i for i, valid in enumerate(self.legal_mask) if valid]
        if not legal:
            raise ValueError('Pending teacher has no authoritative legal action')

        from fp.battle.state import LastUsedMove
        from fp.modes.base import format_decision
        import fp.search.main as native_search
        battle_copy = deepcopy(self.battle)
        battle_copy.user.update_from_request_json(battle_copy.request_json)
        started = time.perf_counter()
        samples = per_sample_ms = 0
        audit = None
        selector_calls = 0

        def action_for_choice(choice):
            if not choice.startswith('switch '):
                # The pinned formatter dereferences active.get_move(...).can_z.
                # Reject absent/ambiguous request moves before that dereference;
                # do not turn arbitrary formatter failures into illegal choices.
                move = choice.replace('-tera', '') if choice.endswith('-tera') else (
                    choice.replace('-mega', '') if choice.endswith('-mega') else choice)
                decision_to_action(['/choose move ' + move], battle_copy.request_json)
            return decision_to_action(format_decision(battle_copy, choice), battle_copy.request_json)

        if len(legal) == 1:
            kind, search_calls = 'forced-request', 0
            decision = singleton_decision(battle_copy, legal[0])
            audit = {'schema_version': 'f37-forced-request-v1', 'action': legal[0],
                     'legal_action_count': 1, 'native_search_invoked': False,
                     'selection_rng_draws': 0, 'reason': 'Exactly one authoritative legal action.'}
        else:
            kind, search_calls = 'native-legal-root', 1
            samples, per_sample_ms = battle_copy.mode.search_params(battle_copy)
            original_selector = native_search.select_move_from_mcts_results

            def select(results):
                nonlocal audit, selector_calls
                selector_calls += 1
                if selector_calls != 1:
                    raise ValueError('Native teacher invoked root selector more than once')
                selected = select_legal_from_mcts_results(
                    results, list(self.legal_mask), action_for_choice, rng=random)
                audit = selected['audit']
                return selected['decision']

            native_search.select_move_from_mcts_results = select
            try:
                decision = native_search.find_best_move(battle_copy)
            finally:
                native_search.select_move_from_mcts_results = original_selector
            if selector_calls != 1 or audit is None:
                raise ValueError('Native teacher did not use the authoritative root selector')

        wire = format_decision(battle_copy, decision)
        action = decision_to_action(wire, battle_copy.request_json)
        if type(action) is not int or not 0 <= action < 14 or not self.legal_mask[action]:
            error = ValueError('Selected teacher action violates authoritative legal mask')
            error.audit = audit
            raise error
        # Match upstream bookkeeping only after the current request/mask guard.
        # AcceptedTeacher restores this value if Showdown rejects submission.
        self.battle.user.last_selected_move = LastUsedMove(self.battle.user.active.name,
            decision.removesuffix('-tera').removesuffix('-mega'), self.battle.turn)
        self.last_decided_request = self.request_count
        return {'action': action, 'decision': decision, 'command': wire[0],
            'request_id': self.request_count, 'seconds': time.perf_counter() - started,
            'search_time_ms': self.search_time_ms, 'sampled_worlds': samples,
            'per_world_ms': per_sample_ms, 'configured_search_total_ms': samples * per_sample_ms,
            'opponent_revealed': len(self.battle.opponent.reserve) + int(self.battle.opponent.active is not None),
            'teacher_policy': TEACHER_POLICY, 'selection': {'kind': kind,
                'legal_mask': list(self.legal_mask), 'request_id': self.request_count,
                'search_calls': search_calls, 'audit': audit}}


class AcceptedTeacher(OriginalAcceptedTeacher):
    """Retain F35 accepted/rejected bookkeeping with an own-request legality input."""
    def __init__(self, side, search_time_ms=100, seed=0, *, adapter=None):
        super().__init__(side, search_time_ms, seed,
            adapter=adapter if adapter is not None else LegalFoulPlayAdapter(side, search_time_ms, seed))
        self.side = side

    def choose(self, envelope, legal_mask, request_id):
        # Check before ingesting or changing request context, including waits.
        if self.pending is not None:
            raise ValueError('Teacher decision requires submission feedback before next observation')
        validate_input(envelope, self.side)
        validate_legality(legal_mask, request_id)
        if request_id != envelope['request_id'] or bool(any(legal_mask)) != envelope['pending']:
            raise ValueError('Authoritative mask/request differs from teacher envelope')
        self.adapter.set_request_legality(legal_mask, request_id)
        return super().choose(envelope)


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    teacher = None
    fields = {'start': {'op', 'id', 'side', 'search_time_ms', 'seed'},
              'choose': {'op', 'id', 'input', 'legal_mask', 'request_id'},
              'feedback': {'op', 'id', 'request_id', 'accepted'}, 'close': {'op', 'id'}}
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
                data = {'started': True, 'teacher_policy': TEACHER_POLICY}
            elif op == 'choose' and teacher:
                data = teacher.choose(command['input'], command['legal_mask'], command['request_id'])
            elif op == 'feedback' and teacher:
                data = teacher.feedback(command['request_id'], command['accepted'])
            elif op == 'close':
                teacher = None
                data = {'closed': True}
            else:
                raise ValueError('Missing teacher or unsupported operation')
            print(json.dumps({'id': command['id'], 'ok': True, 'data': data}, allow_nan=False), flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            reply = {'id': command.get('id') if isinstance(command, dict) else None,
                     'ok': False, 'error': str(exc), 'error_type': type(exc).__name__,
                     'teacher_policy': TEACHER_POLICY}
            if getattr(exc, 'audit', None) is not None:
                reply['selection_audit'] = exc.audit
            print(json.dumps(reply, allow_nan=False), flush=True)
            teacher = None


if __name__ == '__main__':
    main()
