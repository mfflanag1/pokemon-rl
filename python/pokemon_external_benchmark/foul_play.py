"""Pinned Foul Play adapter accepting one legitimate player protocol only.

No server client is instantiated. Public set priors are loaded from pinned bytes;
network connects are denied for this process and spawned search workers. Upstream
Foul Play source remains unmodified. Python random is seeded, but Rust MCTS uses
unseeded randomness and a wall-time budget, so exact replay is not promised.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import random
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/external/foul-play'
PIN = ROOT / 'packages/external-benchmark/pin.json'


def deny_network(event, args):
    if event == 'socket.connect' and args[0].family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('F34 external opponent is offline: network connection denied')


sys.addaudithook(deny_network)
sys.path.insert(0, str(SOURCE))


def verify_runtime():
    import importlib.metadata
    pin = json.loads(PIN.read_text())
    for rel, expected in pin['foul_play']['files'].items():
        if hashlib.sha256((SOURCE / rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Foul Play pinned source drift: ' + rel)
    if importlib.metadata.version('poke-engine') != pin['poke_engine']['version']:
        raise ValueError('Unexpected poke-engine version')
    for rel, expected in pin['poke_engine']['runtime_files'].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError('Poke-engine runtime drift: ' + rel)
    dataset = ROOT / pin['set_prior']['path']
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != pin['set_prior']['sha256']:
        raise ValueError('Public set prior hash mismatch')
    return pin, dataset


def validate_input(value, side):
    expected = {'schema_version','side','request_id','pending','protocol'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('External input inventory')
    if value['schema_version'] != 'f34-external-player-v1' or value['side'] != side:
        raise ValueError('External input version/side')
    if type(value['request_id']) is not int or value['request_id'] < 0 or type(value['pending']) is not bool:
        raise ValueError('External request coordinates')
    if not isinstance(value['protocol'], list) or any(not isinstance(x,str) for x in value['protocol']):
        raise ValueError('External protocol lines')
    for line in value['protocol']:
        if '\n' in line or '\r' in line or not (line == '' or line.startswith('|')):
            raise ValueError('Invalid player protocol line')
        if line.startswith('|split|'):
            raise ValueError('Unsplit protocol forbidden')
        if line.startswith('|request|'):
            request = json.loads(line[9:])
            if request is not None and request.get('side',{}).get('id',side) != side:
                raise ValueError('Foreign private request')


def decision_to_action(decision, request):
    """Translate upstream command via the current authoritative own request."""
    command = decision[0]
    if command.startswith('/switch '):
        slot = int(command.split()[1])
        if not 1 <= slot <= 6:
            raise ValueError('Invalid switch slot')
        return 8 + slot - 1
    if not command.startswith('/choose move '):
        raise ValueError('Unsupported Foul Play command: '+command)
    tokens = command[len('/choose move '):].split()
    if len(tokens) not in (1,2) or (len(tokens)==2 and tokens[1]!='terastallize'):
        raise ValueError('Unsupported move modifier')
    def to_id(s): return ''.join(c for c in s.lower() if c.isalnum())
    matches = [i for i,m in enumerate(request['active'][0]['moves']) if to_id(m['id'])==to_id(tokens[0])]
    if len(matches)!=1:
        raise ValueError('Move absent/ambiguous in current request: '+command)
    return matches[0] + (4 if len(tokens)==2 else 0)


class FoulPlayAdapter:
    def __init__(self, side, search_time_ms=100, seed=0):
        if side not in ('p1','p2') or type(search_time_ms) is not int or not 2 <= search_time_ms <= 10000:
            raise ValueError('Invalid Foul Play seat or search budget')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Invalid Foul Play seed')
        pin, dataset = verify_runtime()
        from fp.config import FoulPlayConfig
        FoulPlayConfig.pokemon_format = 'gen9randombattle'
        FoulPlayConfig.search_time_ms = search_time_ms
        FoulPlayConfig.parallelism = 1
        FoulPlayConfig.search_threads = 1
        FoulPlayConfig.team_preview_search_time_ms = search_time_ms
        FoulPlayConfig.team_preview_search_parallelism = 1
        FoulPlayConfig.log_to_file = False
        from fp.battle.state import Battle
        from fp.modes.random_battle import RandomBattleMode
        self.battle = Battle('battle-f34-offline')
        self.battle.pokemon_format = 'gen9randombattle'
        self.battle.generation = self.battle.format_spec.generation
        self.battle.battle_type = self.battle.format_spec.battle_type
        self.battle.mode = RandomBattleMode()
        self.battle.mode.datasets.raw_pkmn_sets = json.loads(dataset.read_text())
        self.battle.mode.datasets.pkmn_mode = 'gen9randombattle'
        self.battle.mode.datasets._initialize_pkmn_sets()
        self.battle.user.name = side
        self.battle.opponent.name = 'p2' if side=='p1' else 'p1'
        self.side = side
        self.request_count = 0
        self.last_decided_request = -1
        self.initial_lines = []
        self.pending = False
        self.search_time_ms = search_time_ms
        random.seed(seed)

    def ingest(self, envelope):
        validate_input(envelope, self.side)
        from fp.battle.protocol import update_battle, process_battle_updates
        for line in envelope['protocol']:
            if line.startswith('|request|'):
                self.request_count += 1
                request = json.loads(line[9:])
                if request is None:
                    continue
                request['rqid'] = self.request_count  # local request counter, no game information
                line = '|request|' + json.dumps(request,separators=(',',':'))
                if not self.battle.started:
                    if not request.get('side'):
                        raise ValueError('First request lacks own roster')
                    self.battle.user.initialize_first_turn_user_from_json(request)
                    self.battle.request_json = request
                    self.battle.rqid = self.request_count
                    self.battle.started = True
                    # Same initialization as RandomBattleMode.start_battle: the
                    # own initial switch is already represented by the request.
                    try: start = self.initial_lines.index('|start') + 1
                    except ValueError: raise ValueError('Initial start event missing')
                    self.battle.msg_list = [x for x in self.initial_lines[start:]
                        if not x.startswith('|switch|'+self.side)]
                    process_battle_updates(self.battle)
                    self.initial_lines.clear()
                update_battle(self.battle, line)
            elif not self.battle.started:
                self.initial_lines.append(line)
            else:
                update_battle(self.battle, line)
        if self.request_count != envelope['request_id']:
            raise ValueError('External protocol cursor/request mismatch')
        self.pending = envelope['pending']

    def choose(self):
        if not self.pending or not self.battle.started or self.battle.wait:
            raise ValueError('Foul Play cannot choose while waiting')
        if self.last_decided_request == self.request_count:
            raise ValueError('Duplicate Foul Play decision for request')
        if (self.battle.rqid != self.request_count or
                not isinstance(self.battle.request_json,dict) or
                self.battle.request_json.get('rqid') != self.request_count):
            raise ValueError('Foul Play request state is stale')
        from copy import deepcopy
        from fp.battle.state import LastUsedMove
        from fp.modes.base import format_decision
        from fp.search.main import find_best_move
        battle_copy = deepcopy(self.battle)
        battle_copy.user.update_from_request_json(battle_copy.request_json)
        samples, per_sample_ms = battle_copy.mode.search_params(battle_copy)
        started = time.perf_counter()
        decision = find_best_move(battle_copy)
        # Mirror pinned async_pick_move exactly: switch decisions already contain
        # the 'switch ' prefix. format_decision only formats the wire command.
        self.battle.user.last_selected_move = LastUsedMove(self.battle.user.active.name,
            decision.removesuffix('-tera').removesuffix('-mega'),self.battle.turn)
        wire = format_decision(battle_copy,decision)
        action = decision_to_action(wire,battle_copy.request_json)
        self.last_decided_request = self.request_count
        return {'action':action,'decision':decision,'command':wire[0],
            'request_id':self.request_count,'seconds':time.perf_counter()-started,
            'search_time_ms':self.search_time_ms,
            'sampled_worlds':samples,'per_world_ms':per_sample_ms,
            'configured_search_total_ms':samples*per_sample_ms,
            'opponent_revealed':len(self.battle.opponent.reserve)+int(self.battle.opponent.active is not None)}


def main():
    logging.basicConfig(level=logging.WARNING,stream=sys.stderr)
    adapter = None
    for raw in sys.stdin:
        command = None
        try:
            command = json.loads(raw)
            op = command['op']
            if op=='start':
                if set(command) != {'op','id','side','search_time_ms','seed'}: raise ValueError('Start inventory')
                adapter=FoulPlayAdapter(command['side'],command['search_time_ms'],command['seed'])
                data={'started':True}
            elif op=='choose' and adapter:
                if set(command) != {'op','id','input'}: raise ValueError('Choose inventory')
                adapter.ingest(command['input'])
                data=adapter.choose() if adapter.pending else {'waiting':True}
            elif op=='close':
                adapter=None;data={'closed':True}
            else: raise ValueError('Unknown operation or missing adapter')
            print(json.dumps({'id':command['id'],'ok':True,'data':data}),flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'id':command.get('id') if isinstance(command,dict) else None,
                'ok':False,'error':str(exc),'error_type':type(exc).__name__}),flush=True)
            adapter=None


if __name__=='__main__':
    main()
