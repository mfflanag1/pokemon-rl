"""Boundary tests run in the isolated Foul Play environment; no search/network."""
import json
import asyncio
from copy import deepcopy
from unittest.mock import patch
import tempfile
from pathlib import Path
import socket
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from pokemon_external_benchmark.foul_play import FoulPlayAdapter,validate_input,decision_to_action,ROOT

class AdapterBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'artifacts/reviews/f34-policy-reset/external-integration/timing-smoke-001/transcript.jsonl'
        cls.initial=json.loads(p.read_text().splitlines()[0])['external']

    def test_first_observation_reveals_only_opponent_lead(self):
        adapter=FoulPlayAdapter('p2',100,1);adapter.ingest(self.initial)
        self.assertEqual(len(adapter.battle.user.reserve),5)
        self.assertEqual(len(adapter.battle.opponent.reserve),0)
        self.assertEqual(adapter.battle.opponent.active.name,'dragonite')
        self.assertEqual(len(adapter.battle.opponent.active.moves),0)
        self.assertEqual(adapter.request_count,1)

    def test_reject_private_opponent_payload(self):
        value={**self.initial,'protocol':['|request|{"side":{"id":"p1"}}']}
        with self.assertRaisesRegex(ValueError,'Foreign'):validate_input(value,'p2')

    def test_reject_coordinator_fields(self):
        for key in ['players','team','future_actions','state','opponent_private']:
            with self.assertRaises(ValueError):validate_input({**self.initial,key:{}},'p2')

    def test_reject_unsplit_and_multiline(self):
        for line in ['|split|p1','|turn|1\n|request|{}']:
            with self.assertRaises(ValueError):validate_input({**self.initial,'protocol':[line]},'p2')

    def test_network_denied(self):
        with socket.socket() as sock:
            with self.assertRaisesRegex(RuntimeError,'offline'):sock.connect(('127.0.0.1',9))

    def test_current_request_action_mapping(self):
        request={'active':[{'moves':[{'id':'thunderbolt'},{'id':'Hidden Power Ice'}]}]}
        self.assertEqual(decision_to_action(['/choose move thunderbolt','1'],request),0)
        self.assertEqual(decision_to_action(['/choose move thunderbolt terastallize','1'],request),4)
        self.assertEqual(decision_to_action(['/choose move hiddenpowerice','1'],request),1)
        self.assertEqual(decision_to_action(['/switch 6','1'],request),13)
        with self.assertRaises(ValueError):decision_to_action(['/choose move surf','1'],request)
        with self.assertRaises(ValueError):decision_to_action(['/switch 7','1'],request)

    def test_duplicate_protocol_is_rejected(self):
        adapter=FoulPlayAdapter('p2',100,1);adapter.ingest(self.initial)
        with self.assertRaisesRegex(ValueError,'cursor'):adapter.ingest(self.initial)

    def test_upstream_move_tera_switch_bookkeeping_parity(self):
        from fp.modes.base import async_pick_move
        for decision in ['spikes','bravebird-tera','switch kingambit']:
            with self.subTest(decision=decision):
                adapter=FoulPlayAdapter('p2',100,1);adapter.ingest(self.initial)
                upstream=deepcopy(adapter.battle)
                with patch('fp.modes.base.find_best_move',return_value=decision):
                    expected=asyncio.run(async_pick_move(upstream))
                with patch('fp.search.main.find_best_move',return_value=decision):
                    observed=adapter.choose()
                self.assertEqual(observed['command'],expected[0])
                self.assertEqual(adapter.battle.user.last_selected_move,upstream.user.last_selected_move)
                self.assertEqual(adapter.battle.user.last_selected_move.move,decision.removesuffix('-tera'))

    def test_stale_request_is_rejected_before_search(self):
        adapter=FoulPlayAdapter('p2',100,1);adapter.ingest(self.initial)
        adapter.battle.request_json['rqid']=0
        with patch('fp.search.main.find_best_move') as search:
            with self.assertRaisesRegex(ValueError,'stale'):adapter.choose()
            search.assert_not_called()

    def test_actual_switch_target_after_request_reordering(self):
        from pokemon_external_benchmark.transport import JsonWorker
        evidence=ROOT/'artifacts/reviews/f34-policy-reset/external-integration'
        with tempfile.TemporaryDirectory(prefix='review-slot-test-',dir=evidence) as temp:
            with JsonWorker(['node','packages/external-benchmark/worker.mjs'],Path(temp)/'node.stderr') as node:
                bundle=node.request('start',run_seed=1027341901,episode_index=0,
                    swap=False,external_side='p2',tactical_side=None)
                adapter=FoulPlayAdapter('p2',100,1)
                # Both use request slot 2, but that slot changes from Kingambit
                # to the original lead after the first actual engine switch.
                for expected_target in ['kingambit','delibird']:
                    adapter.ingest(bundle['external'])
                    request=adapter.battle.request_json
                    self.assertEqual(request['rqid'],bundle['external']['request_id'])
                    self.assertEqual(request['side']['pokemon'][1]['details'].split(',')[0].lower(),expected_target)
                    with patch('fp.search.main.find_best_move',return_value='switch '+expected_target):
                        choice=adapter.choose()
                    self.assertEqual(choice['command'],'/switch 2')
                    self.assertEqual(choice['action'],9)
                    ids={side:bundle['frame']['players'][side]['request_id'] for side in ['p1','p2']}
                    bundle=node.request('step',actions={'p1':1,'p2':choice['action']},request_ids=ids)
                    switch_lines=[line for line in bundle['external']['protocol'] if line.startswith('|switch|p2a:')]
                    self.assertTrue(switch_lines)
                    resolved=switch_lines[0].split('|')[3].split(',')[0].lower()
                    self.assertEqual(resolved,expected_target)
                    self.assertTrue(all(row['accepted'] for row in bundle['frame']['submitted']))

if __name__=='__main__':unittest.main()
