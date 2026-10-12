import copy
import json
from pathlib import Path
import unittest
import tempfile
from pokemon_pilot.campaign import verify_record,Reference
from pokemon_pilot.provenance import mechanics_seed

ROOT=Path(__file__).resolve().parents[2]


class EvidenceTests(unittest.TestCase):
    def record(self):
        p=json.loads((ROOT/'contracts/pilot-v1/plan.json').read_text())
        c=json.loads((ROOT/'tests/showdown-chance/fixtures/player-case.json').read_text())
        final=c['turns'][-1]['views']
        win=next(x for x in final[0]['protocol'] if x.startswith('|win|'))
        r={'seed':101,'update':0,'opponent':'random','pair':0,'seat':0,
           'mechanics_seed':mechanics_seed(p,101,'random',0,'smoke'),'winner':int(win[-1])-1,
           'decisions':[{'turn':t['turn'],'actions':t['actions']} for t in c['turns']], 'final_views':final}
        return p,r

    def test_public_outcomes_roster_and_action_tape(self):
        p,r=self.record();self.assertIn(verify_record(r,p,101,0,'random',0,'smoke'),(0.,1.))
        for field,value in [('winner',1-r['winner']),('pair',1),('mechanics_seed','wrong')]:
            bad=copy.deepcopy(r);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):verify_record(bad,p,101,0,'random',0,'smoke')

    def test_executed_action_tamper_and_missing_turn_fail(self):
        p,r=self.record();bad=copy.deepcopy(r)
        bad['decisions'][0]['actions']=[1-a for a in bad['decisions'][0]['actions']]
        with self.assertRaises(ValueError):verify_record(bad,p,101,0,'random',0,'smoke')
        for index in (0,-1):
            bad=copy.deepcopy(r);bad['decisions'].pop(index)
            with self.assertRaises(ValueError):verify_record(bad,p,101,0,'random',0,'smoke')

    def test_reference_worker_exits_after_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=Reference(Path(directory)/'stderr.log')
            try:
                worker.finish();self.assertEqual(worker.process.returncode,0)
            finally:
                if worker.process.poll() is None:worker.abort()


if __name__=='__main__':unittest.main()
