import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest
import numpy as np
from pokemon_soak.campaign import admission,check_diagnostics,check_gap,counters,fingerprint,replay
from pokemon_showdown.loop import Executables
from pokemon_showdown.checkpoint import Runner,load

ROOT=Path(__file__).resolve().parents[2]

class CampaignTests(unittest.TestCase):
    def test_selected_contract_and_diagnostic_failures(self):
        self.assertAlmostEqual(check_gap((100.,1000.),(101.,1001.),30),1.)
        for current in ((101.,4600.),(200.,1100.),(101.,900.)):
            with self.assertRaises(ValueError):check_gap((100.,1000.),current,30)
        a=admission();self.assertEqual(a['plan']['qualification_seconds'],7200)
        self.assertEqual(a['selected']['contract']['version'],'showdown-chance-1')
        report={'minimum_behavior_probability':2**-24,'metrics':{'max_ratio_error':np.array([0.,.1])}}
        check_diagnostics(report)
        for x in (0,2**-25,1.1,float('nan')):
            bad=dict(report,minimum_behavior_probability=x)
            with self.assertRaises(ValueError):check_diagnostics(bad)
        bad=copy.deepcopy(report);bad['metrics']['max_ratio_error'][0]=float('inf')
        with self.assertRaises(ValueError):check_diagnostics(bad)

    def test_original_and_restored_fingerprints(self):
        exe=Executables();r=Runner(73,exe)
        for _ in range(3):r.run()
        initial=counters(r)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'boundary.json';item=replay(path,r,exe)
            self.assertEqual(initial,counters(r))
            restored=load(path,exe);metrics=restored.run()
            self.assertEqual(fingerprint(restored.learner),item['next_learner_fingerprint'])
            self.assertEqual(fingerprint(metrics),item['next_metrics_fingerprint'])
            changed=copy.deepcopy(metrics);changed['decisions']+=2
            self.assertNotEqual(fingerprint(changed),item['next_metrics_fingerprint'])
            mod=runpy.run_path(str(ROOT/'scripts/seal-policy-soak.py'))
            self.assertEqual(mod['verify_replays'](path.parent,[item],exe,['boundary.json'],73),[initial])
            bad=dict(item,next_learner_fingerprint='wrong')
            with self.assertRaises(ValueError):mod['verify_replays'](path.parent,[bad],exe,['boundary.json'],73)
            stub=path.parent/'old-run';stub.mkdir();(stub/'report.json').write_text('{"status":"running"}')
            self.assertIsNone(mod['historical_report'](stub)['content_hash'])

    def test_telemetry_missing_reordered_and_unlinked(self):
        mod=runpy.run_path(str(ROOT/'scripts/seal-policy-soak.py'))
        plan=json.loads((ROOT/'contracts/soak-v1/plan.json').read_bytes())
        initial={'updates':8,'next_game':512,'decisions':3200,'executed_moves':2800}
        rows=[]
        for i in range(1,121):
            row={'scheduled_seconds':i*60,'elapsed_seconds':i*60+.001,'cycles':i*100,'games':i*6400,
                 'decisions':i*40000,'executed_moves':i*35000,'counters':{'updates':8+i*100,'next_game':512+i*6400,
                 'decisions':3200+i*40000,'executed_moves':2800+i*35000},'unexpected_compilations':0,
                 'maximum_host_gap_seconds':.02,'memory':{'peak_process_rss_bytes':500000000}}
            rows.append(row)
        s={k:rows[-1][k] for k in ('cycles','games','decisions','executed_moves')}
        s.update(telemetry_rows=120,elapsed_seconds=7200.1,initial_counters=initial,peak_process_rss_bytes=500000000)
        validate=mod['validate_telemetry'];validate(rows,s,plan)
        for bad in (rows[:-1],rows[::-1]):
            with self.assertRaises(ValueError):validate(bad,s,plan)
        for key,value in [('scheduled_seconds',61),('elapsed_seconds',100),('unexpected_compilations',1),('games',1)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):validate(bad,s,plan)

if __name__=='__main__':unittest.main()
