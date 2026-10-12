import copy
import json
from pathlib import Path
import unittest
from pokemon_soak.gates import validate_summary

ROOT=Path(__file__).resolve().parents[2]
PLAN=json.loads((ROOT/'contracts/soak-v1/plan.json').read_text())

def good_report():
    return {'status':'passed','mode':'qualification','two_hour_qualification':True,
            'full_g2':'open','full_g3':'open','prime_used':False,'strength_evaluation':False,
            'final_reservation_unchanged':True,
            'compile_control':{'first_call_compilation_events':1,'same_shape_repeat_events':0},
            'soak':{'target_seconds':7200,'elapsed_seconds':7200.1,'cycles':1000,'games':64000,
                    'decisions':400000,'executed_moves':350000,'cycle_seconds':7190.,
                    'process_cpu_seconds':7000.,'maximum_host_gap_seconds':.02,
                    'minimum_exp_behavior_logp':2**-24,'maximum_update_ratio_error':.2,
                    'unexpected_compilations':0,'warmup_compilation_count':123,
                    'traces':{'cycle':1,'collect':1,'update':1},'telemetry_rows':120,
                    'periodic_replays':11,'replays':13,'device_memory_headroom_certified':False,
                    'initial_counters':{'updates':8,'next_game':512,'decisions':3200,'executed_moves':2800},
                    'final_counters':{'updates':1008,'next_game':64512,'decisions':403200,'executed_moves':352800},
                    'peak_process_rss_bytes':500000000}}

class GateTests(unittest.TestCase):
    def test_qualified_and_floor_rounding(self):
        validate_summary(good_report(),PLAN)
        r=good_report();r['soak']['minimum_exp_behavior_logp']=2**-24*(1-2e-6);validate_summary(r,PLAN)

    def test_scope_time_and_compile_controls(self):
        for path,value in [(['mode'],'smoke'),(['two_hour_qualification'],False),(['full_g2'],'passed'),
                (['full_g3'],'passed'),(['prime_used'],True),(['strength_evaluation'],True),
                (['final_reservation_unchanged'],False),(['status'],'running'),
                (['soak','elapsed_seconds'],7199),(['soak','target_seconds'],60),
                (['soak','maximum_host_gap_seconds'],31),(['soak','unexpected_compilations'],1),
                (['soak','warmup_compilation_count'],0),(['soak','traces'],{'cycle':2,'collect':1,'update':1}),
                (['compile_control','first_call_compilation_events'],0),
                (['compile_control','same_shape_repeat_events'],1),
                (['soak','device_memory_headroom_certified'],True)]:
            r=good_report();p=r
            for k in path[:-1]:p=p[k]
            p[path[-1]]=value
            with self.subTest(path=path,value=value),self.assertRaises(ValueError):validate_summary(r,PLAN)

    def test_numerical_accounting_and_coverage(self):
        for field,value in [('elapsed_seconds',10**10000),('traces',{'cycle':1,'collect':1,3:1})]:
            r=good_report();r['soak'][field]=value
            with self.assertRaises(ValueError):validate_summary(r,PLAN)
        for field,value in [('cycles',0),('games',63999),('decisions',100),('executed_moves',0),
                ('cycle_seconds',0),('cycle_seconds',7300),('process_cpu_seconds',0),
                ('minimum_exp_behavior_logp',0),('minimum_exp_behavior_logp',2**-25),
                ('minimum_exp_behavior_logp',1.1),('maximum_update_ratio_error',-1),
                ('maximum_update_ratio_error',2**25),('telemetry_rows',10),
                ('periodic_replays',10),('replays',12),('peak_process_rss_bytes',0),
                ('elapsed_seconds',float('nan')),('maximum_update_ratio_error',float('inf'))]:
            r=good_report();r['soak'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate_summary(r,PLAN)
        for field in ('updates','next_game','decisions','executed_moves'):
            r=good_report();r['soak']['final_counters'][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):validate_summary(r,PLAN)

if __name__=='__main__':unittest.main()
