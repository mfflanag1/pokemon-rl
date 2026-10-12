import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('gates',ROOT/'python/pokemon_switch_soak/gates.py')
gates=importlib.util.module_from_spec(spec);spec.loader.exec_module(gates)
PLAN=json.loads((ROOT/'contracts/switch-soak-v1/plan.json').read_text())


def fixture(smoke=False):
    seconds,interval=(20,4) if smoke else (7200,60)
    def counts(n):return dict(cycles=n,games=32*n,decisions=400*n,terminal=10*n,truncated=22*n,actions=[100*n]*4)
    def counters(n):return dict(updates=8+n,optimizer_step=2*(8+n),next_game=32*(8+n))
    s=dict(**counts(seconds),elapsed_seconds=float(seconds),process_cpu_seconds=9000.,cycle_seconds=seconds*.8,
        maximum_host_gap_seconds=.1,unexpected_compilations=0,warmup_compilation_count=20,minimum_logp=-12.,maximum_logp=0.,
        peak_process_rss_bytes=500000000,initial_counters=counters(0),final_counters=counters(seconds),
        telemetry_rows=seconds//interval,replays=4 if smoke else 13,decisions_per_elapsed_second=400.)
    rows=[dict(**counts(i),scheduled_seconds=i,elapsed_seconds=float(i),counters=counters(i),
        maximum_host_gap_seconds=.1,unexpected_compilations=0,minimum_logp=-12.,maximum_logp=0.,
        memory={'peak_process_rss_bytes':500000000}) for i in range(interval,seconds+1,interval)]
    r=dict(status='passed',mode='smoke' if smoke else 'qualification',two_hour_qualification=not smoke,
        final_reservation_unchanged=True,compile_control={'first_call_compilation_events':1,'same_shape_repeat_events':0},soak=s)
    for k in ('full_g2','full_g3','full_g4','public_ladder_strength','strength_evaluation','final_evaluation_used','public_server_used','prime_used'):r[k]=PLAN[k]
    return r,copy.deepcopy(PLAN),rows


class Gates(unittest.TestCase):
    def test_valid_modes(self):
        for smoke in (False,True):self.assertIsNone(gates.validate(*fixture(smoke)))

    def test_scope_and_target(self):
        for field,value in [('status','running'),('two_hour_qualification',False),('final_evaluation_used',True),('full_g3','closed')]:
            r,p,rows=fixture();r[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):gates.validate(r,p,rows)
        for key in ('qualification_seconds','seed','warmup_cycles','maximum_updates'):
            r,p,rows=fixture();p[key]=float(p[key])
            with self.subTest(key=key),self.assertRaises(ValueError):gates.validate(r,p,rows)
        r,p,rows=fixture();p['qualification_seconds']=10
        with self.assertRaises(ValueError):gates.validate(r,p,rows)

    def test_linkage_and_counts(self):
        for field in ('cycles','games','decisions','terminal','truncated'):
            r,p,rows=fixture();r['soak'][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):gates.validate(r,p,rows)
        r,p,rows=fixture();r['soak']['final_counters']['next_game']+=32
        with self.assertRaises(ValueError):gates.validate(r,p,rows)

    def test_instability(self):
        for field,value in [('elapsed_seconds',7199.),('maximum_host_gap_seconds',31.),('minimum_logp',-34.),('maximum_logp',.01),('peak_process_rss_bytes',2**31+1),('unexpected_compilations',1),('replays',12)]:
            r,p,rows=fixture();r['soak'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):gates.validate(r,p,rows)

    def test_telemetry_corruption(self):
        for change in ('missing','order','counters','actions','peak','negative'):
            r,p,rows=fixture()
            if change=='missing':rows.pop()
            if change=='order':rows[1]=copy.deepcopy(rows[0])
            if change=='counters':rows[1]['counters']['updates']+=1
            if change=='actions':rows[1]['actions'][0]+=1
            if change=='peak':rows[1]['memory']['peak_process_rss_bytes']=600000000
            if change=='negative':rows[1]['truncated']=-1
            with self.subTest(change=change),self.assertRaises(ValueError):gates.validate(r,p,rows)

    def test_malformed(self):
        for value in (None,[],{},True):
            r,p,rows=fixture()
            with self.subTest(value=value),self.assertRaises(ValueError):gates.validate(value,p,rows)
        for value in (True,float('nan'),float('inf'),'7200',None):
            r,p,rows=fixture();r['soak']['elapsed_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):gates.validate(r,p,rows)


if __name__=='__main__':unittest.main()
