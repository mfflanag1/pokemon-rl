import copy, json, subprocess, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from pokemon_terminal_matchup.transport import Worker
import numpy as np
from pokemon_terminal_matchup.features import encode, extras
from pokemon_terminal_matchup.common import ROOT, read, PLAN, evaluation_specs, load_policy, check_collection
from pokemon_terminal_matchup.collector import Pool
from pokemon_matchup.network import extend

CAT=read(ROOT/read(PLAN)['catalog'])
FAILED=read(ROOT/'artifacts/reviews/f23-matchup-ppo/failure-player.json')

class Dynamic(unittest.TestCase):
    def test_spawn_failure_closes_log_and_preserves_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            worker=Worker.__new__(Worker)
            with patch('pokemon_terminal_matchup.transport.subprocess.Popen',side_effect=OSError('synthetic spawn failure')):
                with self.assertRaises(OSError):worker.__init__(Path(directory))
            self.assertTrue(worker.stderr.closed)
            paths=list(Path(directory).glob('initialization-failure-*.json'))
            self.assertEqual(len(paths),1)
            self.assertEqual(read(paths[0])['stage'],'worker-initialization')

    def test_exact_rejected_pawmot_player_is_finite_in_both_arms(self):
        self.assertTrue(any(m and m['types']==['???','Fighting'] for m in FAILED['opponent']))
        for arm in ['zero','matchup']:
            x,m=encode(FAILED,CAT,arm)
            self.assertEqual(x.shape,(1278,));self.assertTrue(np.isfinite(x).all())
            self.assertEqual(bool(np.any(x[928:])),arm=='matchup')

    def test_known_typeless_is_neutral_distinct_from_unrevealed(self):
        p=copy.deepcopy(FAILED)
        foe=next(m for m in p['opponent'] if m and m['active']);foe['types']=['???'];foe['tera']=None
        r=extras(p,CAT).reshape(14,25);moves=np.flatnonzero(np.asarray(p['mask'])[:8])
        for a in moves:
            self.assertEqual(r[a,9],1);self.assertEqual(r[a,10],.25)
        foe['types']=['???','???'];r=extras(p,CAT).reshape(14,25)
        for a in moves:self.assertEqual(r[a,10],.25)
        p['opponent']=[None]*6;r=extras(p,CAT).reshape(14,25)
        for a in moves:self.assertEqual(r[a,9],0);self.assertEqual(r[a,10],0)

    def test_own_removed_type_loses_nominal_stab_and_added_third_is_used(self):
        p=copy.deepcopy(FAILED);own=next(m for m in p['own'] if m['active'])
        p['active_moves'][0].update(id='thunderbolt',type='Electric',category='Special',base_power=90)
        p['mask']=[True]+[False]*13;own['types']=['Electric'];own['tera']=None
        self.assertEqual(extras(p,CAT).reshape(14,25)[0,8],.75)
        own['types']=['???'];self.assertEqual(extras(p,CAT).reshape(14,25)[0,8],.5)
        foe=next(m for m in p['opponent'] if m and m['active']);foe['types']=['Water','Flying','Grass'];foe['tera']=None
        self.assertEqual(extras(p,CAT).reshape(14,25)[0,10],.5)
        for bad in [[],['Bird'],['Fire']*4]:
            foe['types']=bad
            for arm in ['zero','matchup']:
                with self.assertRaises(ValueError):encode(p,CAT,arm)

    def test_v2_schema_requires_only_authorized_own_fields(self):
        p=copy.deepcopy(FAILED);p['schema_version']='full-showdown-player-v2'
        for m in p['own']:m['types']=None
        encode(p,CAT,'matchup')
        for obj in [p['own'][0],next(m for m in p['opponent'] if m)]:
            obj['private_set']='hidden'
            with self.assertRaises(ValueError):encode(p,CAT,'matchup')
            del obj['private_set']

    def test_exact_failed_chunk_now_completes_both_arms_without_updates(self):
        plan=read(ROOT/'contracts/matchup-ppo-v1/plan.json')
        p,_,_=load_policy(read(ROOT/'contracts/shared-battle-v1/plan.json'),7103,'reference');p,_=extend(p)
        jobs=evaluation_specs(plan,7103,'random',100,4)
        outcomes=[]
        with tempfile.TemporaryDirectory() as directory:
            for arm in ['zero','matchup']:
                with Pool(Path(directory)/arm,8,arm,CAT) as pool:collected,_=pool.collect(p,jobs)
                check_collection(collected,100,4,'development','random')
                self.assertEqual(len(collected),8);self.assertTrue(all(not rows for rows,_ in collected))
                outcomes.append([g['outcome'] for _,g in collected])
        self.assertEqual(*outcomes)

    def test_f24_exact_failure_batch_completes_both_arms_without_updates(self):
        jobs=read(ROOT/'artifacts/runs/f24-dynamic-matchup-v1/7103/reference-workers/collector-failure.json')['specs']
        p,_,_=load_policy(read(ROOT/'contracts/shared-battle-v1/plan.json'),7103,'reference');p,_=extend(p)
        outcomes=[]
        with tempfile.TemporaryDirectory() as directory:
            for arm in ['zero','matchup']:
                with Pool(Path(directory)/arm,8,arm,CAT) as pool:collected,_=pool.collect(p,jobs)
                check_collection(collected,72,4,'development','tactical')
                self.assertEqual(len(collected),8);self.assertTrue(all(not rows for rows,_ in collected))
                outcomes.append([g['outcome'] for _,g in collected])
        self.assertEqual(*outcomes)

    def test_failed_chunk_history_matches_original_coordinates(self):
        script="""import {officialPlan} from './packages/full-training/session.mjs';
import {readFileSync} from 'node:fs';
const d=JSON.parse(readFileSync('artifacts/reviews/f23-matchup-ppo/failure-diagnostic.json'));
console.log(JSON.stringify(d.specs.map(s=>({spec:s,teams:officialPlan(s.episode,s.seed,s.split,s.swap).teams.map(t=>({team_key:t.team_key,partition:t.partition}))}))));"""
        r=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(r.stdout),read(ROOT/'contracts/dynamic-matchup-v1/f23-failed-chunk.json'))

if __name__=='__main__':unittest.main()
