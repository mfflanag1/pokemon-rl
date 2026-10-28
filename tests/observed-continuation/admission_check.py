"""F32-specific admission, history separation and unchanged campaign guards."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pokemon_observed_continuation import common as c, campaign
from pokemon_full_train.storage import ROOT, canonical, digest
from pokemon_worker_watch.collector import Pool


def game(key):
    return {'provenance':{'teams':[{'team_key':key}]}}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(canonical(value)+b'\n')


class Admission(unittest.TestCase):
    def test_plan_does_not_reuse_failed_evaluation_or_relax_budget(self):
        plan=c.read(c.PLAN);c.validate_plan(plan);old=c.read(c.BASE_PLAN)
        for key in ['evaluation_seed','selfplay_evaluation_seed']:
            self.assertNotEqual(plan[key],old[key])
            bad=copy.deepcopy(plan);bad[key]=old[key]
            with self.assertRaises(ValueError):c.validate_plan(bad)
        self.assertEqual(plan['arm_budgets'],old['arm_budgets'])
        self.assertEqual(plan['expected_counts'],old['expected_counts'])
        for key in ['worker_qualification_sha256','failed_execution_sha256','observed_transport']:
            bad=copy.deepcopy(plan);bad[key]='changed'
            with self.assertRaises(ValueError):c.validate_plan(bad)

    def test_all_new_panels_differ_from_f29_and_failed_f30(self):
        plan=c.read(c.PLAN)
        old=[c.read(c.BASE_PLAN),c.read(ROOT/'contracts/nonlinear-policy-v1/plan.json')]
        for seed in plan['seeds']:
            fresh=set()
            for opponent in plan['opponents']:
                jobs=c.evaluation_specs(plan,seed,opponent,0,128)
                self.assertEqual(len(jobs),256)
                fresh.update(j['seed'] for j in jobs)
                for previous in old:
                    self.assertTrue({j['seed'] for j in jobs}.isdisjoint(j['seed'] for j in c.evaluation_specs(previous,seed,opponent,0,128)))
            diag=c.diagnostic_specs(plan,seed,0,16)
            self.assertTrue(fresh.isdisjoint(j['seed'] for j in diag))
            for previous in old:
                self.assertTrue({j['seed'] for j in diag}.isdisjoint(j['seed'] for j in c.diagnostic_specs(previous,seed,0,16)))

    def test_campaign_uses_watched_pool_and_preserves_algorithm(self):
        self.assertIs(campaign.Pool,Pool)
        old=ast.parse((ROOT/'python/pokemon_nonlinear_continuation/campaign.py').read_text())
        new=ast.parse((ROOT/'python/pokemon_observed_continuation/campaign.py').read_text())
        for tree in [old,new]:
            tree.body=[node for node in tree.body if not isinstance(node,ast.ImportFrom) or not any(a.name=='Pool' for a in node.names)]
        self.assertEqual(ast.dump(old),ast.dump(new))

    def test_partial_history_keeps_training_main_and_diagnostic_games(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write(root/'7103/control/update-256.json',{'ledger':{'games':[game('train')]}})
            write(root/'9311/control/256/games/random-0000.json',{'games':[game('main')]})
            write(root/'7103/nonlinear/512/selfplay/0000.json',{'games':[game('diagnostic')]})
            training,development=set(),set();c.add_history(root,training,development)
            self.assertEqual(training,{'train'});self.assertEqual(development,{'main','diagnostic'})

    def receipt_fixture(self,root):
        write(root/'source.py',{'source':1});write(root/'evidence.json',{'evidence':1})
        body={'source_files':{'source.py':digest(root/'source.py')},'evidence_files':{'evidence.json':digest(root/'evidence.json')}}
        h=hashlib.sha256(canonical(body)).hexdigest();receipt={**body,'content_hash':h}
        path=root/'receipt.json';write(path,receipt)
        archive=root/'artifacts/manifests'/('f31-worker-watch-'+h)
        write(archive/'receipt.json',receipt)
        for group,rel in [('source','source.py'),('evidence','evidence.json')]:
            (archive/group).mkdir();(archive/group/rel).write_bytes((root/rel).read_bytes())
        return path,archive

    def test_preserved_receipt_checks_live_and_archive_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,archive=self.receipt_fixture(root);expected=digest(path)
            with patch.object(c,'ROOT',root):
                c.preserved_receipt(path,expected,'f31-worker-watch')
                for p in [root/'source.py',archive/'source/source.py',root/'evidence.json',archive/'evidence/evidence.json',archive/'receipt.json']:
                    original=p.read_bytes();p.write_bytes(b'corrupt')
                    with self.assertRaises(ValueError):c.preserved_receipt(path,expected,'f31-worker-watch')
                    p.write_bytes(original)
                with self.assertRaises(ValueError):c.preserved_receipt(path,'0'*64,'f31-worker-watch')

    def history_fixture(self,root):
        failed=root/'failed'
        write(failed/'7103/control/update-256.json',{'ledger':{'games':[game('f30-train')]}})
        write(failed/'7103/control/256/games/random-0000.json',{'games':[game('f30-dev')]})
        for i in range(6):
            write(root/f'artifacts/diagnostics/f31-worker-timeout/parity/{i}/control/original/result.json',{'ledger':{'games':[game('f31-'+str(i))]}})
        return {'status':'passed','summary':{'tests':{'total':37}}},{'status':'failed','scientific_comparison_completed':False,'run':'failed'}

    def test_admission_includes_partial_f30_f31_and_own_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);watched,failed=self.history_fixture(root)
            pf=root/'artifacts/reviews/f32-observed-continuation/preflight';write(pf/'report.json',{'status':'passed'})
            for i in range(6):write(pf/f'{i}/control/result.json',{'ledger':{'games':[game('f32-'+str(i))]}})
            with patch.object(c,'ROOT',root),patch.object(c,'validate_plan'),patch.object(c,'preserved_receipt',side_effect=[watched,failed]),patch.object(c,'prior_admit',return_value=('prior','oracle',{'old-train'},{'old-dev'})):
                previous,oracle,train,past=c.admit(c.read(c.PLAN))
            self.assertEqual((previous,oracle),('prior','oracle'))
            self.assertEqual(len(train),14);self.assertEqual(past,{'old-dev','f30-dev'})
            self.assertTrue({'f30-train','f31-0','f32-5'}<=train)

    def test_history_role_collision_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);watched,failed=self.history_fixture(root)
            with patch.object(c,'ROOT',root),patch.object(c,'validate_plan'),patch.object(c,'preserved_receipt',side_effect=[watched,failed]),patch.object(c,'prior_admit',return_value=('prior','oracle',{'f30-dev'},set())):
                with self.assertRaisesRegex(ValueError,'Historical role collision'):c.admit(c.read(c.PLAN))

    def test_failed_predecessor_cannot_be_relabelled_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);watched,failed=self.history_fixture(root);failed['scientific_comparison_completed']=True
            with patch.object(c,'ROOT',root),patch.object(c,'validate_plan'),patch.object(c,'preserved_receipt',side_effect=[watched,failed]):
                with self.assertRaisesRegex(ValueError,'failed-execution identity'):c.admit(c.read(c.PLAN))

if __name__=='__main__':unittest.main()
