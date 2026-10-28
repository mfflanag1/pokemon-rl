"""Exact F37 cutoff qualification using fresh imports and mocked acquisition only."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOOT = '''
from copy import deepcopy
import importlib, json
from pathlib import Path
from unittest.mock import patch
from pokemon_teacher_legality import deadline as d

def plan():
    p=json.loads((d.ROOT/'contracts/teacher-legality-v1/plan.json').read_text())
    p['deadline_utc']=d.DEADLINE_UTC
    p['execution']['deadline_authorization']={'path':d.AUTHORIZATION_RELATIVE,'sha256':d.AUTHORIZATION_SHA256}
    return p
'''


class DeadlineTests(unittest.TestCase):
    def child(self, code):
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(BOOT) + textwrap.dedent(code)],
            cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT/'python')},
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_uninstalled_f35_behavior_remains_original(self):
        self.child('''
            from pokemon_teacher_assisted import common
            original=json.loads((d.ROOT/d.ORIGINAL_PLAN_RELATIVE).read_text())
            assert common.DEADLINE == d.ORIGINAL_DEADLINE
            assert common.validate_plan(original) is original
            changed=deepcopy(original); changed['deadline_utc']=d.DEADLINE_UTC
            try: common.validate_plan(changed)
            except ValueError: pass
            else: raise AssertionError('Frozen validator generalized without overlay')
        ''')

    def test_exact_plan_identity_and_fixed_hard_boundary(self):
        self.child('''
            evidence=d.install_deadline()
            from pokemon_teacher_assisted import common
            p=plan(); before=deepcopy(p)
            assert common.validate_plan(p) is p and p==before
            assert common.DEADLINE == d.DEADLINE
            assert evidence['clock_reset'] is False
            for now in (d.ORIGINAL_DEADLINE+1, d.DEADLINE-0.001):
                with patch.object(common.time,'time',return_value=now): common.before_deadline()
            for now in (d.DEADLINE, d.DEADLINE+1):
                with patch.object(common.time,'time',return_value=now):
                    try: common.before_deadline()
                    except TimeoutError: pass
                    else: raise AssertionError('Hard cutoff admitted work')
        ''')

    def test_all_other_plan_fields_and_deadlines_rejected(self):
        self.child('''
            d.install_deadline()
            from pokemon_teacher_assisted import common
            cases=[]
            for timestamp in (d.ORIGINAL_DEADLINE_UTC,'2026-10-28T18:00:01Z','2026-10-29T18:00:00Z'):
                p=plan();p['deadline_utc']=timestamp;cases.append(p)
            for key,value in [('start_utc','2026-10-28T11:30:00Z'),('checkpoint_utc','2026-10-28T15:30:00Z'),
                              ('cohorts_per_seed_arm',257),('workers',9),('seeds',[7103,8209])]:
                p=plan();p[key]=value;cases.append(p)
            for key,value in [('acquisition_cap_seconds',4501),('search_time_ms',101),('train_games_per_host',130)]:
                p=plan();p['teacher_data'][key]=value;cases.append(p)
            p=plan();p['expected_counts']['optimizer_steps']+=1;cases.append(p)
            p=plan();del p['execution']['deadline_authorization'];cases.append(p)
            p=plan();p['execution']['teacher_policy']['policy_id']='old-native';cases.append(p)
            for p in cases:
                try: common.validate_plan(p)
                except ValueError: pass
                else: raise AssertionError('Unauthorized plan change admitted')
        ''')

    def test_six_frozen_stages_capture_approved_deadline_after_full_overlay(self):
        self.child('''
            from pokemon_teacher_legality.stage import install_overlay
            evidence=install_overlay()
            from pokemon_teacher_assisted import common
            assert evidence['deadline']['deadline_epoch']==d.DEADLINE
            for name in d.STAGES:
                module=importlib.import_module(name)
                if hasattr(module,'DEADLINE'): assert module.DEADLINE==d.DEADLINE,name
                if hasattr(module,'before_deadline'):
                    assert module.before_deadline is common.before_deadline,name
                    with patch.object(common.time,'time',return_value=d.DEADLINE):
                        try: module.before_deadline()
                        except TimeoutError: pass
                        else: raise AssertionError(name)
            # All six obtain their remaining common checks through the same
            # patched module, including local imports in demonstrations.acquire.
            assert common.before_deadline.__globals__['DEADLINE']==d.DEADLINE
            assert d.install_deadline()==evidence['deadline']
        ''')

    def test_every_preimported_stage_is_rejected_before_mutation(self):
        for stage in ('schedules','demonstrations','campaign','external_evaluation','verification','reporting'):
            with self.subTest(stage=stage):
                self.child('''
                    importlib.import_module('pokemon_teacher_assisted.%s')
                    from pokemon_teacher_assisted import common
                    try: d.install_deadline()
                    except RuntimeError as exc: assert 'precede stage imports' in str(exc)
                    else: raise AssertionError('Stale stage accepted')
                    assert common.DEADLINE==d.ORIGINAL_DEADLINE
                ''' % stage)

    def test_stale_deadline_and_validator_aliases_rejected(self):
        for attribute in ('DEADLINE','saved_validator'):
            with self.subTest(attribute=attribute):
                self.child('''
                    from types import ModuleType
                    import sys
                    from pokemon_teacher_assisted import common
                    alias=ModuleType('deadline_stale_fixture')
                    setattr(alias,%r, common.DEADLINE if %r=='DEADLINE' else common.validate_plan)
                    sys.modules[alias.__name__]=alias
                    try: d.install_deadline()
                    except RuntimeError as exc: assert 'aliases' in str(exc)
                    else: raise AssertionError('Stale alias accepted')
                    assert common.DEADLINE==d.ORIGINAL_DEADLINE
                ''' % (attribute,attribute))

    def test_repeat_install_checks_bindings_and_cannot_reset_clock(self):
        self.child('''
            first=d.install_deadline()
            from pokemon_teacher_assisted import common
            assert d.install_deadline()==first
            first['deadline_epoch']=0
            assert d.install_deadline()['deadline_epoch']==d.DEADLINE
            common.DEADLINE+=1
            try: d.install_deadline()
            except RuntimeError: pass
            else: raise AssertionError('Changed common binding accepted')
            common.DEADLINE=d.DEADLINE
            schedules=importlib.import_module('pokemon_teacher_assisted.schedules')
            schedules.DEADLINE=d.ORIGINAL_DEADLINE
            try: d.install_deadline()
            except RuntimeError: pass
            else: raise AssertionError('Changed stage binding accepted')
        ''')

    def test_missing_or_changed_approval_rejected_before_install(self):
        self.child('''
            import tempfile
            original_root=d.ROOT
            with tempfile.TemporaryDirectory() as folder:
                d.ROOT=Path(folder)
                try: d.install_deadline()
                except FileNotFoundError: pass
                else: raise AssertionError('Missing approval accepted')
                path=d.ROOT/d.AUTHORIZATION_RELATIVE;path.parent.mkdir(parents=True)
                body=json.loads((original_root/d.AUTHORIZATION_RELATIVE).read_text())
                body['approved_deadline_utc']='2026-10-29T18:00:00Z'
                path.write_text(json.dumps(body))
                try: d.install_deadline()
                except ValueError as exc: assert 'hash changed' in str(exc)
                else: raise AssertionError('Modified approval accepted')
            assert d._evidence is None
        ''')

    def test_authorization_changes_after_install_are_not_accepted(self):
        self.child('''
            import tempfile
            d.install_deadline()
            from pokemon_teacher_assisted import common
            p=plan()
            with tempfile.TemporaryDirectory() as folder, patch.object(d,'ROOT',Path(folder)):
                for operation in (d.install_deadline,lambda:common.validate_plan(p)):
                    try: operation()
                    except FileNotFoundError: pass
                    else: raise AssertionError('Lost approval was ignored')
        ''')

    def test_acquisition_cap_uses_plan_deadline_without_game_or_checkpoint(self):
        self.child('''
            import tempfile
            from pokemon_teacher_legality.stage import install_overlay
            install_overlay()
            from pokemon_teacher_assisted import common, demonstrations
            class StopBeforeCheckpoint(Exception): pass
            p=plan()
            for start in (d.DEADLINE-7200,d.DEADLINE-30):
                with tempfile.TemporaryDirectory(dir=d.ROOT/'artifacts/reviews') as temp:
                    folder=Path(temp); path=folder/'plan.json';path.write_text(json.dumps(p))
                    frozen={'plan':p,'source_files':{str(path.relative_to(d.ROOT)):'fake-hash'}}
                    with patch.object(common,'verify_inputs',return_value=frozen), \\
                         patch.object(common,'load_initial',side_effect=StopBeforeCheckpoint), \\
                         patch.object(common.time,'time',return_value=start), \\
                         patch.object(demonstrations,'digest',return_value='fake-hash'), \\
                         patch.object(demonstrations,'run_game',side_effect=AssertionError('No games')):
                        try: demonstrations.acquire(folder/'output',path,folder/'admission')
                        except StopBeforeCheckpoint: pass
                        else: raise AssertionError('Checkpoint boundary not reached')
                    schedule=json.loads((folder/'output/schedule.json').read_text())
                    assert schedule['acquisition_deadline_epoch']==min(start+4500,d.DEADLINE)
                    assert len(schedule['schedule'])==480
        ''')

    def test_fresh_driver_import_captures_approved_deadline(self):
        self.child('''
            import importlib.util
            spec=importlib.util.spec_from_file_location('f37_deadline_driver_fixture',d.ROOT/'scripts/run-teacher-legality.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            assert module.DEADLINE==d.DEADLINE
            assert module.before_deadline.__globals__['DEADLINE']==d.DEADLINE
        ''')


if __name__ == '__main__':
    unittest.main()
