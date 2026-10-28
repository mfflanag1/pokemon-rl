import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
import json
import runpy
import tempfile
import unittest
from pokemon_learner.checkpoint import digest,file_hash,wire
G=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/seal-live-learner.py'))


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(wire(data)+b'\n')


class GateTests(unittest.TestCase):
    def test_review_requires_roster_completed_packet_and_bound_dispositions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);expected={'implementation.py':'abc'}
            write(root/'implementation-final/request.json',{'engine':'codex','mode':'review','files_sha256':expected})
            write(root/'implementation-final/status.json',{'status':'completed'})
            (root/'implementation-final/result.md').write_text('Independent review result')
            write(root/'claude-fallback-unavailable/status.json',{'status':'failed'})
            (root/'review-dispositions.md').write_text('Locally verified findings')
            record={'format':'live-learner-review-1','review_family':'codex','cross_family_completed':False,
                    'blocking_findings_open':0,'tests_verified':True,'reviewed_source_files':expected}
            for field,path in [('request_sha256','implementation-final/request.json'),('status_sha256','implementation-final/status.json'),
                ('result_sha256','implementation-final/result.md'),('dispositions_sha256','review-dispositions.md'),
                ('unavailable_status_sha256','claude-fallback-unavailable/status.json')]:record[field]=file_hash(root/path)
            write(root/'review-verification.json',record);self.assertEqual(G['review_gate'](root,expected),record)
            for bad in ({**record,'reviewed_source_files':{}},{**record,'cross_family_completed':True},
                        {**record,'blocking_findings_open':1},{**record,'result_sha256':'wrong'}):
                write(root/'review-verification.json',bad)
                with self.assertRaises(ValueError):G['review_gate'](root,expected)
            write(root/'review-verification.json',record)
            (root/'review-dispositions.md').write_text('Changed after adjudication')
            with self.assertRaises(ValueError):G['review_gate'](root,expected)

    def test_launcher_binds_command_run_admission_report_and_log(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'review';out=Path(td)/'run';folder.mkdir();out.mkdir();admitted={'test':1}
            write(out/'report.json',{'status':'test'});(folder/'production.log').write_text('test run')
            record={'format':'live-learner-launch-1','command':[str(G['ROOT']/'.venv/bin/python'),'-m','pokemon_live.canary',str(out)],
                'run':str(out),'admission_hash':digest(admitted),'exit_code':0,'started_at':'2026-10-13T01:00:00+00:00',
                'finished_at':'2026-10-13T01:00:02+00:00','elapsed_seconds':2.0,
                'report_sha256':file_hash(out/'report.json'),'log_sha256':file_hash(folder/'production.log')}
            write(folder/'exit.json',record);self.assertEqual(G['exit_gate'](folder,out,admitted),record)
            for bad in ({'exit_code':0},{**record,'command':['true']},{**record,'admission_hash':'wrong'},
                        {**record,'report_sha256':'wrong'},{**record,'elapsed_seconds':100}):
                write(folder/'exit.json',bad)
                with self.assertRaises(ValueError):G['exit_gate'](folder,out,admitted)
            write(folder/'exit.json',record);(folder/'production.log').write_text('altered')
            with self.assertRaises(ValueError):G['exit_gate'](folder,out,admitted)

if __name__=='__main__':unittest.main()
