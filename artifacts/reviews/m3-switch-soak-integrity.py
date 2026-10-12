"""Verify every current/live and archived receipt entry; output stays outside its archive."""
import hashlib,json,runpy,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def wire(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
path=ROOT/'artifacts/reports/m3-switch-soak.json';receipt=json.loads(path.read_text())
assert receipt['content_hash']==hashlib.sha256(wire({k:v for k,v in receipt.items() if k!='content_hash'})).hexdigest()
assert receipt['status']=='complete-restricted-cpu-qualification' and receipt['two_hour_qualification'] is True
archive=ROOT/'artifacts/manifests'/('m3-switch-soak-'+receipt['content_hash'])
assert (archive/'receipt.json').read_bytes()==path.read_bytes()
counts={}
for kind,field in [('source','source_files'),('evidence','evidence_files')]:
 files=receipt[field]
 for name,expected in files.items():
  assert (ROOT/name).resolve().is_relative_to(ROOT),name
  assert sha(ROOT/name)==expected,'Live drift: '+name
  assert sha(archive/kind/name)==expected,'Archived drift: '+name
 counts[kind]=len(files)
review=ROOT/'artifacts/reviews/m3-switch-soak'
v=json.loads((review/'verification.json').read_text())
assert v['status']=='passed' and v['sources_unchanged'] is True
assert v['source_files']==runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()
assert v['test_log_sha256']==sha(review/'npm-test.log')
assert json.loads((review/'exit.json').read_text())['exit_code']==0
report=json.loads((ROOT/receipt['run']/'report.json').read_text())
assert report==receipt['report'] and report['status']=='passed'
assert json.loads((review/'build-verification.json').read_text())==json.loads((review/'completion-build-verification.json').read_text())
result={'format':'switch-soak-integrity-1','status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),
 'receipt_content_hash':receipt['content_hash'],'receipt_file_sha256':sha(path),'archived_receipt_identical':True,
 'source_files':counts['source'],'evidence_files':counts['evidence'],'live_and_archive_hash_checks':2*sum(counts.values()),
 'tested_source_files_unchanged':len(v['source_files']),'tests':receipt['tests'],
 'preflight_and_completion_build_equal':True,'launcher_exit_code':0,'full_g2':'open','full_g3':'open','full_g4':'open',
 'public_ladder_strength':'unqualified','final_evaluation_used':False,'prime_used':False}
if len(sys.argv)>1 and sys.argv[1]=='--repeat':
 first=json.loads((ROOT/'artifacts/reviews/m3-switch-soak-integrity-first.json').read_text())
 assert first['receipt_content_hash']==result['receipt_content_hash'] and first['receipt_file_sha256']==result['receipt_file_sha256']
 result['repeat_seal_same_receipt_bytes']=True
print(json.dumps(result,sort_keys=True,indent=2))
