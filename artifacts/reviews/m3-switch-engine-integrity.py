"""Post-seal integrity proof; output intentionally outside the sealed review tree."""
import argparse,json,runpy
from pathlib import Path
from pokemon_learner.checkpoint import digest,file_hash,wire
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--repeat',action='store_true');args=p.parse_args()
path=ROOT/'artifacts/reports/m3-switch-engine.json';r=json.loads(path.read_bytes());h=r['content_hash']
assert digest({k:v for k,v in r.items() if k!='content_hash'})==h
archive=ROOT/'artifacts/manifests'/('m3-switch-engine-'+h)
assert (archive/'receipt.json').read_bytes()==path.read_bytes()
checks=0
for kind,key in [('source','source_files'),('evidence','evidence_files')]:
 for name,expected in r[key].items():
  assert file_hash(ROOT/name)==expected,('live',name)
  assert file_hash(archive/kind/name)==expected,('archive',name)
  checks+=2
preflight=json.loads((ROOT/'artifacts/reviews/m3-switch-engine/preflight-original-hashes.json').read_bytes())
for name,row in preflight.items():
 assert file_hash(ROOT/name)==row['sha256']
 assert file_hash(ROOT/row['copy'])==row['sha256']
sealer=runpy.run_path(str(ROOT/'scripts/seal-switch-engine.py'));now=sealer['admission']()
reg=json.loads((ROOT/r['run']/'preregistration.json').read_bytes())
assert now==reg['admission']
v=json.loads((ROOT/'artifacts/reviews/m3-switch-engine/verification.json').read_bytes())
assert v['source_files']==now['source_files'] and v['sources_unchanged']
proof={'status':'pass','receipt_content_hash':h,'receipt_sha256':file_hash(path),
 'source_files':len(r['source_files']),'evidence_files':len(r['evidence_files']),
 'live_and_archived_hash_checks':checks,'predecessor_hash_checks':now['predecessor_hash_checks'],
 'original_preflight_files':len(preflight),'tested_sources':len(v['source_files']),
 'repeat_seal_identical':args.repeat}
if args.repeat:
 first=json.loads((ROOT/'artifacts/reviews/m3-switch-engine-integrity-first.json').read_bytes())
 assert first['receipt_sha256']==proof['receipt_sha256'] and first['receipt_content_hash']==h
 log=ROOT/'artifacts/reviews/m3-switch-engine-reseal.log';result=json.loads(log.read_bytes())
 assert result['hash']==h and result['status']==r['status'];proof['repeat_log_sha256']=file_hash(log)
out=ROOT/('artifacts/reviews/m3-switch-engine-integrity.json' if args.repeat else 'artifacts/reviews/m3-switch-engine-integrity-first.json')
with out.open('xb') as stream:stream.write(wire(proof)+b'\n')
print(json.dumps(proof))
