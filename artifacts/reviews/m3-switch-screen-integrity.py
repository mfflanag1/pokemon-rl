"""Post-seal hash verification; intentionally outside the sealed review folder."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,file_hash
p=argparse.ArgumentParser();p.add_argument('--expect-raw');p.add_argument('--output',required=True);args=p.parse_args()
path=ROOT/'artifacts/reports/m3-switch-screen.json';r=json.loads(path.read_bytes());raw=file_hash(path)
assert r['content_hash']==digest({k:v for k,v in r.items() if k!='content_hash'})
archive=ROOT/'artifacts/manifests'/('m3-switch-screen-'+r['content_hash']);assert (archive/'receipt.json').read_bytes()==path.read_bytes()
checks=0
for kind,field in [('source','source_files'),('evidence','evidence_files')]:
 for name,h in r[field].items():
  assert file_hash(ROOT/name)==h,('live',name);assert file_hash(archive/kind/name)==h,('archive',name);checks+=2
if args.expect_raw:assert args.expect_raw==raw,'Resealing changed receipt bytes'
result={'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'content_hash':r['content_hash'],'receipt_sha256':raw,
        'source_files':len(r['source_files']),'evidence_files':len(r['evidence_files']),'live_archive_hash_checks':checks,
        'archived_receipt_identical':True,'repeat_seal_same_receipt_bytes':bool(args.expect_raw),
        'learning_screen_passed':r['audit']['all_seed_screen_passed']}
Path(args.output).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
