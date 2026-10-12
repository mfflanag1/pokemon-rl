"""One-shot, persistent sealing stage; all outputs stay outside sealed reviews."""
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('phase',choices=['first','repeat']);args=p.parse_args()
base=ROOT/'artifacts/reviews'/('m3-switch-screen-seal-'+args.phase)
assert json.loads((ROOT/'artifacts/reviews/m3-switch-screen/exit.json').read_text())['exit_code']==0
assert json.loads((ROOT/'artifacts/runs/m3-switch-screen-v1/report.json').read_text())['status']=='passed'
command=[str(ROOT/'.venv/bin/python'),'scripts/seal-switch-screen.py','artifacts/runs/m3-switch-screen-v1']
with base.with_suffix('.launch.json').open('x') as f:json.dump({'phase':args.phase,'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'command':command},f,indent=2)
with base.with_suffix('.log').open('xb') as log:
 result=subprocess.run(['/usr/bin/caffeinate','-i',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
with base.with_suffix('.exit.json').open('x') as f:json.dump({'phase':args.phase,'exit_code':result.returncode,'ended_at':datetime.now(timezone.utc).isoformat()},f,indent=2)
sys.exit(result.returncode)
