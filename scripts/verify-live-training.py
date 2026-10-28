"""Bind inherited and new M7/F1/C1 regression to one unchanged source tree."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_live_train import checkpoint as cp, evidence as e


def main(directory):
    out=Path(directory).resolve()
    if not out.is_relative_to(ROOT/'artifacts/reviews'):raise ValueError('Review path')
    out.mkdir(parents=True,exist_ok=False);before=cp.sources();rows=[]
    commands=[('inherited',['npm','test']), *[(name,[sys.executable,'-m','unittest','discover','-s','tests/'+folder,'-p','test_*.py'])
        for name,folder in [('bounded','bounded-value'),('continuation','continuation'),('live','live-learner'),('training','live-training'),('full-python','full-showdown')]],
        ('full-node',['node','--test',*map(str,sorted((ROOT/'tests/full-showdown').glob('*.test.mjs')))]),
        ('curriculum',['node','--test',*map(str,sorted((ROOT/'tests/training-curriculum').glob('*.test.mjs')))])]
    for label,command in commands:
        log=out/(label+'.log')
        with log.open('xb') as f:
            code=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'python'),'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT).returncode
        rows.append({'label':label,'command':command,'exit_code':code,'log':log.name,'sha256':e.file_hash(log)})
        if code:break
    passed=len(rows)==len(commands) and all(x['exit_code']==0 for x in rows) and before==cp.sources()
    e.once(out/'verification.json',{'status':'passed' if passed else 'failed','source_files':before,'results':rows,'sources_unchanged':before==cp.sources()})
    print(json.dumps({'status':'passed' if passed else 'failed','sources':len(before),'commands':len(rows)}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main(sys.argv[1])
