"""Bind the existing full regression plus M4 and M5 tests to the expanded source tree."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import wire, file_hash
sources=runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']


def main(directory):
    out=Path(directory).resolve()
    if not out.is_relative_to(ROOT/'artifacts/reviews'):raise ValueError('Review output path')
    out.mkdir(parents=True,exist_ok=False)
    before=sources();results=[]
    commands=[['npm','test'],*[ [sys.executable,'-m','unittest','discover','-s',folder,'-p','test_*.py'] for folder in ('tests/bounded-value','tests/continuation') ]]
    for label,command in zip(('predecessor','bounded','continuation'),commands):
        log=out/(label+'.log')
        with log.open('xb') as stream:
            code=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'python')},stdout=stream,stderr=subprocess.STDOUT).returncode
        results.append({'command':command,'exit_code':code,'log':log.name,'sha256':file_hash(log)})
        if code:break
    passed=len(results)==3 and all(r['exit_code']==0 for r in results) and before==sources()
    report={'format':'continuation-regression-1','status':'passed' if passed else 'failed',
            'source_files':before,'sources_unchanged':before==sources(),'results':results}
    with (out/'verification.json').open('xb') as stream:stream.write(wire(report)+b'\n')
    print(json.dumps({'status':report['status'],'source_files':len(before)}))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main(sys.argv[1])
