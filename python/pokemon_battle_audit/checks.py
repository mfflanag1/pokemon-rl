"""Executed, log-bound focused regression receipt for the frozen F11 tree."""
import argparse
import os
from pathlib import Path
import re
import subprocess
from pokemon_full_train.storage import ROOT, source_paths, digest, atomic_json

COMMANDS=[
    ('node-tests.log',['node','--test','tests/battle-audit/describe.test.mjs','tests/battle-audit/session.test.mjs',
                      'tests/full-showdown/protocol.test.mjs','tests/full-training/protocol.test.mjs'],29),
    ('python-tests.log',['.venv/bin/python','-m','unittest','discover','-s','tests/battle-audit','-p','test_*.py'],4),
    ('feedback-tests.log',['.venv/bin/python','-m','unittest','discover','-s','tests/full-transfer-v2','-p','test_*.py'],2)]


def check(review):
    review.mkdir(parents=True,exist_ok=True)
    before={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    runs=[]
    for name,command,count in COMMANDS:
        result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'python'},
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        log=review/name; log.write_bytes(result.stdout)
        text=result.stdout.decode()
        matched=(re.search(rf'(?:tests|# tests) {count}\b',text) if name.startswith('node') else
                 re.search(rf'Ran {count} tests? in ',text) and '\nOK\n' in text)
        runs.append({'command':command,'exit_code':result.returncode,'log':str(log.relative_to(ROOT)),
                     'sha256':digest(log),'tests':count,'count_verified':bool(matched)})
        if result.returncode or not matched: raise ValueError(f'Check failed: {name}')
    after={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    if before!=after: raise ValueError('Source changed during tests')
    record={'status':'passed','tests_passed':sum(c for _,_,c in COMMANDS),'sources':after,'runs':runs}
    atomic_json(review/'checks.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True)
    record=check(p.parse_args().review.resolve());print(record['tests_passed'],'tests passed')
