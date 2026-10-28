"""Focused executed checks bound to the exact production source inventory."""
import argparse
from pathlib import Path
import os
import re
import subprocess
from pokemon_full_train.storage import ROOT,source_paths,digest,atomic_json,runtime

COMMANDS=[('returns-tests.log',['.venv/bin/python','tests/credit/returns_check.py'],8),
    ('integration-tests.log',['.venv/bin/python','tests/credit/integration_check.py'],7)]


def check(review):
    review.mkdir(parents=True,exist_ok=True)
    before={str(p.relative_to(ROOT)):digest(p) for p in source_paths()};tested_runtime=runtime();runs=[]
    for name,command,count in COMMANDS:
        result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'python'},
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        log=review/name;log.write_bytes(result.stdout);text=result.stdout.decode()
        matched=bool(re.search(rf'Ran {count} tests? in ',text) and '\nOK\n' in text)
        runs.append({'command':command,'exit_code':result.returncode,'log':str(log.relative_to(ROOT)),
                     'sha256':digest(log),'tests':count,'count_verified':matched})
        if result.returncode or not matched:raise ValueError(f'Check failed: {name}')
    after={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    if before!=after or runtime()!=tested_runtime:raise ValueError('Source/runtime changed during checks')
    record={'status':'passed','tests_passed':sum(c for _,_,c in COMMANDS),'sources':after,'runtime':tested_runtime,'runs':runs}
    atomic_json(review/'checks.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True)
    print(check(p.parse_args().review.resolve())['tests_passed'],'tests passed',flush=True)
