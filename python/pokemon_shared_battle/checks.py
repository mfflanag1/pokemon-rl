"""Focused checks bound to the exact F21 source/runtime inventory."""
import argparse,os,re,subprocess
from pathlib import Path
from pokemon_full_train.storage import ROOT,source_paths,digest,atomic_json,runtime
COMMANDS=[('actor-tests.log',['.venv/bin/python','-B','tests/shared-battle/actor_check.py'],9),
    ('scope-tests.log',['.venv/bin/python','-B','tests/shared-battle/scope_check.py'],8),
    ('collector-tests.log',['.venv/bin/python','-B','tests/shared-battle/collector_check.py'],4)]

def check(review):
    review.mkdir(parents=True,exist_ok=True);before={str(p.relative_to(ROOT)):digest(p) for p in source_paths()};rt=runtime();runs=[]
    for name,cmd,count in COMMANDS:
        result=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'python'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        path=review/name;path.write_bytes(result.stdout);text=result.stdout.decode()
        passed=bool(re.search(rf'Ran {count} tests? in ',text) and '\nOK\n' in text)
        runs.append({'command':cmd,'exit_code':result.returncode,'log':str(path.relative_to(ROOT)),'sha256':digest(path),'tests':count,'count_verified':passed})
        if result.returncode or not passed:raise ValueError('Focused check failed: '+name)
    after={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    if before!=after or runtime()!=rt:raise ValueError('Source/runtime changed during checks')
    result={'status':'passed','tests_passed':sum(x[2] for x in COMMANDS),'sources':after,'runtime':rt,'runs':runs}
    atomic_json(review/'checks.json',result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);print(check(p.parse_args().review.resolve())['tests_passed'],'tests passed',flush=True)
