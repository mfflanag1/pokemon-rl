"""Focused learner, role, checkpoint and live-collector qualification."""
import argparse,os,re,subprocess
from pathlib import Path
from pokemon_full_train.storage import ROOT,source_paths,digest,atomic_json,runtime
COMMANDS=[('session-tests.log',['node','--test','--test-reporter=tap','tests/opponent-mix/session.test.mjs'],4,'node'),
    ('scope-tests.log',['.venv/bin/python','-B','tests/opponent-mix/scope_check.py'],7,'unittest'),
    ('collector-tests.log',['.venv/bin/python','-B','tests/opponent-mix/collector_check.py'],7,'unittest')]


def check(review):
    review.mkdir(parents=True,exist_ok=True)
    before={str(p.relative_to(ROOT)):digest(p) for p in source_paths()};rt=runtime();runs=[]
    for name,cmd,count,marker in COMMANDS:
        result=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'python'},
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        path=review/name;path.write_bytes(result.stdout);text=result.stdout.decode()
        passed=bool(re.search(rf'Ran {count} tests? in ',text) and '\nOK\n' in text) if marker=='unittest' else bool(re.search(rf'^# tests {count}$',text,re.M) and re.search(r'^# fail 0$',text,re.M) and re.search(r'^# skipped 0$',text,re.M) and re.search(r'^# cancelled 0$',text,re.M))
        runs.append({'command':cmd,'exit_code':result.returncode,'log':str(path.relative_to(ROOT)),
            'sha256':digest(path),'check_groups':count,'marker':marker,'verified':passed})
        if result.returncode or not passed:raise ValueError('Focused check failed: '+name)
    after={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    if before!=after or runtime()!=rt:raise ValueError('Source/runtime changed during qualification')
    result={'status':'passed','check_groups_passed':sum(x[2] for x in COMMANDS),
        'sources':after,'runtime':rt,'runs':runs}
    atomic_json(review/'checks.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True)
    print(check(p.parse_args().review.resolve())['check_groups_passed'],'check groups passed',flush=True)
