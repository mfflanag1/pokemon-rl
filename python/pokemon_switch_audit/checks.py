import argparse,os,subprocess
from pathlib import Path
from pokemon_full_train.storage import ROOT,source_paths,digest,atomic_json,runtime
COMMANDS=[('coverage-tests.log',['.venv/bin/python','-B','tests/switch-audit/coverage_check.py'],8,'Ran 8 tests'),('projection-tests.log',['node','tests/switch-audit/projection_check.mjs'],9,'Passed 9 projection checks'),('replay-tests.log',['.venv/bin/python','-B','tests/switch-audit/replay_check.py'],2,'Ran 2 tests'),('semantic-tests.log',['.venv/bin/python','-B','tests/switch-audit/semantic_check.py'],9,'Ran 9 tests')]
def check(review):
    sources={str(p.relative_to(ROOT)):digest(p) for p in source_paths()};rt=runtime();runs=[]
    for name,cmd,count,marker in COMMANDS:
        r=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':'python','PYTHONDONTWRITEBYTECODE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        (review/name).write_bytes(r.stdout);text=r.stdout.decode()
        if r.returncode or marker not in text or (cmd[0].endswith('python') and '\nOK\n' not in text):raise ValueError('Check failed '+name)
        runs.append({'command':cmd,'tests':count,'marker':marker,'exit_code':r.returncode,'log':str((review/name).relative_to(ROOT)),'sha256':digest(review/name)})
    if sources!={str(p.relative_to(ROOT)):digest(p) for p in source_paths()} or rt!=runtime():raise ValueError('Tested source/runtime drift')
    atomic_json(review/'checks.json',{'status':'passed','tests_passed':sum(r['tests'] for r in runs),'sources':sources,'runtime':rt,'runs':runs})
    print('28 focused checks passed on',len(sources),'source/configuration files')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--review',type=Path,required=True);check(a.parse_args().review.resolve())
