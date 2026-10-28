"""Source-bound engineering admission before any F33 scientific panel."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import re

from pokemon_full_train.storage import ROOT,atomic_json,digest,freeze,admission
from pokemon_full_train.learning import derived_seed
from pokemon_worker_watch.transport import Worker
from .core import COMMAND,Search,load_frozen,predictions,frozen_actions
from .campaign import PLAN,scientific,sha,play


def qualify(path):
    path.mkdir(parents=True,exist_ok=False);plan=json.loads(PLAN.read_text());manifest=freeze(path,plan)
    started=time.perf_counter();checks=[]
    commands=[['node','--test','tests/opening-search/world.test.mjs'],
        [str(ROOT/'.venv/bin/python'),'-m','unittest','discover','-s','tests/opening-search','-p','test_*.py','-v'],
        ['node','--input-type=module','-e',
         "import {verifyBuild,verifySource} from './packages/reference-node/provenance.mjs'; verifySource(); console.log(JSON.stringify(verifyBuild()));"]]
    for i,command in enumerate(commands):
        with (path/f'check-{i}.log').open('xb') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        checks.append({'command':command,'exit_code':result.returncode,'log_sha256':digest(path/f'check-{i}.log')})
        if result.returncode:raise RuntimeError('Qualification check failed: '+str(i))
    prior=json.loads((ROOT/'artifacts/reports/f32-observed-continuation.json').read_text())
    for rel,expected in prior['source_files'].items():
        if digest(ROOT/rel)!=expected:raise ValueError('Frozen predecessor source changed: '+rel)
    roots=[];game_checks=[]
    for seed in plan['seeds']:
        p,catalog,identity=load_frozen(seed)
        real=Worker(path/f'{seed}/real',command=COMMAND);hyp=Worker(path/f'{seed}/hypothetical',command=COMMAND)
        try:
            search=Search(hyp,p,catalog,plan['search'])
            for episode in range(plan['engineering_openings']):
                swap=bool(episode%2);side='p2' if swap else 'p1'
                f=real.request('start',run_seed=plan['engineering_seed'],episode_index=episode,
                    split='development',swap=swap,tactical_side=None)
                input=real.request('opening',side=side);prior_prob,_=predictions(p,[f['players'][side]],catalog)[0]
                fallback=frozen_actions(p,[f['players'][side]],catalog,[seed],[0])[0][2]
                search_seed=derived_seed('f33-search-input',seed,sha(input))
                root=search.decide(input,search_seed,prior_prob,fallback)
                root['seed']=seed;root['episode']=episode;root['input']=input
                atomic_json(path/f'{seed}/root-{episode}.json',root);roots.append(root)
                if episode==0:
                    repeated=search.decide(input,search_seed,prior_prob,fallback)
                    expected={k:v for k,v in root.items() if k not in ('seed','episode','input')}
                    if scientific(repeated)!=scientific(expected):raise ValueError('Search output reproducibility')
                if not input['player']['mask'][root['action']]:raise ValueError('Root legality')
                if not root['applied'] and (root['action']!=fallback or root['probabilities']!=list(map(float,prior_prob))):
                    raise ValueError('Fallback identity')
            # Full episode plus fresh replay exercises action/value tapes and
            # cutoff-aware finalization without touching a scientific panel.
            for arm in ('policy','search'):
                first=play(real,p,catalog,plan,seed=plan['engineering_seed'],episode=0,swap=False,
                    opponent='tactical',model_seed=seed,search=search if arm=='search' else None)
                replay=play(real,p,catalog,plan,seed=plan['engineering_seed'],episode=0,swap=False,
                    opponent='tactical',model_seed=seed,search=search if arm=='search' else None)
                if scientific(first)!=scientific(replay):raise ValueError('Engineering full-game replay')
                atomic_json(path/f'{seed}/game-{arm}.json',first)
                game_checks.append({'seed':seed,'arm':arm,'decisions':first['decisions'],'outcome':first['outcome']})
        finally:real.close();hyp.close()
    admission(path,manifest)
    applied=sum(r['applied'] for r in roots);maximum=max(r['seconds'] for r in roots)
    report={'status':'passed' if applied/len(roots)>=plan['minimum_engineering_search_fraction'] and
        maximum<=plan['engineering_maximum_root_seconds'] else 'failed',
        'checks':checks,'source_files':len(manifest['sources']),'plan_sha256':digest(PLAN),
        'input_sha256':digest(path/'inputs/manifest.json'),'f32_source_files_unchanged':len(prior['source_files']),
        'roots':len(roots),'applied':applied,'maximum_root_seconds':maximum,'game_checks':game_checks,
        'physical_full_games':2*len(game_checks),'seconds':time.perf_counter()-started,
        'tests':{'node':int(re.search(r'tests\s+(\d+)',(path/'check-0.log').read_text()).group(1)),
            'python':int(re.search(r'Ran (\d+) tests',(path/'check-1.log').read_text()).group(1))},
        'frozen_source_manifest':manifest['sources']}
    atomic_json(path/'report.json',report)
    if report['status']!='passed':raise RuntimeError('Engineering coverage/latency admission failed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    qualify(Path(parser.parse_args().output).resolve())
