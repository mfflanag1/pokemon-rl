"""Verify and archive the fixed-shape CPU loop milestone; full G2/G3 stay open."""
import argparse
import json
from pathlib import Path
import re
import runpy
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require
from pokemon_fixed.campaign import admission
from pokemon_fixed.loop import Executables
from pokemon_fixed.checkpoint import load

REVIEW=ROOT/'artifacts/reviews/m1-fixed-loop'


def read(path):return json.loads(path.read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Changed sealed artifact: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(data)


def main(directory):
    run=Path(directory).resolve();require(run.is_relative_to(ROOT/'artifacts/runs'),'Run directory')
    result=read(run/'report.json');admitted=read(run/'admission.json');p=admitted['plan']
    require(result['content_hash']==digest({k:v for k,v in result.items() if k!='content_hash'}),'Result hash')
    require(result['status']=='passed' and result['full_g2']==result['full_g3']=='open','Gate status')
    control=result['compile']['logging_positive_control']
    require(control['first_call_compilation_events']>=1 and control['same_shape_repeat_events']==0,'Real compilation positive control')
    require(result['admission_hash']==digest(admitted) and admitted==admission(),'Admission/source drift')
    require(admitted['prior_gates']['final_reservation_unchanged'] is True,'Final reservation gate')
    parity,profile,soak=[result[k] for k in ('parity','profile','soak')]
    for key in ('parity','profile','soak'):
        require(result[key]==read(run/f'{key}.json') and result[key]['status']=='passed','Result linkage '+key)
    batches=len(p['parity_seeds'])*p['parity_batches_per_seed']
    require(parity['games']==64*batches and parity['disk_resumes_with_compared_next_cycle']==batches,'Parity budget')
    require(parity['atol']==p['atol'] and parity['rtol']==p['rtol'],'Tolerance drift')
    for value in parity['comparisons'].values():
        require(value['comparisons']>0 and value['elements']>0 and 0<=value['max_fraction_of_tolerance']<=1,'Numerical parity')
    actor_width={'actions':1,'features':59,'game_id':1,'mask':2,'old_logp':1,'old_values':1,'side':1,'turn':1}
    parameter_width={'policy_b':2,'policy_w':118,'value_b':1,'value_w':59}
    expected={f'actor/{k}':(batches,parity['decisions']*v) for k,v in actor_width.items()}
    expected.update({f'{group}/{k}':(batches,parity['decisions']) for group,names in
        [('targets',('advantages','returns','rewards','terminal')),('behavior',('old_logp','old_values'))] for k in names})
    expected.update({f'{group}/{k}':(batches,batches*v) for group in ('parameters','m','v') for k,v in parameter_width.items()})
    update_cases=parity['comparisons']['update/loss']['comparisons']
    require(batches*4<=update_cases<=batches*8 and update_cases%2==0,'Nonempty optimizer case budget')
    expected.update({f'update/{k}':(update_cases,update_cases) for k in
        ('loss','actor_loss','value_loss','entropy','approx_kl','clip_fraction','max_ratio_error')})
    expected['fused_vs_separate']=(batches*15,batches*543)
    require(set(parity['comparisons'])==set(expected),'Missing/extra parity comparisons')
    for k,(cases,elements) in expected.items():
        require(parity['comparisons'][k]['comparisons']==cases and parity['comparisons'][k]['elements']==elements,'Parity coverage '+k)
    require(profile['cycles']==p['profile_cycles'] and profile['host_profiler_enabled'] is True,'Profile budget')
    require(soak['elapsed_seconds']>=soak['target_seconds']==p['soak_seconds']==7200,'Two-hour soak incomplete')
    for item in (profile,soak):
        require(item['unexpected_compilations']==0 and item['entry_traces']=={'collect':1,'update':1,'cycle':1},'Compilation stability')
        require(item['warmup_compilation_count']>0,'Vacuous compilation counter')
        require(item['games']==item['cycles']*64 and 4*item['games']<=item['decisions']<=8*item['games'],'Fixed-loop accounting')
        require(item['decisions']-item['games']<=item['executed_moves']<=item['decisions'],'Canceled-move accounting')
        require(sum(item['latency_histogram']['counts'])==item['cycles'],'Latency accounting')
        require(item['complete_cycle_seconds']>0 and item['decisions_per_cycle_second']==item['decisions']/item['complete_cycle_seconds'],'Throughput calculation')
    require(soak['full_g2']==soak['full_g3']=='open' and soak['device_memory_headroom_certified'] is False,'Scope/headroom claim')
    telemetry=[json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
    require(len(telemetry)>=119,'Soak telemetry coverage')
    require(all(a['elapsed_seconds']<b['elapsed_seconds'] and a['soak_cycles']<b['soak_cycles'] for a,b in zip(telemetry,telemetry[1:])),'Telemetry monotonicity')
    require(telemetry[-1]['elapsed_seconds']>=7140 and telemetry[-1]['soak_cycles']<=soak['cycles'],'Telemetry endpoint')
    ex=Executables();final=load(run/'soak-final.checkpoint.json',ex);initial=load(run/'soak-start.checkpoint.json',ex)
    require(initial.seed==final.seed==p['soak_seed'],'Soak checkpoint seed')
    require(int(initial.learner.version)==p['warmup_cycles'],'Warmup checkpoint')
    require(int(final.learner.version)-int(initial.learner.version)==soak['cycles'] and final.decisions-initial.decisions==soak['decisions'],'Checkpoint sample accounting')
    require(int(final.learner.step)==soak['final_counters']['optimizer_steps'],'Optimizer endpoint')
    require(final.executed_moves-initial.executed_moves==soak['executed_moves'],'Executed move endpoint')
    require(soak['initial_counters']=={'updates':int(initial.learner.version),'decisions':initial.decisions,'next_game':int(initial.learner.next_game)},'Initial counter linkage')
    require(soak['final_counters']=={'updates':int(final.learner.version),'optimizer_steps':int(final.learner.step),
                                    'decisions':final.decisions,'next_game':int(final.learner.next_game)},'Final counter linkage')
    for row in telemetry:
        require(row['decisions']<=soak['decisions'] and row['updates']==int(initial.learner.version)+row['soak_cycles']
                and row['updates']<=int(final.learner.version) and row['elapsed_seconds']<=soak['elapsed_seconds'],'Telemetry counter linkage')
    boundaries=sorted(run.glob('soak-boundary-*.checkpoint.json'),key=lambda x:int(x.name.split('-')[2].split('.')[0]))
    require(len(boundaries)+1==soak['disk_resumes_with_compared_next_cycle'] and len(boundaries)>=11,'Soak recovery evidence')
    last_version=int(initial.learner.version)
    for path in boundaries:
        boundary=load(path,ex)
        require(boundary.seed==p['soak_seed'] and last_version<int(boundary.learner.version)<=int(final.learner.version),'Boundary identity/order')
        last_version=int(boundary.learner.version)
    for seed in p['parity_seeds']:
        for batch in range(p['parity_batches_per_seed']):
            boundary=load(run/f'parity-{seed}-{batch}.checkpoint.json',ex)
            require(boundary.seed==seed and int(boundary.learner.version)==batch,'Parity boundary checkpoint')
        d=run/f'first-buffer-{seed}';m=read(d/'manifest.json')
        require(m['actor_has_targets'] is False and m['capacity']==512,'Actor export schema')
        for name,h in m['files'].items():require(file_hash(d/name)==h,'Export hash')
        with np.load(d/'actor.npz',allow_pickle=False) as actor, np.load(d/'targets.npz',allow_pickle=False) as target:
            valid=target['valid'];require(valid.shape==(512,) and int(valid.sum())==m['valid'],'Validity export')
            require(set(actor.files)==set(m['actor_fields'])=={'features','mask','actions','old_logp','old_values','game_id','side','turn'},'Actor field boundary')
            require(np.all(actor['features'][valid,1:3]==0) and np.all(actor['features'][~valid]==0),'Actor outcome/padding')
    log=(REVIEW/'npm-test.log').read_text();match=re.search(r'^ℹ tests (\d+)$',log,re.M)
    verification=read(REVIEW/'verification.json')
    current=runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()
    require(verification['status']=='passed' and verification['exit_code']==0 and verification['sources_unchanged'] is True
            and verification['source_files']==current and verification['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),
            'Regression log is not bound to current tested sources')
    require(match is not None,'Node test summary');node=int(match[1])
    require(re.search(rf'^ℹ pass {node}$',log,re.M) and re.search(r'^ℹ fail 0$',log,re.M),'Node failures')
    counts=[int(x) for x in re.findall(r'^Ran (\d+) tests? in ',log,re.M)]
    require(len(counts)==11 and log.count('\nOK\n')==11 and 'FAILED (' not in log,'Python suite completeness')
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md'):
        require((REVIEW/name).is_file(),'Missing review evidence '+name)
    sources=[]
    for directory in ('packages','python','contracts','scripts','tests','docs','tools/mechanics-deps'):
        sources.extend(x for x in (ROOT/directory).rglob('*') if x.is_file() and not {'node_modules','__pycache__'} & set(x.parts))
    sources.extend(ROOT/name for name in ('README.md','package.json','package-lock.json','showdown.pin.json','requirements-contract.txt','requirements-learner.txt','requirements-jax.txt','.node-version','.gitignore'))
    sources=sorted(set(sources));runs=sorted(set([run,*((ROOT/'artifacts/runs').glob('m1-fixed-loop-v*'))]))
    interrupted=ROOT/'artifacts/runs/m1-fixed-loop-v1'
    require(interrupted in runs and read(interrupted/'report.json')['status']=='failed','Missing interrupted-run evidence')
    campaign_history={str(d.relative_to(ROOT)):read(d/'report.json')['status'] for d in runs}
    evidence=sorted(set([x for d in [REVIEW,*runs] for x in d.rglob('*') if x.is_file()]+[ROOT/'artifacts/reports/m1-jax-learner.json']))
    report={'format':'m1-fixed-loop-completion-1','status':'complete-for-declared-scope','full_g2':'open','full_g3':'open',
            'campaign_directory':str(run.relative_to(ROOT)),'campaign_hash':result['content_hash'],'campaign':result,
            'predecessor':p['predecessor'],'final_reservation_unchanged':admitted['prior_gates']['final_reservation_unchanged'],
            'campaign_history':campaign_history,'test_verification_sha256':file_hash(REVIEW/'verification.json'),
            'tests':{'node_passed':node,'python_passed':sum(counts),'python_suite_counts':counts,'log_sha256':file_hash(REVIEW/'npm-test.log')},
            'source_files':{str(x.relative_to(ROOT)):file_hash(x) for x in sources},
            'evidence_files':{str(x.relative_to(ROOT)):file_hash(x) for x in evidence},
            'limitations':['Fixed public 1v1 Rattata duel and small linear policy only; no strength claim.',
                           'Only completed whole-episode updates; no live-game cuts or partial continuation.',
                           'New full-capacity shuffle; numerical parity uses the identical valid sample order.',
                           'Process RSS is not accelerator allocation or certified system headroom.',
                           'Known nominal-versus-finite-word Showdown sampler distinction remains.',
                           'Inherited FP32 inverse-CDF action sampling has finite-grid tail precision limits; nominal tiny action probabilities are not exact finite-grid selection masses.',
                           'Final evaluation unopened; CPU only; Prime unnecessary.'],
            'next':'Decide action-tail sampling precision/floor and G2 coverage/chance-law contracts; then preregister frozen-opponent policy evaluation before G3 promotion.'}
    report['content_hash']=digest(report);content=wire(report)+b'\n'
    archive=ROOT/'artifacts/manifests'/('m1-fixed-loop-'+report['content_hash'])
    for group,files in [('source',sources),('evidence',evidence)]:
        for path in files:
            rel=str(path.relative_to(ROOT));dest=archive/group/rel;write_once(dest,path.read_bytes())
            require(file_hash(dest)==report[group+'_files'][rel],'Archive hash')
    write_once(archive/'completion.json',content);write_once(ROOT/'artifacts/reports/m1-fixed-loop.json',content)
    print(json.dumps({'report':str(ROOT/'artifacts/reports/m1-fixed-loop.json'),'content_hash':report['content_hash'],
                      'tests':report['tests'],'soak_seconds':soak['elapsed_seconds'],'soak_decisions':soak['decisions'],
                      'sources':len(sources),'evidence':len(evidence)},sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');main(parser.parse_args().run)
