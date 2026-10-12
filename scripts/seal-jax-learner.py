"""Seal checked JAX learner/collector evidence; keep full G2/G3 explicitly open."""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require
from pokemon_jax_learner.checkpoint import provenance,load,state
from pokemon_jax_learner.campaign import admission_checks,PREDECESSOR
from pokemon_jax_learner.collector import policy_hash

REVIEW=ROOT/'artifacts/reviews/m1-jax-learner'


def read(path):return json.loads(path.read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Refusing changed sealed artifact: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(data)


def main(run):
    run=Path(run).resolve();require(run.is_relative_to(ROOT/'artifacts/runs'),'Campaign directory')
    log=(REVIEW/'npm-test.log').read_text();match=re.search(r'^ℹ tests (\d+)$',log,re.M)
    require(match is not None,'Node test summary missing');node=int(match[1])
    require(re.search(rf'^ℹ pass {node}$',log,re.M) and re.search(r'^ℹ fail 0$',log,re.M),'Node tests failed')
    suites=[int(x) for x in re.findall(r'^Ran (\d+) tests? in ',log,re.M)]
    require(len(suites)==10 and log.count('\nOK\n')==10 and 'FAILED (' not in log,'Python suites incomplete')
    tests={'node_passed':node,'python_passed':sum(suites),'python_suite_counts':suites,'log_sha256':file_hash(REVIEW/'npm-test.log')}
    result=read(run/'report.json');admission=read(run/'admission.json');plan=read(ROOT/'contracts/jax-learner-v1/plan.json')
    require(result['content_hash']==digest({k:v for k,v in result.items() if k!='content_hash'}),'Campaign checksum')
    require(result['status']=='passed' and result['full_g2']==result['g3']=='open','Promotion status')
    require(result['admission_hash']==digest(admission) and admission['plan']==plan and admission['provenance']==provenance(),'Admission/source/runtime drift')
    require(admission['predecessors']==admission_checks(),'Predecessor/final-data gate')
    parity=result['parity'];n=len(plan['parity_seeds'])*plan['parity_cases_per_seed']
    require(parity['ppo_cases']==parity['gae_cases']==n and parity['status']=='passed','Parity cases')
    require(parity['adam_steps']==len(plan['parity_seeds'])*plan['adam_steps'] and parity['table_update_steps']==len(plan['parity_seeds'])*plan['table_update_steps'],'Update parity counts')
    require(parity['atol']==plan['atol'] and parity['rtol']==plan['rtol'],'Tolerance drift')
    for row in parity['comparisons'].values():require(row['comparisons']>0 and row['elements']>0 and 0<=row['max_fraction_of_tolerance']<=1,'Parity tolerance gate')
    toy=read(ROOT/plan['canary_plan']);canaries=result['canaries']
    require(canaries['status']=='passed' and canaries['plan_hash']==digest(toy),'Canary plan link')
    expected={(task,seed) for task in ('bandit','delayed') for seed in toy['seeds']}
    require(len(canaries['results'])==len(expected) and {(r['task'],r['seed']) for r in canaries['results']}==expected,'Canary completeness')
    for row in canaries['results']:
        task,seed=row['task'],row['seed'];detail=read(run/f'canary-{task}-{seed}.json')
        require(detail['result']==row and len(detail['trace'])==toy[task]['max_updates'],'Canary trace linkage')
        require(row['decisions']==toy[task]['sample_budget'] and min(row['probabilities'])>=toy[task]['required_probability'],'Canary learning/budget')
        require(detail['trace'][-1]['probabilities']==row['probabilities'],'Canary endpoint')
    collector=result['collector'];require(collector['status']=='passed' and collector['all_buffers_observation_only'] is True,'Collector gate')
    require([r['seed'] for r in collector['seeds']]==plan['collector_seeds'],'Collector seeds')
    games=decisions=resumes=0;wall=0.
    for row in collector['seeds']:
        seed=row['seed'];directory=run/f'collector-{seed}';require(read(directory/'report.json')==row,'Collector report linkage')
        require(row['updates']==plan['collector_updates'] and row['games']==plan['collector_updates']*plan['games_per_collection'],'Collector budget')
        require(row['resume_compared_updates']==row['disk_resumes_with_compared_update']==plan['collector_updates']-1,'Resume evidence')
        trace=row['trace'];require(len(trace)==row['updates'] and [r['update'] for r in trace]==list(range(1,len(trace)+1)),'Collection trace order')
        require(sum(r['decisions'] for r in trace)==row['decisions'] and sum(r['terminal_reward_sum'] for r in trace)==row['terminal_score_pair_sum']==row['games'],'Measured counts/scores')
        for r in trace:require(r['actor_boundary_check'] is True and r['behavior_max_ratio_error']<=1e-6,'Frozen actor check')
        trainer=load(directory/'final.checkpoint.json',expected_plan=plan)
        require(trainer.games==row['games'] and trainer.decisions==row['decisions'] and policy_hash(trainer.parameters)==row['final_policy_hash'],'Final checkpoint linkage')
        for update in range(1,plan['collector_updates']):
            name='boundary.checkpoint.json' if update==1 else f'boundary-{update}.checkpoint.json'
            boundary=load(directory/name,expected_plan=plan);require(boundary.updates==update,'Disk boundary completeness')
        manifest=read(directory/'first-buffer/manifest.json');require(manifest['actor_has_targets'] is False and manifest['collection_hash']==trace[0]['collection_hash'],'Buffer export linkage')
        for name,h in manifest['files'].items():require(file_hash(directory/'first-buffer'/name)==h,'Buffer file hash')
        with np.load(directory/'first-buffer/actor.npz',allow_pickle=False) as actor:
            require(set(actor.files)==set(manifest['actor_fields'])=={'features','mask','actions','old_logp','old_values','game_id','side','turn'},'Actor export fields')
            require(np.all(actor['features'][:,1:3]==0) and np.all(actor['mask'].any(axis=1)),'Actor terminal/target contamination')
        games+=row['games'];decisions+=row['decisions'];resumes+=row['disk_resumes_with_compared_update'];wall+=sum(r['complete_cycle_seconds'] for r in trace)
    require((REVIEW/'claude-review.md').exists() and (REVIEW/'review-dispositions.md').exists(),'Independent review evidence')
    sources=[]
    for directory in ('packages','python','contracts','scripts','tests','docs','tools/mechanics-deps'):
        sources.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and not {'node_modules','__pycache__'} & set(p.parts))
    sources.extend(ROOT/name for name in ('README.md','package.json','package-lock.json','showdown.pin.json','requirements-contract.txt','requirements-learner.txt','requirements-jax.txt','.node-version','.gitignore'))
    sources=sorted(set(sources));runs=sorted((ROOT/'artifacts/runs').glob('m1-jax-learner-v*'))
    evidence=sorted(set([p for d in [REVIEW,*runs] for p in d.rglob('*') if p.is_file()]+[ROOT/'artifacts/reports/m1-g1.json',ROOT/'artifacts/reports/m1-g2.json']))
    source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in sources};evidence_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in evidence}
    report={'format':'m1-jax-learner-completion-1','status':'complete-for-declared-scope','full_g2':'open','g3':'open',
        'campaign_directory':str(run.relative_to(ROOT)),'campaign_hash':result['content_hash'],'campaign':result,'runtime':admission['provenance']['runtime'],
        'tests':tests,'summary':{'ppo_cases':n,'gae_cases':n,'canaries':len(expected),'collector_games':games,'collector_decisions':decisions,
                               'disk_resumes_with_compared_update':resumes,'complete_cycle_seconds':wall,'decisions_per_complete_cycle_second':decisions/wall},
        'previous_g2_hash':PREDECESSOR,'final_reservation_unchanged':True,'source_files':source_hashes,'evidence_files':evidence_hashes,
        'review_dispositions':'artifacts/reviews/m1-jax-learner/review-dispositions.md',
        'limitations':['Numerical tolerance parity, not bit-identical NumPy/JAX updates.',
            'Synthetic canaries use the unchanged CPU toy collector with JAX autodiff updates.',
            'Tiny linear fixed-duel correctness model; no competitive strength, full G2 or G3 claim.',
            'Only whole-episode completed-update checkpoints; no live/cut/partial continuation.',
            'CPU complete-cycle times include shape compilation and exclude audit replay and checkpoint/export I/O.',
            'Known nominal-vs-finite-word Showdown chance distinction remains; final evaluation unused; no Prime/GPU.'],
        'next':'Fixed-shape complete collect/update loop and CPU soak/resume profiling; separately close G2 coverage before policy promotion.'}
    report['content_hash']=digest(report);content=wire(report)+b'\n';archive=ROOT/'artifacts/manifests'/('m1-jax-learner-'+report['content_hash'])
    for group,files,hashes in [('source',sources,source_hashes),('evidence',evidence,evidence_hashes)]:
        for p in files:
            rel=str(p.relative_to(ROOT));target=archive/group/rel;write_once(target,p.read_bytes());require(file_hash(target)==hashes[rel],'Archive hash mismatch')
    write_once(archive/'completion.json',content);write_once(ROOT/'artifacts/reports/m1-jax-learner.json',content)
    print(json.dumps({'report':str(ROOT/'artifacts/reports/m1-jax-learner.json'),'archive':str(archive),'content_hash':report['content_hash'],'tests':tests,'summary':report['summary'],'sources':len(sources),'evidence':len(evidence)},sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');main(parser.parse_args().run)
