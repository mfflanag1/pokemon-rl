"""F30 duration budgets and exact continuation from immutable F29 states."""
import numpy as np
from pokemon_full_train.storage import ROOT, digest
from pokemon_full_train.learning import PPO
from pokemon_full.learner import parameter_hash
from pokemon_fit.data import fingerprint
from pokemon_nonlinear_policy.common import (read, receipt, team_keys, save, load,
    metadata, same_state, evaluation_specs, diagnostic_specs, check_collection,
    selfplay_value, tactical, shapes, check_state, load_policy)
from pokemon_full_learning.statistics import compare
PLAN=ROOT/'contracts/nonlinear-continuation-v1/plan.json'
BUDGETS={a:dict(start_cohort=256,cohorts=1024,additional_cohorts=768,selected_rows=1024,
    optimizer_steps_per_cohort=8,maximum_games_per_stream=64,endpoints=[256,512,1024],
    checkpoint_cohorts=[256,511,512,1023,1024],recovery_indices=[511,1023])
    for a in ['control','nonlinear']}
EXPECTED=dict(cohorts=4608,optimizer_steps=36864,selected_ppo_rows=4718592,
    ppo_row_presentations=9437184,evaluation_games=13824,selfplay_diagnostic_games=576,
    checkpoint_comparisons=30,original_next_update_recoveries=12)
def budget(arm):
    if arm not in BUDGETS:raise ValueError('Unknown continuation arm')
    return BUDGETS[arm]
def validate_plan(p):
    predecessor=read(ROOT/'contracts/nonlinear-policy-v1/plan.json')
    if p!=read(PLAN):raise ValueError('Fixed F30 plan')
    fixed=dict(seeds=[7103,8209,9311],arms=['control','nonlinear'],arm_budgets=BUDGETS,
        start_cohort=256,final_cohort=1024,expected_counts=EXPECTED,evaluation_seed=1025263001,
        selfplay_evaluation_seed=1025263002,ppo=PPO)
    if any(p.get(k)!=v for k,v in fixed.items()):raise ValueError('Fixed continuation scope/budget')
    for k in ['workers','max_turns','training_chunk_games','training_seed_namespace',
            'optimizer_seed_namespace','opponents','evaluation_pairs_per_opponent',
            'evaluation_chunk_pairs','selfplay_pairs','features','extra_features',
            'residual_width','parameter_count','added_policy_parameters','catalog',
            'catalog_sha256','primary_opponent','comparisons']:
        if p[k]!=predecessor[k]:raise ValueError('Changed continuation rule: '+k)
    if not all(p[k] is True for k in ['no_early_stopping','no_checkpoint_selection','no_budget_extension']):raise ValueError('Fixed stopping/selection rules')
    if p['statistics']!={**predecessor['statistics'],'seed':1025263003}:raise ValueError('Statistical family')
    if any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used','promotion_authorized']):raise ValueError('Forbidden scope')

def admit(plan):
    from pokemon_nonlinear_policy.common import admit as f29_admit
    _,oracle,train,past=f29_admit(read(ROOT/'contracts/nonlinear-policy-v1/plan.json'))
    prior=receipt(ROOT/plan['f29_receipt'],plan['f29_sha256'],'f29-nonlinear-policy')
    if prior['status']!='passed' or prior['counts']['cohorts']!=1536:raise ValueError('F29 completion')
    run=ROOT/prior['run'];paths=sorted(run.glob('*/*/update-*.json'))
    if len(paths)!=1536:raise ValueError('F29 training history incomplete')
    for path in paths:train.update(team_keys(read(path)['ledger']['games']))
    for pattern in ['*/*/*/games/*.json','*/*/*/selfplay/*.json']:
        for path in sorted(run.glob(pattern)):past.update(team_keys(read(path)['games']))
    preflight=ROOT/'artifacts/reviews/f30-nonlinear-continuation/preflight'
    if (preflight/'report.json').exists():
        evidence=sorted(preflight.glob('*/*/result.json'))
        if len(evidence)!=6 or read(preflight/'report.json')['status']!='passed':raise ValueError('Preflight history incomplete')
        for path in evidence:train.update(team_keys(read(path)['ledger']['games']))
    if train & past:raise ValueError('Historical role collision')
    return prior,oracle,train,past

def initial(previous,seed,arm):
    budget(arm);plan=read(PLAN)
    if seed not in plan['seeds']:raise ValueError('Unknown continuation seed')
    path=ROOT/plan['f29_receipt'];prior=read(path)
    if digest(path)!=plan['f29_sha256'] or (previous is not None and previous!=prior):raise ValueError('F29 source receipt binding')
    run=ROOT/prior['run'];report=read(run/'report.json')
    if report['status']!='passed' or digest(run/'report.json')!=prior['report_sha256']:raise ValueError('F29 source report binding')
    a=report['seeds'][str(seed)]['arms'][arm];cursor=a['final_next_episode'];h=digest(run/'inputs/manifest.json')
    if h!=prior['input_hash'] or a['optimizer_steps']!=2048:raise ValueError('F29 input/update binding')
    folder=run/str(seed)/arm/'checkpoints/256'
    p,opt=load(folder,h,seed,arm,256,cursor)
    if parameter_hash(p)!=a['endpoints']['262144']['parameter_hash']:raise ValueError('F29 final parameter binding')
    if int(opt[2])!=2048:raise ValueError('Adam continuation counter')
    identity=dict(predecessor_receipt_sha256=plan['f29_sha256'],predecessor_input_hash=h,
        checkpoint_pointer_sha256=digest(folder/'latest.json'),checkpoint_sha256=read(folder/'latest.json')['sha256'],
        seed=seed,arm=arm,cohort=256,next_episode=cursor,optimizer_step=int(opt[2]),
        parameter_hash=parameter_hash(p),adam_m_sha256=fingerprint(opt[0]),adam_v_sha256=fingerprint(opt[1]),
        optimizer_reset=False,source_checkpoint=str(folder.relative_to(ROOT)))
    return p,opt,identity

def analysis(panels, plan):
    pairs = [('control_learning',panels['control']['262144'],panels['control']['1048576']),
        ('nonlinear_learning',panels['nonlinear']['262144'],panels['nonlinear']['1048576']),
        ('nonlinear_vs_control',panels['control']['1048576'],panels['nonlinear']['1048576'])]
    comparisons = {name:{o:compare(a[o],b[o],plan['statistics']) for o in plan['opponents']} for name,a,b in pairs}
    def gate(cells, primary='tactical'):
        p = cells[primary]; s = plan['statistics']
        checks = {'point_gain':p['cutoff_aware_point_interval'][0]>=s['practical_improvement'],
            'adjusted_lower_positive':p['approximate_familywise_bootstrap_interval'][0]>0,
            'no_detected_regression':all(cells[o]['gate']!='regression' for o in ['random','base_power']),
            'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=s['max_final_cutoff_rate'] for c in cells.values())}
        return {'checks':checks,'passed':all(checks.values())}
    return {'comparisons':comparisons,
        'learning':{a:gate(comparisons[a+'_learning']) for a in plan['arms']},
        'feature_benefit':gate(comparisons['nonlinear_vs_control']),
        'base_power_learning':{a:gate(comparisons[a+'_learning'],'base_power') for a in plan['arms']},
        'base_power_benefit':gate(comparisons['nonlinear_vs_control'],'base_power')}
