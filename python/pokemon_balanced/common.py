"""Admit the immutable F15 control and register bounded F16 conclusions."""
import hashlib
import numpy as np
from pokemon_full_train.storage import ROOT,canonical,digest,load_checkpoint,runtime
from pokemon_imitation.common import read,initialize,team_keys,admit as admit_f14
from pokemon_transfer.common import retention
from pokemon_full_learning.statistics import compare
PLAN=ROOT/'contracts/balanced-imitation-v1/plan.json'

def validate_plan(p):
    if p!=read(PLAN):raise ValueError('Unregistered F16 plan')
    expected={'seeds':[7103,8209,9311],'bc_steps':2048,'teacher_quota':[32,16,8,8],'old_rows':64,'probe_games':128,'evaluation_pairs_per_opponent':128,'workers':8}
    if any(p[k]!=v for k,v in expected.items()) or any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used']):raise ValueError('F16 fixed scope/budget')

def admit(plan):
    path=ROOT/plan['predecessor_receipt'];r=read(path);body=dict(r);h=body.pop('content_hash')
    if digest(path)!=plan['predecessor_sha256'] or hashlib.sha256(canonical(body)).hexdigest()!=h:raise ValueError('F15 receipt mismatch')
    if path.read_bytes()!=(ROOT/'artifacts/manifests'/('f15-full-imitation-'+h)/'receipt.json').read_bytes():raise ValueError('F15 archive mismatch')
    for rel,sha in r['source_files'].items():
        if digest(ROOT/rel)!=sha or digest(ROOT/r['source_snapshot_root']/rel)!=sha:raise ValueError('F15 source drift '+rel)
    for rel,sha in r['evidence_files'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('F15 evidence drift '+rel)
    old,oracle=admit_f14(read(ROOT/'contracts/full-imitation-v1/plan.json'))
    run=ROOT/plan['predecessor_run']
    if read(run/'inputs/manifest.json')['runtime']!=runtime():raise ValueError('F15/F16 runtime mismatch')
    with np.load(run/'teacher/dataset.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    fit_games=[read(run/'teacher'/f'game-{i:04d}.json') for i in range(512)]
    probe_games=[read(run/'probe'/f'game-{i:04d}.json') for i in range(128)]
    past_dev=team_keys(probe_games)
    for seed in read(run/'report.json')['plan']['seeds']:
        for endpoint in read(run/'report.json')['plan']['endpoints']:
            panel=read(run/str(seed)/endpoint/'evaluation.json')['games']
            past_dev.update(team_keys([g for games in panel.values() for g in games]))
    return old,oracle,data,team_keys(fit_games),past_dev

def uniform_control(seed,plan):
    run=ROOT/plan['predecessor_run'];h=digest(run/'inputs/manifest.json')
    p,opt,meta=load_checkpoint(run/str(seed)/'imitation',h)
    if meta['seed']!=seed or meta['cohorts']!=2048 or meta['cursors']!={'bc':2048} or int(opt[2])!=2048:raise ValueError('Uniform control checkpoint identity')
    return p,opt

def analysis(panels,tacticals,probes,plan):
    comparisons={name:{o:compare(panels[a][o],panels[b][o],plan['statistics']) for o in plan['opponents']} for name,a,b in [('uniform_vs_initial','initial','uniform'),('balanced_vs_initial','initial','balanced'),('balanced_vs_uniform','uniform','balanced')]}
    retained={e:retention(tacticals['initial'],tacticals[e],plan['tactical_thresholds']) for e in ['uniform','balanced']}
    u,b=probes['uniform']['families'],probes['balanced']['families'];t=plan['coverage_thresholds']
    coverage={'tera_accuracy':b['tera_move']['game_accuracy']>=t['tera_min_accuracy'],'tera_gain':b['tera_move']['game_accuracy']-u['tera_move']['game_accuracy']>=t['tera_min_accuracy_gain'],'tera_mass':b['tera_move']['game_target_mass']>=t['tera_min_target_mass'],'voluntary_switch_accuracy':b['voluntary_switch']['game_accuracy']>=t['voluntary_switch_min_accuracy'],'normal_move_preserved':u['normal_move']['game_accuracy']-b['normal_move']['game_accuracy']<=t['maximum_other_accuracy_drop'],'forced_switch_preserved':u['forced_switch']['game_accuracy']-b['forced_switch']['game_accuracy']<=t['maximum_other_accuracy_drop'],'known_contexts_retained':retained['balanced']['passed']}
    def strength(name):
        cs=comparisons[name];q=cs['tactical'];s=plan['statistics']
        return q['gate']=='improvement' and q['cutoff_aware_point_interval'][0]>=s['practical_improvement'] and all(c['gate']!='regression' and c['final']['cutoffs']/c['final']['games']<=s['max_final_cutoff_rate'] for c in cs.values())
    return {'comparisons':comparisons,'retention':retained,'coverage_checks':coverage,'coverage_screen':all(coverage.values()),'strength_gate':strength('balanced_vs_initial') and strength('balanced_vs_uniform') and retained['balanced']['passed']}
