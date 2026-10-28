"""Immutable checkpoint admission, paired schedules and explicit development gates."""
import json
from pathlib import Path
import numpy as np
from pokemon_full_train.storage import ROOT,digest,runtime
from pokemon_full.learner import parameter_hash
from pokemon_full_learning.statistics import compare
from pokemon_full_learning.campaign import evaluation_specs,check_games
from pokemon_diversity.common import read,receipt,team_keys,load_strict_checkpoint
from pokemon_shared.common import admit as f20_inputs,load as load_candidate
from pokemon_shared.learning import check_state
PLAN=ROOT/'contracts/shared-battle-v1/plan.json'
CONTRASTS=[['shared_vs_flat','flat','shared'],['shared_vs_reference','reference','shared'],['flat_vs_reference','reference','flat']]

def validate_plan(p):
    fixed={'seeds':[7103,8209,9311],'arms':['flat','shared','reference'],
        'opponents':['random','base_power','tactical'],'workers':8,'max_turns':120,
        'evaluation_seed':1025262101,'evaluation_pairs_per_opponent':128,
        'evaluation_chunk_pairs':4,'expected_games':6912,'optimizer_updates':0,
        'checkpoint_loads':9,'candidate_step':8192,'reference_checkpoint':'rehearsal/final',
        'reference_cohorts':128,'reference_optimizer_step':1024,'comparisons':CONTRASTS}
    if p!=read(PLAN) or any(p.get(k)!=v for k,v in fixed.items()):
        raise ValueError('F21 fixed schedule/budget mismatch')
    if any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used']):
        raise ValueError('Forbidden evaluation scope')
    if p['statistics']!={'method':'paired-cluster-percentile-bootstrap','family_cells':27,
        'alpha':.05,'resamples':20000,'seed':1025262102,'practical_improvement':.05,
        'max_final_cutoff_rate':.05}:
        raise ValueError('Changed statistical family')

def admit(plan):
    r20=receipt(ROOT/plan['predecessor_receipt'],plan['predecessor_sha256'],'f20-shared-policy')
    r10=receipt(ROOT/plan['reference_receipt'],plan['reference_sha256'],'f10-long-transfer')
    if r20['status']!='passed' or r10['status']!='completed':
        raise ValueError('Incomplete predecessors')
    _,oracle,_,_,train,past=f20_inputs(read(ROOT/'contracts/shared-policy-v1/plan.json'))
    base20=ROOT/plan['predecessor_run'];base10=ROOT/plan['reference_run']
    past.update(team_keys([read(base20/'probe'/f'game-{i:04d}.json') for i in range(256)]))
    for seed in plan['seeds']:
        for label in ['initial']+['plain','rehearsal']:
            name='initial-evaluation.json' if label=='initial' else label+'/final-evaluation.json'
            panel=read(base10/str(seed)/name)['games']
            past.update(team_keys([g for games in panel.values() for g in games]))
        for arm in ['plain','rehearsal']:
            for i in range(128):
                train.update(team_keys(read(base10/str(seed)/arm/f'update-{i:03d}.json')['ledger']['games']))
    if train&past:raise ValueError('Predecessor training/development overlap')
    return oracle,train,past

def load_policy(plan,seed,arm):
    if seed not in plan['seeds'] or arm not in plan['arms']:raise ValueError('Unregistered checkpoint')
    architecture='shared' if arm=='shared' else 'flat'
    base=ROOT/plan['reference_run' if arm=='reference' else 'predecessor_run']
    h=digest(base/'inputs/manifest.json')
    if arm=='reference':
        folder=base/str(seed)/plan['reference_checkpoint'];p,opt,meta=load_strict_checkpoint(folder,h)
        if meta['seed']!=seed or meta['cohorts']!=128 or meta['optimizer_step']!=1024 or int(opt[2])!=1024:
            raise ValueError('Wrong reference endpoint')
        expected=read(base/str(seed)/'rehearsal/final-evaluation.json')['parameter_hash']
    else:
        folder=base/str(seed)/arm/'8192';p,opt=load_candidate(folder,h,seed,arm,8192)
        expected=read(base/str(seed)/arm/'8192/summary.json')['parameter_hash']
    check_state(p,opt,architecture)
    if parameter_hash(p)!=expected:raise ValueError('Sealed parameter identity mismatch')
    return p,opt,{'architecture':architecture,'directory':str(folder.relative_to(ROOT)),
        'input_hash':h,'pointer':read(folder/'latest.json'),'parameter_hash':expected,
        'optimizer_step':int(opt[2])}

def role_check(games,train,past):
    keys=team_keys(games)
    if keys&(train|past):raise ValueError('Fresh evaluation exact-team overlap')
    return keys

def check_collection(collection,first,count,opponent):
    games=[g for rows,g in collection]
    if any(rows for rows,_ in collection):raise ValueError('Frozen evaluation produced training rows')
    check_games(games,split='development',first=first,count=count,opponent=opponent)
    for g in games:
        controlled=[r for r in g['trace'] if r[0]==g['learner_side']]
        records=g['actor_trace']
        if len(controlled)!=len(records):raise ValueError('Actor trace inventory mismatch')
        for row,a in zip(controlled,records):
            if [row[1],row[2],row[3],row[4]]!=[a['request_id'],a['action'],a['value'],a['feature_sha256']]:
                raise ValueError('Actor trace linkage mismatch')
            mask=np.asarray(a['mask'])
            if mask.shape!=(14,) or mask.dtype!=np.bool_ or not 0<=a['action']<14 or not mask[a['action']]:
                raise ValueError('Illegal actor trace')
            if not np.isfinite(a['log_probability']) or a['log_probability']>1e-6:
                raise ValueError('Invalid actor likelihood')
        if type(g['unavailable_choices']) is not int or not 0<=g['unavailable_choices']<=g['decisions']:
            raise ValueError('Invalid rejected-choice count')
    return games

def analysis(panels,mastery,plan):
    comparisons={name:{o:compare(panels[a][o],panels[b][o],plan['statistics'])
        for o in plan['opponents']} for name,a,b in plan['comparisons']}
    cs=comparisons['shared_vs_flat'];t=cs['tactical'];s=plan['statistics']
    checks={'tactical_point_gain':t['cutoff_aware_point_interval'][0]>=s['practical_improvement'],
        'tactical_adjusted_lower_positive':t['approximate_familywise_bootstrap_interval'][0]>0,
        'no_detected_other_regression':all(cs[o]['gate']!='regression' for o in ['random','base_power']),
        'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=s['max_final_cutoff_rate'] for c in cs.values())}
    return {'comparisons':comparisons,'battle_checks':checks,'battle_benefit':all(checks.values()),
        'familiar_mastery':mastery,'promotion_authorized':False}
