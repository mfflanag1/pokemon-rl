"""F13 admission, fixed-bank role handling and terminal-only diagnostics."""
import hashlib
from collections import Counter
import numpy as np
from pokemon_full_train.storage import ROOT,canonical,digest,load_checkpoint
from pokemon_curriculum.common import read,admit as admit_f11,initialize as f10_initialize
from pokemon_full_train.learning import derived_seed,prepare_batch
from pokemon_fit.data import fingerprint
from .returns import targets

def admit(plan):
    path=ROOT/plan['predecessor_receipt'];r=read(path);body=dict(r);h=body.pop('content_hash')
    if digest(path)!=plan['predecessor_sha256'] or hashlib.sha256(canonical(body)).hexdigest()!=h:raise ValueError('F12 receipt changed')
    if path.read_bytes()!=(ROOT/'artifacts/manifests'/('f12-curriculum-'+h)/'receipt.json').read_bytes():raise ValueError('F12 receipt archive changed')
    for rel,sha in r['source_files'].items():
        if digest(ROOT/rel)!=sha or digest(ROOT/r['source_snapshot_root']/rel)!=sha:raise ValueError('F12 source changed: '+rel)
    for rel,sha in r['evidence_files'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('F12 evidence changed: '+rel)
    _,_,supervised=admit_f11(read(ROOT/'contracts/curriculum-learning-v1/plan.json'))
    return supervised

def initialize(seed):return f10_initialize(seed,read(ROOT/'contracts/curriculum-learning-v1/plan.json'))

def endpoint(seed,name,plan):
    if name=='initial':return initialize(seed)[0]
    return load_checkpoint(ROOT/plan['f12_run']/str(seed)/name/'final',plan['f12_input_hash'])[0]

def specs(bank,plan,seed,first,count,role):
    admitted=[s for s in bank if s['origin']['role']==role]
    if role not in ['fit','probe'] or not admitted:raise ValueError('Unknown or empty bank role')
    result=[]
    for ordinal in range(first,first+count):
        if role=='fit':
            family=plan['families'][ordinal%2];available=sorted((s for s in admitted if s['families']==[family]),key=lambda s:s['content_hash'])
            chosen=available[int(np.random.default_rng(derived_seed('f13-bank',seed,ordinal)).integers(len(available)))]
        else:
            ordered=sorted(admitted,key=lambda s:(s['families'][0],s['origin']['teacher_episode']))
            chosen=ordered[ordinal//plan['evaluation_repeats']]
        result.append(dict(seed=derived_seed('f13-suffix',seed,role),episode=ordinal,split='train',swap=False,
            max_turns=plan['max_turns'],opponent='tactical',learner_side=chosen['learner_side'],
            actor_seed=derived_seed('f13-actor',seed,role),scenario_hash=chosen['content_hash']))
    return result

def batch(rows,lam):
    original=prepare_batch(rows);independent=targets(rows,.95)
    error=float(np.max(np.abs(original['returns']-independent['returns'])))
    if error>2e-6:raise ValueError('Independent lambda-return mismatch')
    if lam==1.:
        alternative=targets(rows,1.)
        original.update(returns=alternative['returns'].astype(np.float32),advantages=alternative['advantages'].astype(np.float32))
    elif lam!=.95:raise ValueError('Unregistered target lambda')
    return original,error

def diagnostics(rows):
    a=targets(rows,.95);b=targets(rows,1.);sealed=prepare_batch(rows)
    error=float(np.max(np.abs(a['returns']-sealed['returns'])))
    if error>2e-6:raise ValueError('Return parity failed')
    terminal=b['terminal_chain'];loss=terminal&(b['returns']==0);switch=np.asarray([r['voluntary_switch'] for r in rows])
    groups={}
    for label,mask in [('all_terminal',terminal),('terminal_loss',loss),('loss_switch',loss&switch),('loss_other',loss&~switch),('cutoff',~terminal)]:
        groups[label]={'rows':int(mask.sum()),'mean_value':float(np.mean([rows[i]['value'] for i in np.flatnonzero(mask)])) if mask.any() else None,
            'mean_gae_return':float(a['returns'][mask].mean()) if mask.any() else None,
            'mean_lambda1_return':float(b['returns'][mask].mean()) if mask.any() else None,
            'positive_gae_advantages':int(np.sum(a['advantages'][mask]>0)),
            'positive_lambda1_advantages':int(np.sum(b['advantages'][mask]>0)),
            'mean_distance_to_end':float(a['distance_to_end'][mask].mean()) if mask.any() else None}
    return {'rows':len(rows),'max_target_parity_error':error,'groups':groups,
            'sealed_batch_hash':fingerprint(sealed),'independent_targets_hash':fingerprint(a),
            'interpretation':'Terminal loss advantages under lambda1 cannot be positive for bounded nonnegative values; positive GAE advantages are bootstrap estimates, not automatically implementation errors. Cutoff lambda1 targets remain bootstrapped.'}

def outcome_summary(games):
    terminal=[g for g in games if g['outcome']['kind']=='terminal']
    return {'games':len(games),'terminals':len(terminal),'cutoffs':len(games)-len(terminal),
        'wins':sum(g['outcome']['rewards'][g['learner_side']]==1 for g in terminal),
        'ties':sum(g['outcome']['rewards'][g['learner_side']]==.5 for g in terminal),
        'decisions':sum(g['decisions'] for g in games),'learner_decisions':sum(g['learner_decisions'] for g in games)}
