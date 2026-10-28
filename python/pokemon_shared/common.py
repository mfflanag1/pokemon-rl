"""Frozen corpus admission and architecture-bound strict checkpoint recovery."""
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from pokemon_full_train.storage import ROOT,digest,save_checkpoint
from pokemon_diversity.common import read,receipt,team_keys
from pokemon_learning_curve.common import admit as admit_f19,probe_roles
from .learning import check_state,shapes
PLAN=ROOT/'contracts/shared-policy-v1/plan.json'

def validate_plan(p):
    fixed={'seeds':[7103,8209,9311],'arms':['flat','shared'],'endpoints':[0,2048,8192],'bc_steps':8192,'workers':8,'max_turns':120,'teacher_quota':[32,16,8,8],'old_rows':64,'training_games':2048,'training_rows':106149,'new_training_games':0,'probe_games':256,'probe_seed':1025262001,'expected_updates':49152,'expected_presentations':6291456,'expected_new_games':256,'checkpoint_comparisons':30,'checkpoint_recoveries':12,'checkpoint_steps_per_arm':[0,2047,2048,8191,8192],'recovered_update_indices_per_arm':[2047,8191],'parameter_counts':{'flat':60431,'shared':60479},'policy_parameter_counts':{'flat':60366,'shared':60318}}
    if p!=read(PLAN) or any(p.get(k)!=v for k,v in fixed.items()) or any(p[k] for k in ['prime_used','public_server_used','final_reserved_data_used']):raise ValueError('F20 fixed scope/budget mismatch')

def admit(plan):
    receipt(ROOT/plan['predecessor_receipt'],plan['predecessor_sha256'],'f19-learning-curve')
    old,oracle,data,rows,train,past=admit_f19(read(ROOT/'contracts/learning-curve-v1/plan.json'))
    past.update(team_keys([read(ROOT/plan['predecessor_run']/'probe'/f'game-{i:04d}.json') for i in range(256)]))
    if train & past:raise ValueError('Training/development role overlap')
    return old,oracle,data,rows,train,past

def metadata(h,seed,arm,step):return {'input_hash':h,'seed':seed,'architecture':arm,'cohorts':step,'cursors':{'bc':step,'arm':arm},'optimizer_step':step}

def save(directory,p,opt,h,seed,arm,step):
    check_state(p,opt,arm)
    if int(opt[2])!=step:raise ValueError('Save counter mismatch')
    return save_checkpoint(directory,p,opt,metadata(h,seed,arm,step))

def load(directory,h,seed,arm,step):
    pointer=read(directory/'latest.json')
    if set(pointer)!={'file','sha256'} or Path(pointer['file']).name!=pointer['file']:raise ValueError('Checkpoint pointer schema')
    path=directory/pointer['file']
    if digest(path)!=pointer['sha256']:raise ValueError('Checkpoint hash')
    with np.load(path,allow_pickle=False) as z:
        expected={'metadata'}|{prefix+'__'+key for prefix in ['p','m','v'] for key in shapes(arm)}
        if set(z.files)!=expected or z['metadata'].dtype!=np.uint8 or z['metadata'].ndim!=1:raise ValueError('Checkpoint inventory')
        m=json.loads(z['metadata'].tobytes())
        if m!=metadata(h,seed,arm,step) or any(type(m[k]) is not int for k in ['seed','cohorts','optimizer_step']):raise ValueError('Checkpoint identity')
        states=[{k:z[prefix+'__'+k] for k in shapes(arm)} for prefix in ['p','m','v']]
        check_state(states[0],(states[1],states[2],np.asarray(step,np.int32)),arm)
        states=[{k:jnp.asarray(a) for k,a in s.items()} for s in states]
    return states[0],(states[1],states[2],jnp.int32(step))

def same_state(p,opt,q,qo):
    for a,b in [(p,q),(opt[0],qo[0]),(opt[1],qo[1])]:
        if set(a)!=set(b) or any(not np.array_equal(np.asarray(a[k]),np.asarray(b[k])) for k in a):raise ValueError('Checkpoint state diverged')
    if int(opt[2])!=int(qo[2]):raise ValueError('Checkpoint step diverged')

def gate(arms,plan):
    f=arms['flat']['endpoints']['8192'];s=arms['shared']['endpoints']['8192'];a=f['probe']['families'];b=s['probe']['families'];t=plan['coverage_thresholds']
    checks={'voluntary_accuracy':b['voluntary_switch']['game_accuracy']>=t['voluntary_min_accuracy'],'voluntary_gain':b['voluntary_switch']['game_accuracy']-a['voluntary_switch']['game_accuracy']>=t['voluntary_min_gain'],**{k+'_preserved':a[k]['game_accuracy']-b[k]['game_accuracy']<=t['maximum_other_accuracy_drop'] for k in ['normal_move','tera_move','forced_switch']},'known_contexts_mastered':s['mastery']['passed']}
    return {'checks':checks,'passed':all(checks.values())}
