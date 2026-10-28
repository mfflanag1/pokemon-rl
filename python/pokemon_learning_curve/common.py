"""Admit the frozen F18 corpus and resume its exact expanded optimizer state."""
import numpy as np
from pokemon_full_train.storage import ROOT,digest
from pokemon_diversity.common import read,receipt,admit as admit_f18,load_strict_checkpoint,team_keys
from pokemon_diversity.collection import validate_metadata
from pokemon_balanced.sampling import classify
PLAN=ROOT/'contracts/learning-curve-v1/plan.json'

def validate_plan(p):
    fixed={'seeds':[7103,8209,9311],'endpoints':[2048,4096,8192],'start_step':2048,'final_step':8192,
        'workers':8,'max_turns':120,'teacher_quota':[32,16,8,8],'old_rows':64,'training_games':2048,
        'new_training_games':0,'probe_games':256,'probe_seed':1025261901,'expected_new_updates':18432,
        'expected_new_presentations':2359296,'expected_accumulated_updates':24576,'expected_new_games':256,
        'checkpoint_comparisons':15,'checkpoint_recoveries':6,
        'checkpoint_steps_per_seed':[2048,4095,4096,8191,8192],
        'recovered_update_indices_per_seed':[4095,8191]}
    if p!=read(PLAN) or any(p.get(k)!=v for k,v in fixed.items()) or any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used']):
        raise ValueError('F19 fixed scope/budget mismatch')

def admit(plan):
    receipt(ROOT/plan['predecessor_receipt'],plan['predecessor_sha256'],'f18-data-coverage')
    old,oracle,original,original_rows,train_keys,past=admit_f18(read(ROOT/'contracts/data-coverage-v1/plan.json'))
    base=ROOT/plan['predecessor_run']
    with np.load(base/'inputs/expanded-teacher.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    classify(data)
    if len(data['targets'])!=106149 or set(np.unique(data['game_index']))!=set(range(2048)):
        raise ValueError('Frozen expanded corpus inventory mismatch')
    for name,expected in [('original-teacher',original),('old-supervised',old)]:
        with np.load(base/'inputs'/(name+'.npz'),allow_pickle=False) as z:
            if set(z.files)!=set(expected) or any(not np.array_equal(z[k],expected[k]) for k in expected):
                raise ValueError('Frozen corpus/rehearsal linkage mismatch')
    if read(base/'inputs/tactical-oracle.json')!=oracle:raise ValueError('Oracle linkage mismatch')
    added_rows=[{**r,'game':r['game']+512} for i in range(1536) for r in read(base/'additional'/f'coverage-{i:04d}.json')]
    rows=original_rows+added_rows;validate_metadata(data,rows)
    train_keys.update(team_keys([read(base/'additional'/f'game-{i:04d}.json') for i in range(1536)]))
    past.update(team_keys([read(base/'probe'/f'game-{i:04d}.json') for i in range(256)]))
    if train_keys & past:raise ValueError('Training/development team overlap')
    return old,oracle,data,rows,train_keys,past

def baseline(seed,plan):
    base=ROOT/plan['predecessor_run'];directory=base/str(seed)/'expanded/final';h=digest(base/'inputs/manifest.json')
    p,opt,meta=load_strict_checkpoint(directory,h)
    require_baseline_meta(meta,opt,seed)
    return p,opt,{'directory':str(directory.relative_to(ROOT)),'pointer':read(directory/'latest.json'),'input_hash':h,'metadata':meta}

def require_baseline_meta(meta,opt,seed):
    if meta['seed']!=seed or meta['cohorts']!=2048 or meta['cursors']!={'bc':2048,'arm':'expanded'} or int(opt[2])!=2048 or meta['optimizer_step']!=2048:
        raise ValueError('Baseline must be the same-seed expanded step2048 checkpoint without reset')

def probe_roles(train,past,fresh):
    if fresh & (train|past):raise ValueError('Fresh probe exact-team overlap')
    return {'frozen_training_teams':len(train),'prior_development_teams_excluded':len(past),'fresh_probe_teams':len(fresh),'disallowed_overlap':0}
