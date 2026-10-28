"""Admit immutable F17/F16 evidence; no predecessor source is modified."""
import hashlib
import json
from pathlib import Path
import numpy as np
from pokemon_full_train.storage import ROOT, canonical, digest, runtime, load_checkpoint
from pokemon_balanced.common import admit as admit_f15, initialize, team_keys
from .collection import validate_metadata

PLAN = ROOT/'contracts/data-coverage-v1/plan.json'
def read(path):
    return json.loads(path.read_text())

def receipt(path, expected, archive_prefix):
    r = read(path); body = dict(r); h = body.pop('content_hash')
    if digest(path) != expected or hashlib.sha256(canonical(body)).hexdigest() != h:
        raise ValueError('Predecessor receipt mismatch')
    if path.read_bytes() != (ROOT/'artifacts/manifests'/f'{archive_prefix}-{h}'/'receipt.json').read_bytes():
        raise ValueError('Predecessor receipt archive mismatch')
    for rel, sha in r['source_files'].items():
        if digest(ROOT/rel) != sha or digest(ROOT/r['source_snapshot_root']/rel) != sha:
            raise ValueError('Predecessor source drift: '+rel)
    for rel, sha in r['evidence_files'].items():
        if digest(ROOT/rel) != sha:
            raise ValueError('Predecessor evidence drift: '+rel)
    if read(ROOT/r['run']/'inputs/manifest.json')['runtime'] != runtime():
        raise ValueError('Predecessor runtime mismatch')
    return r

def validate_plan(p):
    expected = {'seeds':[7103,8209,9311], 'arms':['original','expanded'], 'workers':8,
        'max_turns':120, 'bc_steps':2048, 'teacher_quota':[32,16,8,8], 'old_rows':64,
        'original_games':512, 'additional_games':1536, 'probe_games':256,
        'teacher_seed':1025261801, 'probe_seed':1025261802, 'expected_updates':12288,
        'expected_presentations':1572864, 'expected_new_games':1792,
        'checkpoint_comparisons':18, 'checkpoint_recoveries':6}
    if p != read(PLAN) or any(p.get(k) != v for k,v in expected.items()) or any(p[k] for k in ['prime_used','public_server_used','final_reserved_data_used']):
        raise ValueError('F18 fixed scope/budget mismatch')

def admit(plan):
    f17 = receipt(ROOT/plan['predecessor_receipt'], plan['predecessor_sha256'], 'f17-switch-representation')
    receipt(ROOT/f17['predecessor_receipt'], f17['predecessor_sha256'], 'f16-balanced-imitation')
    old, oracle, data, train_keys, past = admit_f15(read(ROOT/'contracts/balanced-imitation-v1/plan.json'))
    f16 = ROOT/plan['f16_run']
    for name, expected in [('teacher',data),('old-supervised',old)]:
        with np.load(f16/'inputs'/(name+'.npz'),allow_pickle=False) as z:
            if set(z.files) != set(expected) or any(not np.array_equal(z[k],expected[k]) for k in expected):
                raise ValueError('Original corpus/rehearsal differs from sealed F16')
    past.update(team_keys([read(f16/'probe'/f'game-{i:04d}.json') for i in range(128)]))
    for seed in plan['seeds']:
        for endpoint in ['initial','uniform','balanced']:
            panel = read(f16/str(seed)/endpoint/'evaluation.json')['games']
            past.update(team_keys([g for games in panel.values() for g in games]))
    rows = [r['metadata'] for i in range(512) for r in read(ROOT/'artifacts/runs/f17-switch-audit-v1/train'/f'game-{i:04d}.json')]
    validate_metadata(data, rows)
    return old, oracle, data, rows, train_keys, past

def load_strict_checkpoint(directory, input_hash):
    pointer = read(directory/'latest.json')
    if Path(pointer['file']).name != pointer['file']:
        raise ValueError('Invalid checkpoint filename')
    path = directory/pointer['file']
    if digest(path) != pointer['sha256']:
        raise ValueError('Checkpoint digest mismatch')
    with np.load(path,allow_pickle=False) as z:
        meta = json.loads(z['metadata'].tobytes())
        expected = {'metadata'} | {prefix+'__'+key for prefix in ['p','m','v'] for key in ['w','b','pw','pb','vw','vb']}
        if set(z.files) != expected or type(meta['optimizer_step']) is not int or not 0 <= meta['optimizer_step'] < 2**31:
            raise ValueError('Checkpoint inventory/counter mismatch')
    return load_checkpoint(directory,input_hash)

def compare_control(parameters, optimizer, seed, plan):
    base = ROOT/plan['f16_run']; h = digest(base/'inputs/manifest.json')
    p, o, meta = load_strict_checkpoint(base/str(seed)/'balanced', h)
    if meta['seed'] != seed or meta['cohorts'] != 2048 or meta['cursors'] != {'bc':2048}:
        raise ValueError('F16 control metadata mismatch')
    for a,b in [(parameters,p), (optimizer[0],o[0]), (optimizer[1],o[1])]:
        if set(a) != set(b) or any(not np.array_equal(np.asarray(a[k]),np.asarray(b[k])) for k in a):
            raise ValueError('Original control diverged from sealed F16')
    if int(optimizer[2]) != int(o[2]):
        raise ValueError('Control optimizer counter mismatch')
    return True
