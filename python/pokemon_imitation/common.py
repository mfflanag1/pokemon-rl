"""Frozen predecessor admission and explicitly limited F15 gates."""
import hashlib
import numpy as np
from pokemon_full_train.storage import ROOT,canonical,digest
from pokemon_entropy.common import admit as admit_f13
from pokemon_credit.common import read,initialize
from pokemon_transfer.common import retention
from pokemon_full_learning.statistics import compare
PLAN=ROOT/'contracts/full-imitation-v1/plan.json'

def validate_plan(p):
    if p!=read(PLAN):raise ValueError('Unregistered F15 plan')
    expected={'seeds':[7103,8209,9311],'bc_steps':2048,'teacher_games':512,'probe_games':128,'rl_cohorts':32,'selected_rows':1024,'workers':8,'bc_teacher_rows':64,'bc_old_rows':64,'evaluation_pairs_per_opponent':64}
    if any(p[k]!=v for k,v in expected.items()) or any(p[k] for k in ['final_reserved_data_used','public_server_used','prime_used']):raise ValueError('F15 fixed budget/scope mismatch')

def admit(p):
    path=ROOT/p['predecessor_receipt'];r=read(path);body=dict(r);h=body.pop('content_hash')
    if digest(path)!=p['predecessor_sha256'] or hashlib.sha256(canonical(body)).hexdigest()!=h:raise ValueError('F14 receipt mismatch')
    if path.read_bytes()!=(ROOT/'artifacts/manifests'/('f14-entropy-'+h)/'receipt.json').read_bytes():raise ValueError('F14 archive mismatch')
    for rel,sha in r['source_files'].items():
        if digest(ROOT/rel)!=sha or digest(ROOT/r['source_snapshot_root']/rel)!=sha:raise ValueError('F14 source drift '+rel)
    for rel,sha in r['evidence_files'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('F14 evidence drift '+rel)
    _,sup,oracle=admit_f13(read(ROOT/'contracts/entropy-ablation-v1/plan.json'))
    return sup,oracle

def team_keys(games):return {t['team_key'] for g in games for t in g['provenance']['teams']}

def analysis(panels,tactical,plan):
    comparisons={name:{o:compare(panels[a][o],panels[b][o],plan['statistics']) for o in plan['opponents']} for name,a,b in [
        ('imitation_vs_initial','initial','imitation'),('control_rl_vs_initial','initial','control_rl'),('imitation_rl_vs_initial','initial','imitation_rl'),('imitation_rl_vs_control_rl','control_rl','imitation_rl'),('imitation_rl_vs_imitation','imitation','imitation_rl')]}
    retained={e:retention(tactical['initial'],tactical[e],plan['tactical_thresholds']) for e in plan['endpoints'][1:]}
    def passes(name):
        cs=comparisons[name];t=cs['tactical']
        return (t['gate']=='improvement' and t['cutoff_aware_point_interval'][0]>=plan['statistics']['practical_improvement'] and
            all(c['gate']!='regression' and c['final']['cutoffs']/c['final']['games']<=plan['statistics']['max_final_cutoff_rate'] for c in cs.values()))
    return {'comparisons':comparisons,'retention':retained,
        'imitation_benefit':passes('imitation_vs_initial') and retained['imitation']['passed'],
        'pipeline_benefit':passes('imitation_rl_vs_initial') and passes('imitation_rl_vs_control_rl') and retained['imitation_rl']['passed'],
        'incremental_rl_benefit':passes('imitation_rl_vs_imitation') and retained['imitation_rl']['passed']}
