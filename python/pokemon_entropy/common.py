"""F13 receipt admission and F14's fixed exploratory reporting rules."""
import hashlib
from collections import Counter
import numpy as np
from pokemon_full_train.storage import ROOT,canonical,digest
from pokemon_credit.common import read,admit as admit_f12,initialize
from pokemon_credit.statistics import compare
from pokemon_battle_audit.core import validate_scenario
from pokemon_transfer.common import retention

PLAN=ROOT/'contracts/entropy-ablation-v1/plan.json'

def validate_plan(plan):
    if plan!=read(PLAN):raise ValueError('Plan differs from registration')
    if plan['seeds']!=[7103,8209,9311] or plan['arms']!={'standard':.01,'no_entropy':0.} or plan['target_lambda']!=.95:raise ValueError('Wrong arms or seeds')
    if any(plan[k]!=v for k,v in {'cohorts':32,'selected_rows':256,'epochs':2,'chunk_games':8,'maximum_games_per_cohort':128,'evaluation_repeats':4,'max_turns':120,'rehearsal_coefficient':.1}.items()):raise ValueError('Budget/learner changed')
    if any(plan[k] for k in ['final_reserved_data_used','public_server_used','prime_used']):raise ValueError('Reserved scope')

def admit(plan):
    path=ROOT/plan['predecessor_receipt'];r=read(path);body=dict(r);h=body.pop('content_hash')
    if digest(path)!=plan['predecessor_sha256'] or hashlib.sha256(canonical(body)).hexdigest()!=h:raise ValueError('F13 receipt changed')
    if path.read_bytes()!=(ROOT/'artifacts/manifests'/('f13-credit-'+h)/'receipt.json').read_bytes():raise ValueError('F13 archive changed')
    for rel,sha in r['source_files'].items():
        if digest(ROOT/rel)!=sha or digest(ROOT/r['source_snapshot_root']/rel)!=sha:raise ValueError('F13 source drift: '+rel)
    for rel,sha in r['evidence_files'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('F13 evidence drift: '+rel)
    supervised=admit_f12(read(ROOT/'contracts/credit-audit-v1/plan.json'))
    bank=read(ROOT/plan['bank'])
    for s in bank:validate_scenario(s)
    if len(bank)!=48 or len({s['content_hash'] for s in bank})!=48:raise ValueError('Wrong bank inventory')
    if Counter((s['origin']['role'],s['families'][0]) for s in bank)!=Counter({(r,f):12 for r in ['fit','probe'] for f in ['midgame','endgame']}):raise ValueError('Wrong bank roles')
    oracle=read(ROOT/plan['oracle'])
    if len(oracle['records'])!=512:raise ValueError('Wrong tactical inventory')
    return bank,supervised,oracle

def analysis(initial,finals,initial_tactical,final_tactical,behavior,plan):
    contrasts={}
    for family in plan['families']:
        contrasts[family]={name:compare(a[family],b[family],plan['statistics']) for name,a,b in [
            ('standard_vs_initial',initial,finals['standard']),('no_entropy_vs_initial',initial,finals['no_entropy']),('no_entropy_vs_standard',finals['standard'],finals['no_entropy'])]}
    retained={arm:retention(initial_tactical,final_tactical[arm],plan['tactical_thresholds']) for arm in plan['arms']}
    continuation=all(contrasts[f][c]['gate']=='improvement' and contrasts[f][c]['point_interval'][0]>=plan['statistics']['practical_gain']
        for f in plan['families'] for c in ['no_entropy_vs_initial','no_entropy_vs_standard'])
    delta=behavior['standard']['mean_switch_probability']-behavior['no_entropy']['mean_switch_probability']
    return {'contrasts':contrasts,'retention':retained,'continuation_benefit':continuation,
        'entropy_benefit':continuation and retained['no_entropy']['passed'],
        'fixed_state_switch_reduction':delta,'descriptive_switch_reduction_threshold_met':delta>=plan['mechanism_minimum_switch_reduction']}
