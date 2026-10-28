"""Sealed admission, complete-trajectory stratification and registered F12 gates."""
from collections import Counter
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full_train.storage import ROOT,canonical,digest,load_checkpoint
from pokemon_full_train.learning import derived_seed,prepare_batch
from pokemon_throughput.collector import merge_rows
from pokemon_full_learning.campaign import evaluation_specs,check_games
from pokemon_full_learning.statistics import compare
from pokemon_long_transfer.value import summarize_value
from pokemon_battle_audit.core import validate_scenario
from pokemon_battle_audit.campaign import predecessor as admit_f10
from pokemon_fit.data import fingerprint
from pokemon_fit.learning import validate_batch
from .schedule import quotas,choose_scenario,select_indices


def read(path):return json.loads(path.read_text())


def validate_plan(plan):
    if plan!=read(ROOT/'contracts/curriculum-learning-v1/plan.json'):raise ValueError('Unregistered plan')
    if plan['seeds']!=[7103,8209,9311] or plan['arms']!=['official','curriculum']:raise ValueError('Wrong fixed arms/seeds')
    if plan['updates_per_arm']!=48 or plan['workers']!=8 or plan['selected_rows_per_update']!=1024:raise ValueError('Changed budget/layout')
    expected={'selected_ppo_rows':294912,'ppo_row_presentations':589824,'supervised_row_presentations':294912,
              'optimizer_steps':2304,'evaluation_games':6912,'selected_prefix_rows':27024}
    if plan['expected_counts']!=expected:raise ValueError('Wrong counts')
    if jax.default_backend()!='cpu' or jax.config.x64_enabled or not jax.config.jax_threefry_partitionable:raise ValueError('Wrong runtime')


def admit(plan):
    receipt=ROOT/plan['predecessor_receipt'];r=read(receipt);body=dict(r);h=body.pop('content_hash')
    if digest(receipt)!=plan['predecessor_receipt_sha256'] or h!=plan['predecessor_content_hash'] or h!=hashlib.sha256(canonical(body)).hexdigest():
        raise ValueError('F11 receipt mismatch')
    if receipt.read_bytes()!=(ROOT/'artifacts/manifests'/f'f11-battle-decisions-{h}'/'receipt.json').read_bytes():raise ValueError('F11 archive mismatch')
    for rel,sha in r['source_files'].items():
        if digest(ROOT/rel)!=sha or digest(ROOT/r['source_snapshot_root']/rel)!=sha:raise ValueError('F11 source changed: '+rel)
    for rel,sha in r['evidence_files'].items():
        if digest(ROOT/rel)!=sha:raise ValueError('F11 evidence changed: '+rel)
    admit_f10(read(ROOT/'contracts/battle-audit-v1/plan.json'))
    bank={}
    for seed in plan['seeds']:
        for scenario in read(ROOT/plan['predecessor_run']/'training'/str(seed)/'scenarios.json'):
            validate_scenario(scenario)
            if scenario['content_hash'] in bank:raise ValueError('Duplicate scenario')
            bank[scenario['content_hash']]=scenario
    if len(bank)!=65:raise ValueError('Wrong bank size')
    weights=read(ROOT/plan['predecessor_run']/'curriculum-recipe.json')['family_weights']
    with np.load(ROOT/plan['initialization_run']/'inputs/supervised.npz',allow_pickle=False) as data:
        supervised={k:data[k].copy() for k in data.files}
    validate_batch(supervised)
    return bank,weights,supervised


def initialize(seed,plan):
    p,_,meta=load_checkpoint(ROOT/plan['initialization_run']/str(seed)/'rehearsal/final',plan['initialization_input_hash'])
    if meta['seed']!=seed or meta['cohorts']!=128:raise ValueError('Wrong F10 endpoint')
    zero=jax.tree.map(jnp.zeros_like,p)
    return p,(zero,zero,jnp.int32(0))


def training_specs(plan,seed,stratum,first,bank,weights):
    specs=[]
    for ordinal in range(first,first+plan['chunk_games']):
        spec=dict(seed=derived_seed('f12-official-train',seed),episode=ordinal,max_turns=plan['max_turns'],
            split='train',swap=False,opponent='tactical',learner_side='p1' if ordinal%2==0 else 'p2',
            actor_seed=derived_seed('f12-actor',seed,stratum))
        if stratum=='prefix':
            _,h=choose_scenario(list(bank.values()),weights,derived_seed('f12-bank',seed,ordinal))
            spec.update(scenario_hash=h,learner_side=bank[h]['learner_side'])
        elif stratum!='official':raise ValueError('Unknown training stratum')
        specs.append(spec)
    return specs


def collect_update(pool,p,plan,seed,arm,index,cursors,bank,weights):
    target=quotas(arm,index);collections={};profiles=[];start=dict(cursors);next_cursors=dict(cursors)
    for stratum,quota in target.items():
        collection=[];cursor=cursors[stratum]
        while sum(len(rows) for rows,_ in collection)<quota:
            if cursor-start[stratum]>=plan['maximum_games_per_stratum_update']:raise ValueError('Registered collection cap exhausted')
            specs=training_specs(plan,seed,stratum,cursor,bank,weights)
            chunk,profile=pool.collect(p,specs)
            check_games([g for _,g in chunk],split='train',first=cursor,count=plan['chunk_games'],opponent='tactical')
            for rows,g in chunk:
                if len(rows)!=g['learner_decisions'] or any(r['trajectory'] not in [0,1] for r in rows):raise ValueError('Bad learner rows')
                if (g['prefix']['scenario_hash'] is not None)!=(stratum=='prefix'):raise ValueError('Stratum mismatch')
            collection.extend(chunk);profiles.append(profile);cursor+=len(specs)
        collections[stratum]=collection;next_cursors[stratum]=cursor
    batches={s:prepare_batch(merge_rows(c)) for s,c in collections.items() if c}
    sizes={s:len(batches[s]['actions']) if s in batches else 0 for s in ['official','prefix']}
    ids=select_indices(sizes['official'],sizes['prefix'],target['prefix'],derived_seed('f12-select',seed,index))
    batch={k:np.stack([batches[s][k][i] for s,i in ids]) for k in batches['official']}
    ledger={'first':start,'next':next_cursors,'quotas':target,'collected_rows':sizes,
        'excluded_rows':{s:sizes[s]-target[s] for s in sizes},'selected_indices':ids,
        'full_batch_hashes':{s:fingerprint(b) for s,b in batches.items()},'selected_batch_hash':fingerprint(batch),
        'games':{s:[g for _,g in c] for s,c in collections.items()}}
    return batch,ledger,profiles


def evaluation(pool,p,plan,seed):
    panels={};profiles=[]
    for opponent in plan['opponents']:
        games=[]
        for first in range(0,plan['evaluation_pairs_per_opponent'],plan['evaluation_chunk_pairs']):
            chunk,profile=pool.collect(p,evaluation_specs(plan,seed,opponent,first,plan['evaluation_chunk_pairs']))
            if any(rows for rows,_ in chunk):raise ValueError('Evaluation entered training')
            games.extend(g for _,g in chunk);profiles.append(profile)
        check_games(games,split='development',first=0,count=plan['evaluation_pairs_per_opponent'],opponent=opponent)
        if any(g['prefix']['scenario_hash'] is not None for g in games):raise ValueError('Prefix in evaluation')
        panels[opponent]=games
    return panels,profiles


def summarize(initial,finals,plan):
    comparisons={label:{o:compare(a[o],b[o],plan['statistics']) for o in plan['opponents']}
        for label,a,b in [('official_minus_initial',initial,finals['official']),
                          ('curriculum_minus_initial',initial,finals['curriculum']),
                          ('curriculum_minus_official',finals['official'],finals['curriculum'])]}
    difference=comparisons['curriculum_minus_official'];improvement=comparisons['curriculum_minus_initial']
    checks={'tactical_vs_control_five_points':difference['tactical']['cutoff_aware_point_interval'][0]>=.05,
        'tactical_vs_control_bootstrap_positive':difference['tactical']['approximate_familywise_bootstrap_interval'][0]>0,
        'tactical_vs_initial_five_points':improvement['tactical']['cutoff_aware_point_interval'][0]>=.05,
        'tactical_vs_initial_bootstrap_positive':improvement['tactical']['approximate_familywise_bootstrap_interval'][0]>0,
        'no_detected_other_opponent_regression':all(c[o]['gate']!='regression' for c in [difference,improvement] for o in ['random','base_power']),
        'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=.05 for c in improvement.values())}
    endpoints={'initial':initial,**finals}
    return {'comparisons':comparisons,'curriculum_benefit':{'checks':checks,'passed':all(checks.values())},
        'value_diagnostics':{e:{o:value_diagnostics(g) for o,g in panel.items()} for e,panel in endpoints.items()},
        'decision_diagnostics':{e:{o:dict(sum((Counter(g['diagnostics']) for g in games),Counter())) for o,games in panel.items()}
                                for e,panel in endpoints.items()}}


def value_diagnostics(games):
    result=summarize_value(games)
    terminal=[g for g in games if g['outcome']['kind']=='terminal']
    if terminal:
        targets=np.asarray([g['outcome']['rewards'][g['learner_side']] for g in terminal],np.float64)
        sizes=np.asarray([sum(row[0]==g['learner_side'] for row in g['trace']) for g in terminal],np.float64)
        game_mean=float(targets.mean());decision_mean=float(np.average(targets,weights=sizes))
        result['same_cell_fitted_constant']={'game_prediction':game_mean,'game_mse':float(np.mean((targets-game_mean)**2)),
            'decision_prediction':decision_mean,'decision_mse':float(np.average((targets-decision_mean)**2,weights=sizes)),
            'interpretation':'Post-hoc constant fitted on these same terminal evaluation outcomes; not an independent baseline.'}
    else:result['same_cell_fitted_constant']=None
    return result
