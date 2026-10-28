"""F9 schedules, sealed data admission and fixed development summaries."""
import hashlib
import json
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode
from pokemon_full.policy import forward, distribution
from pokemon_full_train.learning import derived_seed, prepare_batch
from pokemon_full_train.storage import ROOT, canonical, digest
from pokemon_full_train.campaign import value_summary
from pokemon_full_learning.campaign import check_games, evaluation_specs
from pokemon_full_learning.statistics import compare
from pokemon_throughput.collector import merge_rows
from pokemon_retention.common import admit_predecessor, initial_state, input_inventory
from pokemon_decision.collect import evaluation_specs as tactical_specs
from pokemon_fit.data import require, load_data, fingerprint
from pokemon_fit.learning import model_batch


def read(path): return json.loads(path.read_text())


def predecessor(plan):
    r=admit_predecessor(plan)
    path=ROOT/plan['retention_receipt'];prior=read(path);body=dict(prior);h=body.pop('content_hash')
    require(h==plan['retention_content_hash']==hashlib.sha256(canonical(body)).hexdigest(),'F8 receipt binding')
    require(path.read_bytes()==(ROOT/'artifacts/manifests'/f'f8-decision-retention-{h}'/'receipt.json').read_bytes(),'F8 receipt archive')
    for rel,sha in prior['source_files'].items():
        require(digest(ROOT/rel)==sha and digest(ROOT/prior['source_snapshot_root']/rel)==sha,f'F8 source drift: {rel}')
    for rel,sha in prior['evidence_files'].items():require(digest(ROOT/rel)==sha,f'F8 evidence drift: {rel}')
    splits,inventory=load_data(read(ROOT/'contracts/decision-fit-v1/plan.json'))
    require(inventory==read(ROOT/plan['predecessor_run']/'report.json')['dataset'],'F7 dataset inventory drift')
    batch=model_batch(splits['fit_train'])
    return batch,inventory


def initialize(seed,plan): return initial_state(seed,'supervised',plan)


def specs(plan,seed,first):
    return [dict(seed=derived_seed(plan['training_seed_namespace'],seed),episode=i,max_turns=plan['max_turns'],
        split='train',swap=False,opponent=None,learner_side='p1',actor_seed=derived_seed('f9-train-actor',seed))
        for i in range(first,first+plan['training_chunk_games'])]


def select(collection,seed,index,quota=1024):
    # Complete trajectories and their cutoff bootstraps are processed BEFORE subsampling.
    batch=prepare_batch(merge_rows(collection));n=len(batch['actions'])
    require(n>=quota,'Insufficient complete-trajectory rows')
    ids=np.random.default_rng(derived_seed('f9-row-selection',seed,index)).choice(n,quota,replace=False)
    selected={k:a[ids] for k,a in batch.items()}
    return selected,{'collected_rows':n,'selected_rows':quota,'excluded_rows':n-quota,'selected_indices':ids.tolist(),
        'complete_batch_sha256':fingerprint(batch),'selected_batch_sha256':fingerprint(selected)}


def collect_update(pool,p,plan,seed,index,cursor):
    collection=[];profiles=[];first=cursor
    while sum(len(rows) for rows,_ in collection)<plan['selected_rows_per_update']:
        require(cursor-first<plan['maximum_games_per_update'],'Fixed collection cap exhausted')
        chunk,profile=pool.collect(p,specs(plan,seed,cursor));games=[g for _,g in chunk]
        check_games(games,split='train',first=cursor,count=plan['training_chunk_games'])
        require(all(len(rows)==g['decisions'] and g['unavailable_choices']==0 for rows,g in chunk),'Train rows/choice mismatch')
        collection.extend(chunk);profiles.append(profile);cursor+=plan['training_chunk_games']
    batch,selection=select(collection,seed,index,plan['selected_rows_per_update'])
    return batch,{'first_episode':first,'next_episode':cursor,'games':[g for _,g in collection],'selection':selection},profiles


def evaluation(pool,p,plan,seed):
    result={};profiles=[]
    for opponent in plan['opponents']:
        games=[]
        for first in range(0,plan['evaluation_pairs_per_opponent'],plan['evaluation_chunk_pairs']):
            chunk,profile=pool.collect(p,evaluation_specs(plan,seed,opponent,first,plan['evaluation_chunk_pairs']))
            require(all(not rows and g['unavailable_choices']==0 for rows,g in chunk),'Evaluation learner rows or unavailable choice')
            games.extend(g for _,g in chunk);profiles.append(profile)
        check_games(games,split='development',first=0,count=plan['evaluation_pairs_per_opponent'],opponent=opponent)
        result[opponent]=games
    return result,profiles


def probe_specs(plan): return tactical_specs(plan['tactical_evaluation_seed'],plan['tactical_evaluation_repeats'])


def tactical(p,oracle):
    require(len(oracle['records'])==512,'Wrong tactical probe size')
    cells={};predictions=[]
    for first in range(0,512,8):
        records=oracle['records'][first:first+8];encoded=[encode(r['player']) for r in records]
        logits,values=forward(p,jnp.asarray(np.stack([x for x,_ in encoded])))
        probs,_=distribution(logits,jnp.asarray(np.stack([m for _,m in encoded])))
        for r,pr,value,(_,mask) in zip(records,np.asarray(probs),np.asarray(values),encoded):
            require(np.isfinite(pr).all() and np.all(pr>=0) and np.all(pr[~mask]==0) and abs(float(pr.sum())-1)<5e-6,'Invalid tactical probabilities')
            require(np.isfinite(value) and 0<=value<=1,'Invalid full-battle value')
            item={'target_mass':float(pr[r['target']]),'greedy_correct':int(pr.argmax())==r['target'],
                  'expected_local_reward':sum(float(pr[int(a)])*v for a,v in r['scores'].items()),
                  'probabilities':pr.tolist(),'full_battle_value':float(value)}
            predictions.append(item);key=f"{r['spec']['panel']}/{r['spec']['context']}";cells.setdefault(key,[]).append(item)
    require(set(cells)=={f'{p}/{c}' for p in ['train','variant'] for c in range(16)},'Incomplete tactical grid')
    return {'predictions':predictions,'cells':{k:{'n':len(v),'target_mass':float(np.mean([r['target_mass'] for r in v])),
        'greedy_accuracy':float(np.mean([r['greedy_correct'] for r in v])),
        'expected_local_reward':float(np.mean([r['expected_local_reward'] for r in v]))} for k,v in cells.items()}}


def retention(initial,final,settings):
    require(set(initial['cells'])==set(final['cells'])=={f'{p}/{c}' for p in ['train','variant'] for c in range(16)},'Incomplete retention cells')
    cells={}
    for key in initial['cells']:
        a,b=initial['cells'][key],final['cells'][key]
        require(a['n']==b['n']==16,'Wrong retention cell count')
        drop=a['target_mass']-b['target_mass']
        checks={'greedy':b['greedy_accuracy']>=settings['minimum_greedy_accuracy'],
            'mass':b['target_mass']>=settings['minimum_context_mass'],'drop':drop<=settings['maximum_context_mass_drop']}
        cells[key]={'mass_drop':drop,'checks':checks,'passed':all(checks.values())}
    return {'passed':all(c['passed'] for c in cells.values()),'cells':cells}


def summarize(initial,finals,initial_probe,final_probes,plan):
    settings=plan['statistics'];comparisons={}
    for label,a,b in [('plain_minus_initial',initial,finals['plain']),('rehearsal_minus_initial',initial,finals['rehearsal']),
                       ('rehearsal_minus_plain',finals['plain'],finals['rehearsal'])]:
        comparisons[label]={o:compare(a[o],b[o],settings) for o in plan['opponents']}
    retain={arm:retention(initial_probe,final_probes[arm],plan['tactical_thresholds']) for arm in plan['arms']}
    advantage=float(np.mean([final_probes['rehearsal']['cells'][k]['target_mass']-final_probes['plain']['cells'][k]['target_mass'] for k in initial_probe['cells']]))
    learning={}
    for arm in plan['arms']:
        cells=comparisons[f'{arm}_minus_initial'];base=cells['base_power']
        checks={'base_point_gain':base['cutoff_aware_point_interval'][0]>=settings['practical_improvement'],
            'base_bootstrap_positive':base['approximate_familywise_bootstrap_interval'][0]>0,
            'no_detected_regression':all(c['gate']!='regression' for c in cells.values()),
            'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=settings['max_final_cutoff_rate'] for c in cells.values())}
        learning[arm]={'checks':checks,'passed':all(checks.values())}
    return {'comparisons':comparisons,'retention':retain,'rehearsal_mean_target_mass_advantage':advantage,
        'rehearsal_benefit_passed':retain['rehearsal']['passed'] and advantage>=plan['tactical_thresholds']['minimum_mean_rehearsal_advantage'],
        'full_battle_learning':learning}


def verified_inventory(oracle,batch,plan):
    """Bind the historical overlap proxy to the actual admitted rehearsal inputs."""
    actual={fingerprint({'features':x,'mask':m}) for x,m in zip(batch['features'],batch['mask'])}
    result=input_inventory(oracle,plan)
    require(result['supervised_training_unique_inputs']==len(actual),'Rehearsal inventory differs from historical proxy')
    for panel in ['train','variant']:
        states=set()
        for row in oracle['records']:
            if row['spec']['panel']==panel:
                x,m=encode(row['player']);states.add(fingerprint({'features':x,'mask':m}))
        require(result['overlap_with_supervised_training'][panel]==len(states&actual),'Rehearsal overlap differs from historical proxy')
    return {**result,'actual_supervised_batch_verified':True}
