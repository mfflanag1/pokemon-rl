"""Fixed F33 pilot, retained decisions, and independent fresh replay."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest, freeze, admission
from pokemon_full_train.learning import derived_seed
from pokemon_full_train.campaign import value_summary, score_interval
from pokemon_full_learning.statistics import compare
from pokemon_opponent_mix.episode import baseline
from pokemon_worker_watch.transport import Worker
from .core import COMMAND, SIDES, Search, load_frozen, predictions, frozen_actions, joint

PLAN=ROOT/'contracts/opening-search-v1/plan.json'


def scientific(value):
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k!='seconds'}
    if isinstance(value,list):return [scientific(v) for v in value]
    return value


def sha(value):return hashlib.sha256(canonical(value)).hexdigest()


def play(worker,parameters,catalog,plan,*,seed,episode,swap,opponent,model_seed,
         search=None,forced=None,replicate=0):
    side='p2' if swap else 'p1';foe='p1' if swap else 'p2'
    frame=worker.request('start',run_seed=seed,episode_index=episode,split='development',
        swap=swap,tactical_side=foe if opponent=='tactical' else None)
    provenance=frame['provenance'];trace=[];values=[];decisions=0;rejections=0;root=None
    counts={s:0 for s in SIDES};leaf=None;opening_mask=frame['players'][side]['mask']
    if forced is not None and not opening_mask[forced]:raise ValueError('Audit root illegal')
    for request_index in range(plan['max_decisions']):
        pred=predictions(parameters,[frame['players'][s] for s in SIDES],catalog)
        if leaf is None and (frame['outcome'] is not None or joint(frame)):
            leaf={'player':frame['players'][side], 'value':pred[SIDES.index(side)][1],
                'outcome':frame['outcome'],'frame_sha256':sha(frame)}
        if frame['outcome'] is not None:break
        if joint(frame,plan['max_turns']):
            frame=worker.request('cutoff');break
        actor_seeds=[derived_seed('f33-actor',model_seed,seed,episode,swap,replicate,s) for s in SIDES]
        outputs=frozen_actions(parameters,[frame['players'][s] for s in SIDES],catalog,
            actor_seeds,[counts[s] for s in SIDES])
        actions={};ids={};policy={}
        for i,s in enumerate(SIDES):
            player=frame['players'][s]
            if not any(player['mask']):continue
            pr,value=pred[i]
            action_seed=actor_seeds[i]
            action=outputs[i][2]
            if s==foe and opponent!='self_play':
                action=frame['tactical_actions'][s] if opponent=='tactical' else baseline(player,opponent,
                    np.random.default_rng(derived_seed('f33-baseline',seed,episode,swap,s,counts[s])))
            if s==side:
                values.append(value)
                if request_index==0:
                    if forced is not None:action=forced
                    elif search is not None:
                        input=worker.request('opening',side=side)
                        search_seed=derived_seed('f33-search-input',model_seed,sha(input))
                        root=search.decide(input,search_seed,pr,action)
                        root['input']=input
                        action=root['action']
                policy[s]={'probabilities':pr.tolist(),'value':value,'base_sample_seed':action_seed,'counter':counts[s]}
            elif opponent=='self_play':policy[s]={'probabilities':pr.tolist(),'value':value,'base_sample_seed':action_seed,'counter':counts[s]}
            if not 0<=action<14 or not player['mask'][action]:raise ValueError('Actual action legality')
            actions[s]=int(action);ids[s]=player['request_id'];counts[s]+=1
        before=sha(frame);frame=worker.request('step',actions=actions,request_ids=ids)
        trace.append({'before_sha256':before,'actions':actions,'request_ids':ids,'policy':policy,
            'after_sha256':sha(frame),'submitted':frame['submitted']})
        decisions+=len(actions);rejections+=sum(not r['accepted'] for r in frame['submitted'])
    else:raise RuntimeError('Fixed episode request budget exhausted')
    outcome=frame['outcome']
    if outcome is None:raise ValueError('Missing outcome')
    terminal=outcome['kind']=='terminal';reward=outcome['rewards'][side]
    return {'episode':episode,'swap':swap,'learner_side':side,'opponent':opponent,
        'provenance':provenance,'outcome':outcome,'decisions':decisions,'unavailable_choices':rejections,
        'trace':trace,'root_search':root,'leaf':leaf,'opening_mask':opening_mask,
        'value_mean':float(np.mean(values)),'terminal_value_mse':float(np.mean((np.asarray(values)-reward)**2)) if terminal else None,
        'replicate':replicate,'forced_root_action':forced}


def audit_summary(records):
    terminal=[r for r in records if r['outcome']['kind']=='terminal' and r['leaf']]
    labels=np.asarray([r['outcome']['rewards'][r['learner_side']] for r in terminal])
    estimates=np.asarray([r['leaf']['value'] for r in terminal])
    groups={}
    for r in records:groups.setdefault(r['episode'],{}).setdefault(r['forced_root_action'],[]).append(r)
    ranks=[]
    for episode,actions in groups.items():
        rows=[]
        for action,rs in actions.items():
            bounds=[score_interval(r) for r in rs]
            rows.append({'action':action,'value':float(np.mean([r['leaf']['value'] for r in rs if r['leaf']])),
                'return_interval':np.mean(bounds,axis=0).tolist()})
        comparable=concordant=discordant=0
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                av,bv=a['return_interval'],b['return_interval']
                direction=1 if av[0]>bv[1] else -1 if av[1]<bv[0] else 0
                if direction and a['value']!=b['value']:
                    comparable+=1
                    concordant+=int((a['value']-b['value'])*direction>0)
                    discordant+=int((a['value']-b['value'])*direction<0)
        ranks.append({'episode':episode,'actions':rows,'strictly_ordered_pairs':comparable,
            'concordant':concordant,'discordant':discordant,
            'concordance':concordant/comparable if comparable else None})
    return {'games':len(records),'terminals':len(terminal),'cutoffs':len(records)-len(terminal),
        'decisions':sum(r['decisions'] for r in records),
        'fixed_successor_terminal_mse':float(np.mean((estimates-labels)**2)) if len(labels) else None,
        'constant_half_mse':float(np.mean((.5-labels)**2)) if len(labels) else None,
        'fitted_base_rate_mse_descriptive':float(np.mean((labels.mean()-labels)**2)) if len(labels) else None,
        'rankings':ranks,'limitation':'Noisy two-continuation outcome ranks at four fixed openings; ties and overlapping cutoff bounds are not ranked. No calibration or critic-failure gate.'}


def execute(run,plan,original=None):
    run.mkdir(parents=True,exist_ok=False)
    manifest=freeze(run,plan); started=time.perf_counter();all_results={};audit_results={};identities={}
    def retain(rel,record):
        if original is not None:
            expected=json.loads((original/rel).read_text())
            if scientific(record)!=scientific(expected):raise ValueError('Fresh replay mismatch: '+str(rel))
        atomic_json(run/rel,record)
    for seed in plan['seeds']:
        parameters,catalog,identity=load_frozen(seed);identities[str(seed)]=identity
        worker=Worker(run/f'workers/{seed}/real',command=COMMAND)
        hyp=Worker(run/f'workers/{seed}/hypothetical',command=COMMAND)
        try:
            search=Search(hyp,parameters,catalog,plan['search']);records=[]
            for episode in range(plan['audit_openings']):
                frame=worker.request('start',run_seed=plan['audit_seed'],episode_index=episode,
                    split='development',swap=False,tactical_side=None)
                actions=[a for a,legal in enumerate(frame['players']['p1']['mask']) if legal]
                for action in actions:
                    for replicate in range(plan['audit_rollouts_per_action']):
                        record=play(worker,parameters,catalog,plan,seed=plan['audit_seed'],episode=episode,
                            swap=False,opponent='self_play',model_seed=seed,forced=action,replicate=replicate)
                        retain(Path(f'{seed}/audit/{episode}-{action}-{replicate}.json'),record);records.append(record)
                atomic_json(run/'progress.json',{'stage':'audit','seed':seed,'opening_completed':episode+1})
            audit_results[str(seed)]=audit_summary(records);panels={}
            for opponent in plan['opponents']:
                panels[opponent]={a:[] for a in plan['arms']}
                for episode in range(plan['pairs_per_opponent']):
                    for swap in (False,True):
                        for arm in plan['arms']:
                            record=play(worker,parameters,catalog,plan,seed=plan['evaluation_seed'],episode=episode,
                                swap=swap,opponent=opponent,model_seed=seed,search=search if arm=='search' else None)
                            retain(Path(f'{seed}/{opponent}/{arm}/{episode}-{int(swap)}.json'),record)
                            panels[opponent][arm].append(record)
                    atomic_json(run/'progress.json',{'stage':'comparison','seed':seed,'opponent':opponent,'pairs_completed':episode+1})
            comparisons={o:compare(panels[o]['policy'],panels[o]['search'],plan['statistics']) for o in plan['opponents']}
            roots=[r['root_search'] for panel in panels.values() for r in panel['search']]
            tactical=comparisons['tactical']
            checks={'point_gain':tactical['cutoff_aware_point_interval'][0]>=plan['statistics']['practical_improvement'],
                'adjusted_lower_positive':tactical['approximate_familywise_bootstrap_interval'][0]>0,
                'no_detected_regression':all(comparisons[o]['gate']!='regression' for o in ('random','base_power')),
                'cutoff_rates':all(c['final']['cutoffs']/c['final']['games']<=plan['statistics']['max_final_cutoff_rate'] for c in comparisons.values())}
            all_results[str(seed)]={'comparisons':comparisons,'gate':{'checks':checks,'passed':all(checks.values())},
                'search':{'roots':len(roots),'applied':sum(r['applied'] for r in roots),
                    'changed_action':sum(r['action']!=r['fallback_action'] for r in roots),
                    'fallback_reasons':{reason:sum(r['reason']==reason for r in roots) for reason in sorted({r['reason'] for r in roots if r['reason']})},
                    'p50_seconds':float(np.median([r['seconds'] for r in roots])),
                    'p99_seconds':float(np.quantile([r['seconds'] for r in roots],.99)),
                    'maximum_seconds':max(r['seconds'] for r in roots),
                    'surrogate_branches':sum(r.get('attempted_branches',0) for r in roots)},
                'arms':{arm:value_summary([r for panel in panels.values() for r in panel[arm]]) for arm in plan['arms']}}
        finally:worker.close();hyp.close()
    admission(run,manifest)
    total=sum(v['arms'][a]['games'] for v in all_results.values() for a in plan['arms'])
    if total!=plan['expected_main_games']:raise ValueError('Main game count')
    report={'status':'passed','plan_sha256':digest(PLAN),'input_sha256':digest(run/'inputs/manifest.json'),
        'seconds':time.perf_counter()-started,'main_games':total,'audit_games':sum(a['games'] for a in audit_results.values()),
        'optimizer_updates':0,'peak_parent_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'checkpoint_identities':identities,'seeds':all_results,'audit':audit_results,
        'all_seed_benefit_gate':all(r['gate']['passed'] for r in all_results.values()),'fresh_replay':original is not None}
    atomic_json(run/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--replay-of')
    args=parser.parse_args();plan=json.loads(PLAN.read_text())
    execute(Path(args.run).resolve(),plan,Path(args.replay_of).resolve() if args.replay_of else None)


if __name__=='__main__':main()
