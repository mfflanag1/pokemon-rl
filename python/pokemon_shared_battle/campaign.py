"""One frozen battle panel and a fresh exhaustive replay; never train or select."""
import argparse,resource,time,shutil
from collections import Counter
from pathlib import Path
from pokemon_full_train.storage import ROOT,freeze,admission,digest,atomic_json
from pokemon_full.learner import parameter_hash
from pokemon_full_train.campaign import value_summary
from pokemon_long_transfer.value import summarize_value
from pokemon_credit.campaign import record,same
from pokemon_shared.learning import tactical,mastery
from .common import PLAN,read,validate_plan,admit,load_policy,role_check,check_collection,evaluation_specs,analysis
from .collector import Pool
TIMING=['elapsed_seconds','collection_seconds','peak_parent_rss_mib','profiles']

def campaign(run,original=None):
    plan=read(PLAN);validate_plan(plan);oracle,train,past=admit(plan)
    if original:
        previous=read(original/'inputs/manifest.json');admission(original,previous);same(previous['plan'],plan,'plan')
    run.mkdir(parents=True,exist_ok=False);manifest=freeze(run,plan);h=digest(run/'inputs/manifest.json')
    if original:same(manifest,previous,'manifest')
    def evidence(rel,value):record(run/rel,value,original/rel if original else None)
    begun=time.perf_counter();profiles=[];all_games=[];fresh=set();seeds={};loads=0;actor_decisions=0
    evidence('roles.json',{'training_teams_excluded':len(train),'prior_development_teams_excluded':len(past)})
    for seed in plan['seeds']:
        panels={};tacticals={};identities={}
        for arm in plan['arms']:
            p,opt,identity=load_policy(plan,seed,arm);loads+=1;architecture=identity['architecture']
            base=Path(str(seed))/arm;identities[arm]=identity
            evidence(str(base/'checkpoint-identity.json'),identity)
            source=ROOT/identity['directory'];target=run/base/'checkpoint';target.mkdir(parents=True)
            for name in ['latest.json',identity['pointer']['file']]:
                shutil.copyfile(source/name,target/name)
                if original and (target/name).read_bytes()!=(original/base/'checkpoint'/name).read_bytes():
                    raise ValueError('Archived checkpoint differs')
            t=tactical(p,oracle,architecture);tacticals[arm]=mastery(t,plan['mastery_thresholds'])
            if arm!='reference':
                same(t,read(ROOT/plan['predecessor_run']/str(seed)/arm/'8192/tactical.json'),'F20 familiar tactical predictions')
            evidence(str(base/'tactical.json'),t)
            panels[arm]={}
            with Pool(run/base/'workers',plan['workers'],architecture) as pool:
                for opponent in plan['opponents']:
                    games=[]
                    for first in range(0,plan['evaluation_pairs_per_opponent'],plan['evaluation_chunk_pairs']):
                        jobs=evaluation_specs(plan,seed,opponent,first,plan['evaluation_chunk_pairs'])
                        collection,profile=pool.collect(p,jobs)
                        # Persist the complete chunk before coordinator admission; never filter a rejected choice.
                        chunk=[g for _,g in collection]
                        evidence(str(base/'games'/f'{opponent}-{first:04d}.json'),{'specs':jobs,'games':chunk})
                        check_collection(collection,first,plan['evaluation_chunk_pairs'],opponent)
                        fresh.update(role_check(chunk,train,past))
                        if parameter_hash(p)!=identity['parameter_hash'] or int(opt[2])!=identity['optimizer_step']:
                            raise ValueError('Frozen model changed')
                        games.extend(chunk);profiles.append({'seed':seed,'arm':arm,'opponent':opponent,'first_pair':first,**profile})
                        atomic_json(run/'progress.json',{'stage':'evaluation','seed':seed,'arm':arm,'opponent':opponent,
                            'pairs_completed':first+plan['evaluation_chunk_pairs'],'games_completed':len(all_games)+len(games),
                            'elapsed_seconds':time.perf_counter()-begun})
                    panels[arm][opponent]=games;all_games.extend(games)
                    actor_decisions+=sum(len(g['actor_trace']) for g in games)
            evidence(str(base/'evaluation-summary.json'),{o:value_summary(gs) for o,gs in panels[arm].items()})
        result={'identities':identities,'panels':{a:{o:value_summary(gs) for o,gs in panel.items()} for a,panel in panels.items()},
            'seat_panels':{a:{o:{side:value_summary([g for g in gs if g['learner_side']==side]) for side in ['p1','p2']} for o,gs in panel.items()} for a,panel in panels.items()},
            'action_counts':{a:{o:dict(Counter(str(row['action']) for g in gs for row in g['actor_trace'])) for o,gs in panel.items()} for a,panel in panels.items()},
            'terminal_value':{a:{o:summarize_value(gs) for o,gs in panel.items()} for a,panel in panels.items()},
            'analysis':analysis(panels,tacticals,plan)}
        evidence(str(Path(str(seed))/'result.json'),result);seeds[str(seed)]=result
    if len(all_games)!=6912 or loads!=9:raise ValueError('Incomplete fixed panel')
    summary=value_summary(all_games)
    scientific={'status':'passed','plan':plan,'input_hash':h,'games':len(all_games),'checkpoint_loads':loads,
        'optimizer_updates':0,'actor_decisions':actor_decisions,'evaluation':summary,
        'unavailable_choices':sum(g['unavailable_choices'] for g in all_games),
        'roles':{'training_teams_excluded':len(train),'prior_development_teams_excluded':len(past),
            'fresh_evaluation_teams':len(fresh),'disallowed_overlap':0},
        'seeds':seeds,'all_seed_battle_benefit':all(s['analysis']['battle_benefit'] for s in seeds.values()),
        'all_seed_shared_mastery':all(s['analysis']['familiar_mastery']['shared']['passed'] for s in seeds.values()),
        'promotion_authorized':False,'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    evidence('scientific.json',scientific);admission(run,manifest)
    # Recheck original predecessor bytes after the panel without repeating any scientific campaign.
    admit(plan)
    if original:admission(original,previous)
    report={**scientific,'elapsed_seconds':time.perf_counter()-begun,
        'collection_seconds':sum(x['collection_seconds'] for x in profiles),
        'peak_parent_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'profiles':profiles}
    atomic_json(run/'report.json',report)
    if original:
        atomic_json(run/'verification.json',{'status':'passed','input_hash':h,
            'report_sha256':digest(original/'report.json'),'scientific_sha256':digest(original/'scientific.json'),
            'games':6912,'checkpoint_loads':9,'actor_decisions':actor_decisions,
            'all_decisions':summary['decisions'],'optimizer_updates':0,
            'all_seed_battle_benefit':scientific['all_seed_battle_benefit']})
    print({'status':'passed','seconds':report['elapsed_seconds'],'games':6912,
        'all_seed_battle_benefit':scientific['all_seed_battle_benefit']},flush=True)
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--original',type=Path)
    args=a.parse_args();campaign(args.run.resolve(),args.original.resolve() if args.original else None)
