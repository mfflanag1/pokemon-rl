"""Reconstruct counters, checkpoints, screen updates and evaluation tapes."""
from pathlib import Path
import json
import math
import jax
import numpy as np
from pokemon_live import policy
from pokemon_continuation import engine
from pokemon_switch_engine.scalar import require
from . import learner as l, checkpoint as cp, evidence as e, evaluate


def report(directory, format, status):
    out=Path(directory);r=e.read(out/'report.json');admitted=e.read(out/'admission.json')
    require(r['format']==format and r['status']==status,'Report format/status')
    require(r['content_hash']==e.digest({k:v for k,v in r.items() if k!='content_hash'}),'Report checksum')
    require(admitted==e.admission() and r['admission_hash']==e.digest(admitted),'Source/runtime/reservation binding')
    return out,r,admitted


def checkpoint(out, row, admitted, *, seed):
    path=out/row['file'];require(e.file_hash(path)==row['sha256'],'Checkpoint file hash')
    s=cp.load(path,seed,admitted)
    require(cp.fingerprint(s)==row['state'] and e.counters(s)==row['counters'],'Checkpoint counter/state binding')
    require(e.fingerprints(l.transaction(s,seed))==row['next'],'Original-next checkpoint replay')
    require(row['discarded_transactions']==2,'Checkpoint probe accounting')
    return s


def qualification(directory, require_production=True):
    out,r,a=report(directory,'live-training-qualification-1','passed')
    require(r['mode'] in ('qualification','preflight'),'Qualification mode')
    production=r['mode']=='qualification'
    require(not require_production or production,'Preflight cannot qualify production')
    require(r['two_hour_qualification'] is production,'Qualification claim')
    q=l.PLAN['qualification'];conf={**q,**({} if production else l.PLAN['preflight'])}
    require(conf['seconds']<=r['elapsed_seconds']<=conf['seconds']+q['maximum_host_gap_seconds'],'Measured duration')
    require(r['unexpected_compilations']==0 and r['warmup_compilations']>0 and
        r['compile_control']['first_call_compilation_events']>0 and r['compile_control']['same_shape_repeat_events']==0,'Compile evidence')
    require(0<=r['maximum_host_gap_seconds']<=q['maximum_host_gap_seconds'],'Host timing gap')
    for key in ('training_transaction_seconds','process_cpu_seconds','timed_replay_seconds'):
        require(math.isfinite(r[key]) and r[key]>0,'Runtime timing evidence')
    require(r['training_transaction_seconds']+r['timed_replay_seconds']<=r['elapsed_seconds']+.1,'Overlapping timing accounting')
    require(r['final_memory']['peak_process_rss_bytes']<=q['maximum_peak_rss_bytes'],'Final RSS')
    rows=[json.loads(line) for line in (out/'telemetry.jsonl').read_text().splitlines()]
    require(e.file_hash(out/'telemetry.jsonl')==r['telemetry_sha256'] and len(rows)==r['telemetry_rows'] and len(rows)>0,'Telemetry roster')
    previous=0.;prev_counts=None;rss=rows[0]['memory']['peak_process_rss_bytes']
    for i,row in enumerate(rows):
        t=row['totals'];c=row['counters'];elapsed=row['elapsed_seconds']
        require(previous<elapsed<=r['elapsed_seconds'] and elapsed>=(i+1)*conf['telemetry_seconds'],'Telemetry timing linkage')
        require(elapsed-(i+1)*conf['telemetry_seconds']<=q['maximum_host_gap_seconds'],'Missing telemetry interval')
        require(c['windows']-r['initial_counters']['windows']==t['windows'] and
            c['updates']-r['initial_counters']['updates']==t['updates'] and c['adam_steps']==2*c['updates'] and
            c['next_game']-r['initial_counters']['next_game']==t['games'],'Telemetry counter linkage')
        require(t['decisions']==sum(t['actions']) and t['windows']==t['updates']+t['empty_windows'] and
            0<=t['games']-t['terminal']-t['cutoff']<=l.BATCH,'Telemetry outcomes')
        require(c['next_game']==l.BATCH*(1+max(c['windows']-1,0)//5),'Telemetry game allocation')
        require(0<=t['value_min']<=t['value_max']<=1 and -2e-6<=t['return_min']<=t['return_max']<=1+2e-6 and
            all(math.isfinite(v) for v in t['decision_weighted_metrics'].values()),'Telemetry value/metric bounds')
        if prev_counts is not None: require(all(t[k]>=prev_counts[k] for k in ('windows','updates','games','decisions','terminal','cutoff')),'Telemetry monotonicity')
        require(row['unexpected_compilations']==0 and row['memory']['peak_process_rss_bytes']<=q['maximum_peak_rss_bytes'] and
            row['memory']['peak_process_rss_bytes']-rss<=q['maximum_rss_growth_bytes'],'Telemetry stability')
        previous=elapsed;prev_counts=t
    e.closed_counts(r['totals'],r['initial_counters'],r['final_counters'])
    require(all(r['totals'][k]>=prev_counts[k] for k in ('windows','updates','games','decisions','terminal','cutoff')),'Final totals precede telemetry')
    require(0<=r['totals']['value_min']<=r['totals']['value_max']<=1 and
        -2e-6<=r['totals']['return_min']<=r['totals']['return_max']<=1+2e-6,'Final value/target bounds')
    require(all(r['totals'][k]>0 for k in ('updates','post12_decisions','terminal_after12','history_over32_decisions')),'Nonvacuous continuation')
    require(r['final_memory']['peak_process_rss_bytes']-rss<=q['maximum_rss_growth_bytes'],'Final RSS growth')
    require(e.file_hash(out/'replays.json')==r['replays_sha256'],'Replay roster hash');replays=e.read(out/'replays.json')
    boundaries=list(range(conf['checkpoint_seconds'],conf['seconds'],conf['checkpoint_seconds']))
    expected=['start.json','post-refill.json']+[f'boundary-{i+2}.json' for i in range(len(boundaries))]+['final.json']
    require([x['file'] for x in replays]==expected and len(replays)==r['checkpoint_count'] and r['discarded_checkpoint_transactions']==2*len(replays),'Checkpoint roster')
    states=[checkpoint(out,x,a,seed=conf['seed']) for x in replays]
    require(int(states[0].windows)==q['warmup_windows'] and int(states[1].windows)==q['warmup_windows']+1 and int(states[-1].windows)%5==0,'Warmup/refill/final boundaries')
    warm=l.start(conf['seed']);totals=e.Totals()
    for _ in range(q['warmup_windows']):
        result=l.transaction(warm,conf['seed']);totals.add(e.window_summary(warm,result));warm=result[0]
    require(cp.fingerprint(warm)==cp.fingerprint(states[0])==r['initial_fingerprint'] and totals.document()==r['warmup'],'Warm-up identity')
    require(cp.fingerprint(l.transaction(warm,conf['seed'])[0])==cp.fingerprint(states[1]),'Post-refill identity')
    require(e.counters(states[0])==r['initial_counters'] and e.counters(states[-1])==r['final_counters'] and cp.fingerprint(states[-1])==r['final_fingerprint'],'Endpoint identity')
    return {'mode':r['mode'],'checkpoint_replays':len(replays),'telemetry_rows':len(rows),'totals':r['totals']}


def screen(directory, qualification_directory, require_production=True):
    from .screen import checkpoint_windows, settings
    out,r,a=report(directory,'live-training-screen-1','completed')
    require(r['mode'] in ('preflight','production'),'Screen mode');preflight=r['mode']=='preflight'
    require(not require_production or not preflight,'Preflight cannot pass production screen')
    qualification(qualification_directory,require_production=not preflight)
    plan=settings(preflight);checkpoints=checkpoint_windows(preflight)
    require(r['qualification_report_sha256']==e.file_hash(Path(qualification_directory)/'report.json'),'Screen qualification binding')
    require([record['seed'] for record in r['records']]==plan['seeds'],'Screen seed roster')
    frozen=policy.initialize(plan['frozen_policy_seed']);games=decisions=replayed=0
    for record in r['records']:
        seed=record['seed'];s=l.start(seed);initial=s;totals=e.Totals()
        require(len(record['trace'])==plan['windows_per_seed'],'Screen training budget')
        require([x['file'] for x in record['checkpoints']]==[f'checkpoint-{seed}-{w}.json' for w in checkpoints],'Screen checkpoint roster')
        saved={w:checkpoint(out,x,a,seed=seed) for w,x in zip(checkpoints,record['checkpoints'])}
        for window in range(plan['windows_per_seed']+1):
            if window in saved:require(cp.fingerprint(s)==cp.fingerprint(saved[window]),'Screen checkpoint trajectory linkage')
            if window==plan['windows_per_seed']:break
            result=l.transaction(s,seed);row=e.window_summary(s,result)
            require(record['trace'][window]=={**row,'fingerprints':e.fingerprints(result)},'Screen update replay')
            totals.add(row);s=result[0]
        require(totals.document()==record['training'],'Screen training totals')
        require(cp.fingerprint(initial.parameters)==record['initial_parameters'] and cp.fingerprint(s.parameters)==record['final_parameters'] and
            record['initial_parameters']!=record['final_parameters'],'Policy endpoint identity')
        computed=[]
        for label,endpoint in (('initial',initial),('final',s)):
            require([x['opponent'] for x in record[label]]==plan['opponents'],'Opponent roster')
            outcomes=[]
            for opponent,cell in enumerate(record[label]):
                expected=[(seat,start) for seat in (0,1) for start in range(0,plan['pairs_per_opponent'],l.BATCH)]
                require([(b['seat'],b['start']) for b in cell['batches']]==expected,'Evaluation batch roster')
                wins=np.full((plan['pairs_per_opponent'],2),-2,np.int32)
                for b in cell['batches']:
                    seat,start=b['seat'],b['start'];teams,ids=evaluate.schedule(seed,opponent,start,seat)
                    result=evaluate.game_batch(endpoint.parameters,frozen,teams,ids,np.uint32(plan['evaluation_seed']+seed),np.int32(seat),np.int32(opponent))
                    jax.block_until_ready(result);final,tape=result
                    evaluate.audit_tape(final,tape,teams,ids,plan['evaluation_seed']+seed)
                    expected_name=f'eval-{seed}-{label}-{cell["opponent"]}-{seat}-{start}.npz'
                    require(b['file']==expected_name and e.file_hash(out/b['file'])==b['sha256'],'Evaluation tape identity')
                    with np.load(out/b['file'],allow_pickle=False) as data:
                        loaded=(engine.State(*[jnp_array(data['final_'+k]) for k in engine.State._fields]),
                            tuple(jnp_array(data[k]) for k in ('actions','values','logps','valid','turns')))
                    require(cp.fingerprint(loaded)==cp.fingerprint(result)==b['fingerprint'],'Every evaluation decision/value/outcome replay')
                    require(evaluate.diagnostics(final,tape,seat)==b['diagnostics'],'Evaluation value diagnostics')
                    winner=np.asarray(final.winner);wins[start:start+l.BATCH,seat]=np.where(winner==-1,-1,winner==seat)
                    replayed+=l.BATCH
                require(wins.tolist()==cell['outcomes'],'Evaluation outcome linkage');outcomes.append(wins)
            computed.append(outcomes)
        cells={name:evaluate.comparison(computed[1][i],computed[0][i]) for i,name in enumerate(plan['opponents'])}
        require(cells==record['comparisons'] and record['learning_gate_passed']==evaluate.gate(cells) and
            record['gate_interpretation']==evaluate.interpretation(cells) and record['evaluation_parameters_unchanged'] is True,'Learning gate computation')
        games+=totals.counts['games'];decisions+=totals.counts['decisions']
    require(games==r['training_games']==len(plan['seeds'])*plan['windows_per_seed']//5*l.BATCH and decisions==r['training_decisions'],'Screen training count')
    require(replayed==r['evaluation_games']==len(plan['seeds'])*2*len(plan['opponents'])*2*plan['pairs_per_opponent'],'Screen evaluation count')
    require(r['learning_gate_passed']==all(x['learning_gate_passed'] for x in r['records']),'Overall learning gate')
    return {'training_games':games,'training_decisions':decisions,'evaluation_games_replayed':replayed,'learning_gate_passed':r['learning_gate_passed']}


def jnp_array(a):
    import jax.numpy as jnp
    return jnp.asarray(a)
