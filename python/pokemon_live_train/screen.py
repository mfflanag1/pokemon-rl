"""Fixed three-seed learning screen, with initialization and one final endpoint."""
import argparse
from pathlib import Path
import time
import traceback
import jax
import numpy as np
from pokemon_live import policy
from pokemon_switch_engine.scalar import require
from . import learner as l, checkpoint as cp, evidence as e, evaluate

CHECKPOINTS = (0, 1, 2, 5, 250, 500)


def settings(preflight):
    return {**l.PLAN['screen'], **(l.PLAN['screen_preflight'] if preflight else {})}


def checkpoint_windows(preflight):return (0,1,2,5,10) if preflight else CHECKPOINTS


def evaluate_endpoint(out, state, endpoint, frozen, admitted, *, pairs=None, scalar=True):
    plan=l.PLAN['screen']; pairs=plan['pairs_per_opponent'] if pairs is None else pairs
    result=[]
    for opponent, name in enumerate(plan['opponents']):
        outcomes=np.full((pairs,2),-2,np.int32); batches=[]
        for seat in (0,1):
            for start in range(0,pairs,l.BATCH):
                teams,ids=evaluate.schedule(int(state.seed),opponent,start,seat)
                master=plan['evaluation_seed']+int(state.seed)
                final,tape=evaluate.game_batch(state.parameters,frozen,teams,ids,np.uint32(master),np.int32(seat),np.int32(opponent))
                jax.block_until_ready((final,tape))
                if scalar: evaluate.audit_tape(final,tape,teams,ids,master)
                require(np.all((np.asarray(final.winner)!=-1) | (np.asarray(final.turn)>=61)), 'Unfinished evaluation')
                winner=np.asarray(final.winner); values=np.where(winner==-1,-1,(winner==seat).astype(np.int32))
                outcomes[start:start+l.BATCH,seat]=values
                filename=f'eval-{int(state.seed)}-{endpoint}-{name}-{seat}-{start}.npz'
                with (out/filename).open('xb') as stream:
                    np.savez_compressed(stream,**{'final_'+k:np.asarray(v) for k,v in final._asdict().items()},
                        **{k:np.asarray(v) for k,v in zip(('actions','values','logps','valid','turns'),tape)})
                batches.append({'file':filename,'sha256':e.file_hash(out/filename),'seat':seat,'start':start,
                    'diagnostics':evaluate.diagnostics(final,tape,seat),'fingerprint':cp.fingerprint((final,tape))})
                e.write(out/'progress.json',{'phase':'evaluation','seed':int(state.seed),'endpoint':endpoint,
                    'opponent':name,'seat':seat,'pairs_finished':start+l.BATCH})
        require(np.isin(outcomes,(-1,0,1)).all(),'Evaluation roster incomplete')
        result.append({'opponent':name,'outcomes':outcomes.tolist(),'batches':batches})
    return result


def run(directory, qualification, preflight=False):
    from .verify import qualification as audit_qualification
    audit_qualification(Path(qualification),require_production=not preflight)
    out=Path(directory).resolve();require(out.is_relative_to(l.ROOT/'artifacts/runs'),'Run directory');out.mkdir(exist_ok=False)
    admitted=e.admission();e.once(out/'admission.json',admitted);plan=settings(preflight);beg=time.perf_counter()
    result={'format':'live-training-screen-1','status':'running','learning_gate_passed':False,
        'mode':'preflight' if preflight else 'production','strength_claim':False,
        'started_at':e.utc(),'admission_hash':e.digest(admitted),'qualification_report_sha256':e.file_hash(Path(qualification)/'report.json'),
        'records':[],**l.PLAN['scope']}
    e.write(out/'report.json',result);frozen=policy.initialize(plan['frozen_policy_seed'])
    try:
        for seed in plan['seeds']:
            s=l.start(seed);initial=s;totals=e.Totals();trace=[];checkpoints=[]
            for window in range(plan['windows_per_seed']+1):
                if window in checkpoint_windows(preflight): checkpoints.append(e.replay(out,f'checkpoint-{seed}-{window}.json',s,admitted,scalar=window in (0,plan['windows_per_seed'])))
                if window==plan['windows_per_seed']:break
                r=l.transaction(s,seed);summary=e.window_summary(s,r);totals.add(summary)
                trace.append({**summary,'fingerprints':e.fingerprints(r)});s=r[0]
                if int(s.windows)%25==0:e.write(out/'progress.json',{'phase':'training','seed':seed,'window':int(s.windows),'totals':totals.document()})
            # Initial next_game already includes the first cohort; screen begins at
            # window zero, unlike qualification after its closed warm-up cohort.
            initial_counts=e.counters(initial);initial_counts['next_game']=0
            e.closed_counts(totals,initial_counts,e.counters(s))
            require(cp.fingerprint(initial.parameters)!=cp.fingerprint(s.parameters),'Parameters did not change')
            before_evaluation=cp.fingerprint((initial,s,frozen))
            original=evaluate_endpoint(out,initial,'initial',frozen,admitted,pairs=plan['pairs_per_opponent'])
            final=evaluate_endpoint(out,s,'final',frozen,admitted,pairs=plan['pairs_per_opponent'])
            require(cp.fingerprint((initial,s,frozen))==before_evaluation,'Evaluation changed training/frozen parameters')
            cells={a['opponent']:evaluate.comparison(b['outcomes'],a['outcomes']) for a,b in zip(original,final)}
            record={'seed':seed,'training':totals.document(),'trace':trace,'checkpoints':checkpoints,
                'initial_parameters':cp.fingerprint(initial.parameters),'final_parameters':cp.fingerprint(s.parameters),
                'initial':original,'final':final,'comparisons':cells,'learning_gate_passed':evaluate.gate(cells),
                'gate_interpretation':evaluate.interpretation(cells),'evaluation_parameters_unchanged':True}
            result['records'].append(record);e.write(out/f'seed-{seed}.json',record)
        require(e.admission()==admitted,'Source/runtime/reservation drift')
        result.update(status='completed',learning_gate_passed=all(r['learning_gate_passed'] for r in result['records']),
            training_games=sum(r['training']['games'] for r in result['records']),
            training_decisions=sum(r['training']['decisions'] for r in result['records']),
            evaluation_games=len(plan['seeds'])*2*len(plan['opponents'])*plan['pairs_per_opponent']*2,
            elapsed_seconds=time.perf_counter()-beg)
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['ended_at']=e.utc();result['content_hash']=e.digest(result);e.write(out/'report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('qualification');p.add_argument('--preflight',action='store_true');a=p.parse_args();run(a.directory,a.qualification,a.preflight)
