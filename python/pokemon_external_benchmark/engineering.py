"""One-game integration/timing probe. This is not a scientific benchmark."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from .transport import JsonWorker,ROOT


def run_game(output, *, run_seed=1027341901,episode_index=0,swap=False,external_side='p2',
             search_time_ms=100,fp_seed=1027341902,max_turns=240,learner=None,
             learner_uses_history=False,purpose='engineering integration and timing'):
    """Coordinator-only driver; learner receives only its own player projection.

    Return full engineering evidence, never aggregate/select policies. With no
    learner, the unchanged tactical policy is used for a functional smoke test.
    Callers must freeze scientific schedules/budgets separately before reuse.
    Set learner_uses_history=True to call learner(player,index,own_history).
    There is no exception-based signature guessing or fallback policy.
    """
    if type(learner_uses_history) is not bool or not isinstance(purpose,str) or not purpose.strip():
        raise ValueError('Explicit history callback mode and nonempty purpose required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    learner_side='p1' if external_side=='p2' else 'p2'
    record={'schema_version':'f34-external-game-v1','purpose':purpose,
        'run_seed':run_seed,'episode_index':episode_index,'swap':swap,'external_side':external_side,
        'fp_seed':fp_seed,'search_time_ms':search_time_ms,'parallelism':1,'search_threads':1,
        'max_turns':max_turns,'steps':[],'status':'running','outcome':None,
        'illegal_actions':{'p1':0,'p2':0},'rejected_actions':{'p1':0,'p2':0},
        'learner_uses_history':learner_uses_history}
    started=time.perf_counter()
    try:
        with JsonWorker(['node','packages/external-benchmark/worker.mjs'],output/'showdown.stderr') as node, \
             JsonWorker([str(ROOT/'.venv-f34-external/bin/python'),'-m','pokemon_external_benchmark.foul_play'],output/'foul-play.stderr') as fp:
            node.request('hello')
            fp.request('start',side=external_side,search_time_ms=search_time_ms,seed=fp_seed)
            bundle=node.request('start',run_seed=run_seed,episode_index=episode_index,swap=swap,
                external_side=external_side,tactical_side=learner_side if learner is None else None)
            record['provenance']=bundle['frame']['provenance']
            with (output/'transcript.jsonl').open('x') as transcript:
                for index in range(2000):
                    frame=bundle['frame']
                    transcript.write(json.dumps({'frame':frame,'external':bundle['external']})+'\n');transcript.flush()
                    if frame['outcome']:
                        record['outcome']=frame['outcome'];break
                    if all(frame['players'][s]['phase']=='move' and frame['players'][s]['turn']>=max_turns for s in ('p1','p2')):
                        bundle=node.request('cutoff');record['outcome']=bundle['frame']['outcome'];break
                    action_started=time.perf_counter()
                    external=fp.request('choose',input=bundle['external'])
                    decision_wall=time.perf_counter()-action_started
                    actions={};ids={}
                    for side in ('p1','p2'):
                        player=frame['players'][side]
                        if not any(player['mask']):continue
                        if side==external_side:action=external['action']
                        elif learner is None:action=frame['tactical_actions'][side]
                        elif learner_uses_history:action=learner(player,index,frame['observable_history'][side])
                        else:action=learner(player,index)
                        if type(action) is not int or not 0<=action<14 or not player['mask'][action]:
                            record['illegal_actions'][side]+=1
                            raise ValueError(f'Illegal {side} action {action}')
                        actions[side]=action;ids[side]=player['request_id']
                    step={'index':index,'turn':frame['players'][learner_side]['turn'],
                        'actions':actions,'request_ids':ids,'external':external,'external_wall_seconds':decision_wall,
                        'external_input_sha256':hashlib.sha256(json.dumps(bundle['external'],sort_keys=True,separators=(',',':')).encode()).hexdigest()}
                    record['steps'].append(step)
                    bundle=node.request('step',actions=actions,request_ids=ids)
                    step['submitted']=bundle['frame'].get('submitted',[])
                    for submitted in step['submitted']:
                        if not submitted['accepted']:record['rejected_actions'][submitted['side']]+=1
                else:raise RuntimeError('Engineering step cap exhausted')
        record['status']='completed'
    except BaseException as exc:
        record['status']='failed';record['error']=repr(exc)
        raise
    finally:
        record['seconds']=time.perf_counter()-started
        decisions=[s for s in record['steps'] if 'action' in s['external']]
        record['foul_play_decisions']=len(decisions)
        record['total_foul_play_seconds']=sum(s['external_wall_seconds'] for s in decisions)
        record['mean_foul_play_seconds']=sum(s['external_wall_seconds'] for s in decisions)/max(1,len(decisions))
        (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--search-time-ms',type=int,default=100)
    parser.add_argument('--episode-index',type=int,default=0)
    parser.add_argument('--external-side',choices=['p1','p2'],default='p2')
    parser.add_argument('--max-turns',type=int,default=240)
    args=parser.parse_args()
    result=run_game(args.output,search_time_ms=args.search_time_ms,episode_index=args.episode_index,
        external_side=args.external_side,max_turns=args.max_turns)
    print(json.dumps({k:v for k,v in result.items() if k!='steps'}))


if __name__=='__main__':main()
