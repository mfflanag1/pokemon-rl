"""Three-seed fixed CPU canary; full trace replay, not a strength screen."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pokemon_learner.checkpoint import digest, wire, file_hash
from . import learner, checkpoint as cp


def write(path, value):
    with Path(path).open('xb') as f:
        f.write(wire(value) + b'\n')


def summary(result):
    state, buffer, final, metrics = result
    valid = np.asarray(buffer.valid); terminal = np.asarray(buffer.terminal)
    return {'version': int(state.version), 'next_game': int(state.next_game),
            'decisions': int(valid.sum()), 'games': learner.BATCH,
            'terminal_games': int((np.asarray(final.winner) != -1).sum()),
            'truncated_games': int((np.asarray(final.winner) == -1).sum()),
            'value_min': float(np.asarray(buffer.values)[valid].min()),
            'value_max': float(np.asarray(buffer.values)[valid].max()),
            'return_min': float(np.asarray(buffer.returns)[valid].min()),
            'return_max': float(np.asarray(buffer.returns)[valid].max()),
            'bootstrap_min': float(np.asarray(buffer.bootstrap).min()),
            'bootstrap_max': float(np.asarray(buffer.bootstrap).max()),
            'reward_sum': float(np.asarray(buffer.rewards).sum()),
            'terminal_rows': int(terminal.sum()),
            'metrics': {k: np.asarray(v).tolist() for k,v in metrics.items()},
            'fingerprints': {k: cp.fingerprint(v) for k,v in zip(('learner','buffer','final_state','metrics'),result)}}


def audit_checkpoint(out, state, seed, admitted, label):
    path = out / f'checkpoint-{seed}-{label}.json'
    cp.save(path,state,seed,admitted)
    original = summary(learner.transaction(state,seed))
    resumed = summary(learner.transaction(cp.load(path,seed,admitted),seed))
    if original != resumed:
        raise ValueError('Original next-update mismatch')
    return {'file':path.name,'sha256':file_hash(path),'state':cp.fingerprint(state),'original_next':original}


def run(directory):
    out=Path(directory).resolve()
    if not out.is_relative_to(learner.ROOT/'artifacts/runs'):
        raise ValueError('Canary output must be under artifacts/runs')
    out.mkdir(exist_ok=False)
    admitted=cp.admission();write(out/'admission.json',admitted)
    started=time.perf_counter();records=[]
    try:
        for seed in learner.PLAN['seeds']:
            state=learner.start(seed)
            initial=audit_checkpoint(out,state,seed,admitted,'initial');trace=[]
            for _ in range(learner.PLAN['updates_per_seed']):
                result=learner.transaction(state,seed);trace.append(summary(result));state=result[0]
            final=audit_checkpoint(out,state,seed,admitted,'final')
            records.append({'seed':seed,'initial':initial,'trace':trace,'final':final})
        if cp.admission()!=admitted:
            raise ValueError('Source/runtime drift')
        report={'format':'bounded-value-canary-1','status':'passed-engineering-canary','admission_hash':digest(admitted),
                'records':records,'elapsed_seconds':time.perf_counter()-started,
                'training_updates':sum(len(r['trace']) for r in records),
                'training_games':sum(x['games'] for r in records for x in r['trace']),
                'training_decisions':sum(x['decisions'] for r in records for x in r['trace']),
                'terminal_games':sum(x['terminal_games'] for r in records for x in r['trace']),
                'truncated_games':sum(x['truncated_games'] for r in records for x in r['trace']),
                'checkpoint_pairs':2*len(records),'discarded_checkpoint_audit_transactions':4*len(records),
                'continuation':'unimplemented; cutoffs reset','strength':'unqualified','sustained_qualification':'not run',
                'full_g2':'open','full_g3':'open','full_g4':'open','final_reserved_data_used':False,
                'public_server_used':False,'prime_used':False}
        report['content_hash']=digest(report);write(out/'report.json',report)
    except BaseException as error:
        write(out/'failure.json',{'error':repr(error)});raise
    print(json.dumps({k:report[k] for k in ('status','training_updates','training_games','training_decisions','terminal_games','truncated_games','checkpoint_pairs')}))
    return report


def audit(directory):
    out=Path(directory);report=json.loads((out/'report.json').read_text());admitted=json.loads((out/'admission.json').read_text())
    if (out/'failure.json').exists() or report['status']!='passed-engineering-canary' or report['format']!='bounded-value-canary-1':
        raise ValueError('Failed canary')
    if digest({k:v for k,v in report.items() if k!='content_hash'})!=report['content_hash']:
        raise ValueError('Report digest')
    if cp.admission()!=admitted or digest(admitted)!=report['admission_hash']:
        raise ValueError('Admission binding')
    if [r['seed'] for r in report['records']]!=learner.PLAN['seeds']:
        raise ValueError('Seed roster')
    for record in report['records']:
        seed=record['seed']
        if any(record[label]['file'] != f'checkpoint-{seed}-{label}.json' for label in ('initial','final')):
            raise ValueError('Checkpoint filename roster')
        initial=cp.load(out/record['initial']['file'],seed,admitted)
        if cp.fingerprint(initial)!=cp.fingerprint(learner.start(seed)):
            raise ValueError('Fresh initialization')
        state=initial
        if len(record['trace'])!=learner.PLAN['updates_per_seed']:
            raise ValueError('Update budget')
        for row in record['trace']:
            result=learner.transaction(state,seed)
            if summary(result)!=row:
                raise ValueError('Full update trace replay mismatch')
            state=result[0]
        final=cp.load(out/record['final']['file'],seed,admitted)
        if cp.fingerprint(state)!=cp.fingerprint(final):
            raise ValueError('Final checkpoint differs from complete trajectory')
        for key, state in [('initial',initial),('final',final)]:
            evidence=record[key]
            if file_hash(out/evidence['file'])!=evidence['sha256'] or cp.fingerprint(state)!=evidence['state']:
                raise ValueError('Checkpoint file/state hash')
            if summary(learner.transaction(state,seed))!=evidence['original_next']:
                raise ValueError('Original next-update replay mismatch')
    totals={'training_updates':sum(len(r['trace']) for r in report['records'])}
    for total,field in [('training_games','games'),('training_decisions','decisions'),('terminal_games','terminal_games'),('truncated_games','truncated_games')]:
        totals[total]=sum(x[field] for r in report['records'] for x in r['trace'])
    if any(report[k]!=v for k,v in totals.items()) or report['checkpoint_pairs']!=6 or report['discarded_checkpoint_audit_transactions']!=12:
        raise ValueError('Counter accounting')
    for field,expected in [('continuation','unimplemented; cutoffs reset'),('strength','unqualified'),('sustained_qualification','not run'),
                           ('full_g2','open'),('full_g3','open'),('full_g4','open'),('final_reserved_data_used',False),
                           ('public_server_used',False),('prime_used',False)]:
        if report[field]!=expected:raise ValueError('Scope flag: '+field)
    if cp.admission()!=admitted:raise ValueError('Source/runtime drift during audit')
    return {'status':'passed','replayed_updates':totals['training_updates'],'checkpoint_pairs':6,**totals}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--audit',action='store_true');args=p.parse_args()
    if args.audit:print(json.dumps(audit(args.directory)))
    else:run(args.directory)
