"""Frozen-budget engineering smoke, gated on the complete fresh reference campaign."""
import argparse
import json
from pathlib import Path
import resource
import runpy
import subprocess
import time
import numpy as np
from pokemon_learner.checkpoint import wire,file_hash
from . import learner,checkpoint,parity
from .scalar import require

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    with Path(path).open('xb') as stream:stream.write(wire(value)+b'\n')


def record(before,result):
    after,buffer,final,metrics=result;valid=np.asarray(buffer.valid);actions=np.asarray(buffer.actions)
    return {'update':int(after.version),'optimizer_step':int(after.step),'next_game':int(after.next_game),
        'decisions':int(valid.sum()),'terminal':int((np.asarray(final.winner)!=-1).sum()),
        'truncated':int((np.asarray(final.winner)==-1).sum()),
        'actions':[int(np.sum(actions[valid]==a)) for a in range(4)],
        'before':checkpoint.fingerprint(before),'after':checkpoint.fingerprint(after),
        'buffer':checkpoint.fingerprint(buffer),'final':checkpoint.fingerprint(final),
        'metrics_sha256':checkpoint.fingerprint(metrics),
        'metrics':{k:np.asarray(v).tolist() for k,v in metrics.items()}}


def train(directory):
    directory=Path(directory);out=directory/'learner';out.mkdir()
    saved=parity.read(directory/'reference/parity.json')
    require(saved['status']=='pass' and saved['fresh']['games']==learner.PLAN['fresh_games'],'Parity before training')
    start=time.monotonic();runs=[]
    for seed in learner.CONFIG['seeds']:
        folder=out/f'seed-{seed}';folder.mkdir();state=learner.start(seed);records=[];checks=[];checkpoints=[];rollback=[]
        for update in range(learner.CONFIG['updates']):
            if update in (0,32):
                file=f'checkpoint-{update:03d}.json';checkpoint.save(folder/file,state,seed)
                checkpoints.append({'update':update,'file':file,'sha256':file_hash(folder/file),'state':checkpoint.fingerprint(state)})
                if update==32:
                    for stage in ('collect','update'):
                        before=checkpoint.fingerprint(state)
                        try:checkpoint.transaction(state,seed,fault=stage)
                        except RuntimeError as error:require('Injected' in str(error),'Unexpected rollback error')
                        else:raise ValueError('Fault did not fail')
                        require(checkpoint.fingerprint(state)==before,'Rollback mutation')
                        rollback.append({'update':update,'stage':stage,'state':before})
            result=checkpoint.transaction(state,seed);row=record(state,result);records.append(row)
            if update in (0,32):
                # row comes from the original in-memory state; restore only afterward.
                restored=checkpoint.load(folder/f'checkpoint-{update:03d}.json',seed)
                repeated=record(restored,checkpoint.transaction(restored,seed))
                require(row==repeated,'Original-next-update replay')
                checks.append({'checkpoint_update':update,'original_next_update':row,'status':'exact'})
            state=result[0]
            (directory/'progress.json').write_bytes(wire({'seed':seed,'updates':int(state.version),'total_updates':64})+b'\n')
        file='checkpoint-064.json';checkpoint.save(folder/file,state,seed)
        require(checkpoint.fingerprint(checkpoint.load(folder/file,seed))==checkpoint.fingerprint(state),'Final checkpoint replay')
        checkpoints.append({'update':64,'file':file,'sha256':file_hash(folder/file),'state':checkpoint.fingerprint(state)})
        report={'seed':seed,'records':records,'checkpoints':checkpoints,'original_next_update':checks,'rollback':rollback}
        write(folder/'report.json',report);runs.append({'seed':seed,'file':f'seed-{seed}/report.json','sha256':file_hash(folder/'report.json')})
    report={'status':'complete','plan':learner.PLAN,'seeds':runs,'updates':192,'training_games':6144,
        'decisions':sum(sum(r['decisions'] for r in parity.read(out/run['file'])['records']) for run in runs),
        'original_next_update_audits':6,'rollback_audits':6,'checkpoint_load_audits':9,
        'elapsed_seconds':time.monotonic()-start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'parity_sha256':file_hash(directory/'reference/parity.json'),
        'public_server_used':False,'final_evaluation_used':False,'strength_evaluated':False,'prime_used':False}
    write(out/'report.json',report);return report


def main(directory):
    directory=Path(directory).resolve();require(directory.is_relative_to(ROOT/'artifacts/runs'),'Run path');directory.mkdir()
    sealer=runpy.run_path(str(ROOT/'scripts/seal-switch-engine.py'))
    admission=sealer['admission']();write(directory/'preregistration.json',{'plan':learner.PLAN,'admission':admission})
    try:
        with (directory/'capture.log').open('xb') as log:
            subprocess.run(['node','packages/switch-engine/reference.mjs',str(directory/'reference')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        oracle=json.loads(subprocess.check_output(['node','packages/switch-engine/audit.mjs',str(directory/'reference')],cwd=ROOT))
        write(directory/'reference/oracle-audit.json',oracle)
        parity.audit(directory/'reference')
        require(sealer['admission']()==admission,'Source/runtime changed before learning')
        report=train(directory)
        require(sealer['admission']()==admission,'Source/runtime changed during campaign')
        write(directory/'report.json',{'status':'complete','parity_sha256':file_hash(directory/'reference/parity.json'),
              'learner_sha256':file_hash(directory/'learner/report.json'),'admission':admission})
        print(json.dumps(report))
    except BaseException as error:
        write(directory/'failure.json',{'type':type(error).__name__,'message':str(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');main(parser.parse_args().directory)
