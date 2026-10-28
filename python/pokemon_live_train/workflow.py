"""Persistent, single-launch qualify -> fixed screen -> audit/seal workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import evidence as e, learner as l
from pokemon_switch_engine.scalar import require

REVIEW=l.ROOT/'artifacts/reviews/m7-live-training'
QUAL=l.ROOT/'artifacts/runs/m7-live-qualification-v1'
SCREEN=l.ROOT/'artifacts/runs/m7-live-screen-v1'
PYTHON=str(l.ROOT/'.venv/bin/python')


def command(command, label, out):
    start=e.utc();timer=time.perf_counter();log=REVIEW/(label+'.log')
    with log.open('xb') as stream:
        p=subprocess.run(command,cwd=l.ROOT,env={**os.environ,'PYTHONPATH':str(l.ROOT/'python')},stdout=stream,stderr=subprocess.STDOUT)
    record={'command':command,'exit_code':p.returncode,'started_at':start,'ended_at':e.utc(),
        'elapsed_seconds':time.perf_counter()-timer,'log_sha256':e.file_hash(log),
        'report_sha256':e.file_hash(out/'report.json') if (out/'report.json').exists() else None}
    e.once(REVIEW/(label+'-exit.json'),record)
    require(p.returncode==0,'Workflow command failed: '+label)
    return record


def main():
    # O_EXCL is the persistent claim, including failures. It is never removed or
    # reused automatically; a failed run must be investigated and preserved.
    e.once(REVIEW/'workflow-claim.json',{'pid':os.getpid(),'started_at':e.utc()})
    try:
        frozen=e.read(REVIEW/'preproduction.json');require(frozen['admission']==e.admission(),'Preproduction tree changed')
        for name,sha in frozen['review_and_regression'].items():require(e.file_hash(l.ROOT/name)==sha,'Preproduction evidence drift')
        require(not QUAL.exists() and not SCREEN.exists(),'Production output already exists')
        command([PYTHON,'-m','pokemon_live_train.qualify',str(QUAL)],'qualification',QUAL)
        command([PYTHON,'-m','pokemon_live_train.screen',str(SCREEN),str(QUAL)],'screen',SCREEN)
        # Completion docs precede the receipt; the fixed source tree stays intact.
        q=e.read(QUAL/'report.json');s=e.read(SCREEN/'report.json')
        doc=l.ROOT/'docs/M7_LIVE_TRAINING.md'
        text=doc.read_text();require('<!-- MEASURED RESULTS -->' in text,'Handoff results marker')
        rows=['\n## Measured results\n',f"Qualification passed: {q['elapsed_seconds']:.3f} seconds, {q['totals']['updates']:,} optimizer updates, "
            f"{q['totals']['games']:,} games, {q['totals']['decisions']:,} decisions; {q['totals']['terminal']:,} terminal and {q['totals']['cutoff']:,} cutoff games. "
            f"Peak process RSS {q['final_memory']['peak_process_rss_bytes']/2**20:.2f} MiB; no unexpected compilation.\n",
            f"The screen completed {s['training_games']:,} training games and {s['evaluation_games']:,} evaluation games. "
            f"Overall preregistered learning gate: {'passed' if s['learning_gate_passed'] else 'not passed'}.\n"]
        for record in s['records']:
            rows.append(f"Seed {record['seed']}: {record['gate_interpretation']}; comparisons: `{json.dumps(record['comparisons'],sort_keys=True)}`.\n")
        rows.append('Full G2/G3/G4 and public ladder strength remain open. Final reserved data, public servers and Prime were unused.\n')
        doc.write_text(text.split('<!-- MEASURED RESULTS -->')[0]+'<!-- MEASURED RESULTS -->\n'+''.join(rows))
        # Both sealer invocations produce logs outside the sealed review folder.
        for label in ('first','repeat'):
            log=l.ROOT/f'artifacts/reviews/m7-live-training-seal-{label}.log'
            with log.open('xb') as stream:
                p=subprocess.run([PYTHON,str(l.ROOT/'scripts/seal-live-training.py')],cwd=l.ROOT,
                    env={**os.environ,'PYTHONPATH':str(l.ROOT/'python')},stdout=stream,stderr=subprocess.STDOUT)
            require(p.returncode==0,'Sealer failed: '+label)
            if label=='first':receipt=(l.ROOT/'artifacts/reports/m7-live-training.json').read_bytes()
            else:require(receipt==(l.ROOT/'artifacts/reports/m7-live-training.json').read_bytes(),'Repeat receipt differs')
        e.once(l.ROOT/'artifacts/reviews/m7-live-training-completion.json',{'status':'completed','receipt_sha256':e.file_hash(l.ROOT/'artifacts/reports/m7-live-training.json'),'repeat_identical':True,'ended_at':e.utc()})
    except BaseException as error:
        e.once(l.ROOT/'artifacts/reviews/m7-live-training-workflow-failure.json',{'status':'failed','error':repr(error),'ended_at':e.utc()});raise


if __name__=='__main__':main()
