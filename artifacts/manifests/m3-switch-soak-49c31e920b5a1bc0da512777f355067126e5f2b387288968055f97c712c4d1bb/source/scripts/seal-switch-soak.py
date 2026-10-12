"""Fresh checkpoint audit and immutable completion receipt for M3 CPU qualification."""
import argparse
import json
from pathlib import Path
import re
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_switch_engine import learner
from pokemon_switch_engine.scalar import require
from pokemon_switch_soak import checkpoint
from pokemon_switch_soak.provenance import PLAN,PREDECESSOR,admission,binding,read
from pokemon_switch_soak.gates import validate

REVIEW=ROOT/'artifacts/reviews/m3-switch-soak'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13,6,17,17]
write_once=runpy.run_path(str(ROOT/'scripts/seal-switch-engine.py'))['write_once']


def check_boundary(item,rows,summary):
    """The independently restored final checkpoint brackets a missing final tick."""
    before=[r for r in rows if r['elapsed_seconds']<=item['elapsed_seconds']]
    after=[r for r in rows if r['elapsed_seconds']>item['elapsed_seconds']]
    upper=after[0]['counters'] if after else summary['final_counters']
    require(item['elapsed_seconds']<=summary['elapsed_seconds'] and before
        and before[-1]['counters']['updates']<=item['counters']['updates']<upper['updates'],'Checkpoint/telemetry bracket')
    if before[-1]['elapsed_seconds']==item['elapsed_seconds']:
        require(before[-1]['counters']==item['counters'],'Same-cycle telemetry checkpoint linkage')


def audit(directory,*,smoke=False):
    directory=Path(directory).resolve();require(directory.is_relative_to(ROOT/'artifacts/runs'),'Run path')
    require(not list(directory.glob('*.partial')) and not (directory/'failure.json').exists(),'Incomplete evidence')
    report=read(directory/'report.json');admitted=read(directory/'admission.json')
    require(report['content_hash']==digest({k:v for k,v in report.items() if k!='content_hash'}),'Report checksum')
    require(report['mode']==('smoke' if smoke else 'qualification'),'Wrong campaign mode')
    require(admitted==admission() and digest(admitted)==report['admission_hash'],'Source/runtime admission')
    rows=[json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()]
    validate(report,PLAN,rows);require(read(directory/'progress.json')==rows[-1],'Progress linkage')
    require(file_hash(directory/'telemetry.jsonl')==report['telemetry_sha256'] and file_hash(directory/'replays.json')==report['replays_sha256'],'Evidence checksums')
    seconds=PLAN['smoke']['seconds'] if smoke else PLAN['qualification_seconds']
    interval=PLAN['smoke']['checkpoint_seconds'] if smoke else PLAN['checkpoint_seconds']
    schedule=[0,*range(interval,seconds,interval),seconds]
    names=['start.json',*[f'boundary-{i:02d}.json' for i in range(1,len(schedule)-1)],'final.json']
    replays=read(directory/'replays.json');require([r['file'] for r in replays]==names and len(replays)==report['soak']['replays'],'Replay roster')
    bind=binding(admitted,report['mode']);boundaries=[]
    expected_start=learner.start(PLAN['seed'])
    for _ in range(PLAN['warmup_cycles']):expected_start=checkpoint.transaction(expected_start)[0]
    for i,(item,name,scheduled) in enumerate(zip(replays,names,schedule)):
        require(item['scheduled_seconds']==scheduled and item['original_vs_restored_exact'] is True,'Replay schedule/status')
        require(scheduled<=item['elapsed_seconds']<=scheduled+PLAN['maximum_host_gap_seconds'],'Replay schedule delay')
        path=directory/name;require(file_hash(path)==item['sha256'],'Checkpoint bytes')
        state=checkpoint.load(path,bind);c=checkpoint.counters(state);boundaries.append(c)
        require(c==item['counters'] and checkpoint.fingerprint(state)==item['state'],'Checkpoint counters/state')
        require(checkpoint.result_fingerprints(checkpoint.transaction(state))==item['next'],'Original next-update fingerprints')
        if i==0:require(checkpoint.fingerprint(state)==checkpoint.fingerprint(expected_start),'Fresh initialization/warm-up identity')
        elif i<len(replays)-1:
            check_boundary(item,rows,report['soak'])
    s=report['soak'];require(boundaries[0]==s['initial_counters'] and boundaries[-1]==s['final_counters'],'Endpoint linkage')
    require(all(a['updates']<b['updates'] for a,b in zip(boundaries,boundaries[1:])),'Checkpoint order')
    require(replays[-1]['elapsed_seconds']==s['elapsed_seconds'],'Final replay timing linkage')
    return {'status':'verified','mode':report['mode'],'two_hour_qualification':not smoke,'replays':len(replays),
        'report_sha256':file_hash(directory/'report.json'),'admission_hash':digest(admitted)}


def seal(directory):
    directory=Path(directory).resolve();audited=audit(directory);report=read(directory/'report.json');now=admission()
    v=read(REVIEW/'verification.json');log=(REVIEW/'npm-test.log').read_text()
    require(v['status']=='passed' and v['exit_code']==0 and v['sources_unchanged'] is True and v['source_files']==now['source_files']
        and v['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Source-bound regression')
    node=re.search(r'ℹ pass (\d+)',log);counts=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(node and int(node[1])==237 and counts==SUITES and log.count('\nOK\n')==len(SUITES),'Regression coverage')
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md','delegate/outcome.json','preflight.json'):
        require((REVIEW/name).is_file(),'Missing reviewed preflight: '+name)
    outcome=read(REVIEW/'delegate/outcome.json');require(outcome['verdict']=='accepted'
        and outcome['evidence_sha256']==file_hash(REVIEW/'delegate-verification.md'),'Delegation verification')
    preflight=read(REVIEW/'preflight.json');preflight_run=ROOT/preflight['run']
    require(preflight_run.resolve().is_relative_to(ROOT/'artifacts/runs') and preflight['audit']==audit(preflight_run,smoke=True),'Preflight evidence')
    sources={**now['source_files'],**{str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]}}
    evidence={'artifacts/reports/m3-switch-engine.json':file_hash(ROOT/'artifacts/reports/m3-switch-engine.json')}
    for folder in (directory,REVIEW,*sorted((ROOT/'artifacts/runs').glob('m3-switch-soak-smoke-*'))):
        evidence.update({str(p.relative_to(ROOT)):file_hash(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    receipt={'format':'switch-soak-receipt-1','status':'complete-restricted-cpu-qualification','run':str(directory.relative_to(ROOT)),
        'predecessor':PREDECESSOR,'audit':audited,'report':report,'tests':{'node':237,'python':sum(counts),'suite_counts':counts},
        'source_files':sources,'evidence_files':evidence,'two_hour_qualification':True,
        **{k:PLAN[k] for k in ('full_g2','full_g3','full_g4','public_ladder_strength','strength_evaluation','final_evaluation_used','public_server_used','prime_used')}}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m3-switch-soak-'+receipt['content_hash'])
    for kind,files in [('source',sources),('evidence',evidence)]:
        for name,h in files.items():
            require(file_hash(ROOT/name)==h,'Live archive input drift');write_once(archive/kind/name,(ROOT/name).read_bytes())
            require(file_hash(archive/kind/name)==h,'Archive output drift')
    write_once(archive/'receipt.json',wire(receipt)+b'\n');write_once(ROOT/'artifacts/reports/m3-switch-soak.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'hash':receipt['content_hash'],'soak':report['soak'],'tests':receipt['tests']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--audit-smoke',action='store_true');args=p.parse_args()
    if args.audit_smoke:print(json.dumps(audit(args.directory,smoke=True)))
    else:seal(args.directory)
