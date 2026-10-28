"""Replay every M6 window/checkpoint, bind regression/reviews, preserve predecessors."""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_live import checkpoint as cp, canary, learner
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_switch_engine.scalar import require

REVIEW=ROOT/'artifacts/reviews/m6-live-learner'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13,6,17,17,13]


def read(path):return json.loads(Path(path).read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Immutable output differs: '+str(path))
    else:
        with path.open('xb') as f:f.write(data)


def predecessors():
    results=[]
    for name in ('m3-switch-screen','m4-bounded-value','m5-continuation'):
        path=ROOT/'artifacts/reports'/(name+'.json');r=read(path);h=r['content_hash']
        require(h==digest({k:v for k,v in r.items() if k!='content_hash'}),'Predecessor checksum')
        if name=='m5-continuation':require(h==learner.PLAN['predecessor'],'Predecessor identity')
        archive=ROOT/'artifacts/manifests'/(name+'-'+h);checks=0
        require((archive/'receipt.json').read_bytes()==path.read_bytes(),'Receipt archive')
        for kind,key in [('source','source_files'),('evidence','evidence_files')]:
            for filename,sha in r[key].items():
                for p in (ROOT/filename,archive/kind/filename):
                    require(file_hash(p)==sha,'Predecessor drift: '+str(p));checks+=1
        results.append({'name':name,'content_hash':h,'receipt_sha256':file_hash(path),'hash_checks':checks})
    return results


def regression():
    folder=REVIEW/'regression';v=read(folder/'verification.json')
    require(v['status']=='passed' and v['sources_unchanged'] is True and v['source_files']==cp.sources(),'Source-bound regression')
    require(len(v['results'])==4,'Regression roster');logs=[]
    for row,name in zip(v['results'],('predecessor.log','bounded.log','continuation.log','live.log')):
        require(row['exit_code']==0 and row['log']==name and row['sha256']==file_hash(folder/name),'Regression log')
        logs.append((folder/name).read_text())
    node=list(map(int,re.findall(r'ℹ pass (\d+)',logs[0])))
    counts=[list(map(int,re.findall(r'Ran (\d+) tests? in',log))) for log in logs]
    require(node==[237,3] and counts==[SUITES,[21],[13],[17]],'Regression coverage')
    require([log.count('\nOK\n') for log in logs]==[len(SUITES),1,1,1],'Regression success markers')
    return {'node':240,'python_predecessor':236,'python_bounded':21,'python_continuation':13,'python_live':17,'total':527}


def review_gate(folder, expected_sources):
    review=read(folder/'review-verification.json')
    require(set(review)=={'format','review_family','cross_family_completed','blocking_findings_open','tests_verified',
        'reviewed_source_files','dispositions_sha256','request_sha256','result_sha256','status_sha256',
        'unavailable_status_sha256'},'Review verification schema')
    require(review['format']=='live-learner-review-1' and review['review_family']=='codex'
            and review['cross_family_completed'] is False,'Honest review identity')
    require(type(review['blocking_findings_open']) is int and review['blocking_findings_open']==0
            and review['tests_verified'] is True,'Review dispositions')
    require(bool(expected_sources) and review['reviewed_source_files']==expected_sources,'Exact reviewed source roster')
    for field,name in [('dispositions_sha256','review-dispositions.md'),('request_sha256','implementation-final/request.json'),
                       ('result_sha256','implementation-final/result.md'),('status_sha256','implementation-final/status.json'),
                       ('unavailable_status_sha256','claude-fallback-unavailable/status.json')]:
        require(file_hash(folder/name)==review[field],'Review evidence binding: '+name)
    request=read(folder/'implementation-final/request.json');status=read(folder/'implementation-final/status.json')
    require(request['engine']=='codex' and request['mode']=='review' and status['status']=='completed','Completed independent review')
    require(read(folder/'claude-fallback-unavailable/status.json')['status']=='failed','Unavailable cross-family review evidence')
    require(all(request['files_sha256'].get(k)==v for k,v in expected_sources.items()),'Independent packet coverage')
    return review


def exit_gate(folder, out, admitted):
    record=read(folder/'exit.json')
    require(set(record)=={'format','command','run','admission_hash','exit_code','started_at','finished_at',
                         'elapsed_seconds','report_sha256','log_sha256'},'Launcher receipt schema')
    require(record['format']=='live-learner-launch-1' and type(record['exit_code']) is int and record['exit_code']==0,'Launcher exit')
    require(record['command']==[str(ROOT/'.venv/bin/python'),'-m','pokemon_live.canary',str(out)]
            and record['run']==str(out) and record['admission_hash']==digest(admitted),'Launcher command/admission')
    require(record['report_sha256']==file_hash(out/'report.json') and record['log_sha256']==file_hash(folder/'production.log'),'Launcher output hashes')
    from datetime import datetime
    start,end=map(datetime.fromisoformat,(record['started_at'],record['finished_at']))
    require(start.tzinfo is not None and end.tzinfo is not None and end>=start
            and type(record['elapsed_seconds']) in (int,float) and 0<record['elapsed_seconds']<3600,'Launcher timing schema')
    require(abs((end-start).total_seconds()-record['elapsed_seconds'])<5,'Launcher clock linkage')
    return record


def seal(directory):
    out=Path(directory).resolve();require(out==ROOT/'artifacts/runs/m6-live-learner-v1','Production directory')
    prior=predecessors();tests=regression();admitted=read(out/'admission.json')
    require(admitted==cp.admission()==read(REVIEW/'preproduction.json')['admission'],'Production source/runtime freeze')
    launch=exit_gate(REVIEW,out,admitted)
    require(not (out/'failure.json').exists(),'Production failure evidence')
    audited=canary.audit(out)
    required={name:sha for name,sha in cp.sources().items() if name.startswith(('python/pokemon_live/', 'contracts/live-learner-v1/', 'tests/live-learner/')) or name in ('scripts/seal-live-learner.py','scripts/verify-live-learner.py')}
    review=review_gate(REVIEW,required)
    source={**cp.sources(),'docs/M6_LIVE_LEARNER.md':file_hash(ROOT/'docs/M6_LIVE_LEARNER.md')}
    evidence={str((ROOT/'artifacts/reports'/(p['name']+'.json')).relative_to(ROOT)):p['receipt_sha256'] for p in prior}
    for folder in (out,REVIEW,*sorted((ROOT/'artifacts/runs').glob('m6-live-learner-preflight-*'))):
        evidence.update({str(p.relative_to(ROOT)):file_hash(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    receipt={'format':'live-learner-receipt-1','status':'complete-live-learner-canary',
             'run':str(out.relative_to(ROOT)),'predecessors':prior,'audit':audited,'tests':tests,'review':review,'launch':launch,
             'source_files':source,'evidence_files':evidence,'report_sha256':file_hash(out/'report.json'),**canary.SCOPE}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m6-live-learner-'+receipt['content_hash'])
    for kind,files in [('source',source),('evidence',evidence)]:
        for name,sha in files.items():
            require(file_hash(ROOT/name)==sha,'Live drift');write_once(archive/kind/name,(ROOT/name).read_bytes())
            require(file_hash(archive/kind/name)==sha,'Archive drift')
    require(cp.admission()==admitted,'Final source/runtime binding')
    write_once(archive/'receipt.json',wire(receipt)+b'\n');write_once(ROOT/'artifacts/reports/m6-live-learner.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],'audit':audited,'tests':tests}))

if __name__=='__main__':seal(sys.argv[1])
