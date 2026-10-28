"""Seal the small M4 engineering canary with complete update replay and provenance."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_bounded import canary, checkpoint as cp, learner
from pokemon_learner.checkpoint import digest, file_hash, wire

REVIEW=ROOT/'artifacts/reviews/m4-bounded-value'
PRIOR=ROOT/'artifacts/reports/m3-switch-screen.json'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13,6,17,17,13]


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Existing sealed file differs: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(data)


def predecessor():
    prior=read(PRIOR);expected=learner.PLAN['predecessor_receipt']
    require(prior['content_hash']==expected==digest({k:v for k,v in prior.items() if k!='content_hash'}),'Predecessor identity')
    archive=ROOT/'artifacts/manifests'/('m3-switch-screen-'+expected)
    require((archive/'receipt.json').read_bytes()==PRIOR.read_bytes(),'Predecessor receipt archive')
    checks=0
    for kind,key in [('source','source_files'),('evidence','evidence_files')]:
        for name,sha in prior[key].items():
            for path in (ROOT/name,archive/kind/name):
                require(file_hash(path)==sha,'Predecessor drift: '+str(path));checks+=1
    return {'content_hash':expected,'live_and_archive_hashes':checks,'receipt_sha256':file_hash(PRIOR)}


def regression():
    folder=REVIEW/'regression';v=read(folder/'verification.json')
    require(v['status']=='passed' and v['sources_unchanged'] is True and v['source_files']==cp.sources(),'Source-bound regression')
    require(len(v['results'])==2,'Regression roster')
    logs=[]
    for row,name in zip(v['results'],('predecessor.log','bounded.log')):
        require(row['exit_code']==0 and row['log']==name and file_hash(folder/name)==row['sha256'],'Regression log binding')
        logs.append((folder/name).read_text())
    node=list(map(int,re.findall(r'ℹ pass (\d+)',logs[0])))
    python=list(map(int,re.findall(r'Ran (\d+) tests? in',logs[0])))
    bounded=list(map(int,re.findall(r'Ran (\d+) tests? in',logs[1])))
    require(node==[237,3] and python==SUITES and logs[0].count('\nOK\n')==len(SUITES),'Predecessor regression counts')
    require(bounded==[21] and logs[1].count('\nOK\n')==1,'Bounded-value regression counts')
    return {'node':sum(node),'python_predecessor':sum(python),'python_bounded':sum(bounded),'total':sum(node)+sum(python)+sum(bounded)}


def seal(directory):
    out=Path(directory).resolve()
    require(out==ROOT/'artifacts/runs/m4-bounded-value-v1','Production run path')
    prior=predecessor();tests=regression();audited=canary.audit(out)
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md','delegate/outcome.json'):
        require((REVIEW/name).is_file(),'Missing review evidence: '+name)
    outcome=read(REVIEW/'delegate/outcome.json')
    require(outcome['verdict']=='accepted' and outcome['evidence_sha256']==file_hash(REVIEW/'delegate-verification.md'),'Delegated reference verification')
    sources={**cp.sources(),'docs/M4_BOUNDED_VALUE.md':file_hash(ROOT/'docs/M4_BOUNDED_VALUE.md')}
    evidence={str(PRIOR.relative_to(ROOT)):file_hash(PRIOR)}
    for folder in (out,REVIEW):
        evidence.update({str(p.relative_to(ROOT)):file_hash(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    report=read(out/'report.json')
    receipt={'format':'bounded-value-receipt-1','status':'complete-engineering-canary',
             'run':str(out.relative_to(ROOT)),'predecessor':prior,'audit':audited,'report':report,'tests':tests,
             'source_files':sources,'evidence_files':evidence,'continuation':'unimplemented; cutoffs reset',
             'strength':'unqualified','sustained_qualification':'not run','full_g2':'open','full_g3':'open','full_g4':'open',
             'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    receipt['content_hash']=digest(receipt)
    archive=ROOT/'artifacts/manifests'/('m4-bounded-value-'+receipt['content_hash'])
    for kind,files in [('source',sources),('evidence',evidence)]:
        for name,sha in files.items():
            data=(ROOT/name).read_bytes();require(file_hash(ROOT/name)==sha,'Live file drift')
            write_once(archive/kind/name,data);require(file_hash(archive/kind/name)==sha,'Archive drift')
    require(cp.sources()==read(out/'admission.json')['source_files'],'Final source binding')
    write_once(archive/'receipt.json',wire(receipt)+b'\n')
    write_once(ROOT/'artifacts/reports/m4-bounded-value.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],'audit':audited,'tests':tests,'predecessor':prior}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();seal(args.directory)
