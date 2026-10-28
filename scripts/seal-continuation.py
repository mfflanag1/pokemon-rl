"""Replay the complete bounded continuation catalog and seal immutable evidence."""
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_continuation import carry,parity
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_switch_engine.scalar import require

REVIEW=ROOT/'artifacts/reviews/m5-continuation'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13,6,17,17,13]


def read(path):return json.loads(Path(path).read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Immutable output differs: '+str(path))
    else:
        with path.open('xb') as f:f.write(data)


def predecessors():
    results=[]
    for name in ('m3-switch-screen','m4-bounded-value'):
        path=ROOT/'artifacts/reports'/(name+'.json');r=read(path);h=r['content_hash']
        require(h==digest({k:v for k,v in r.items() if k!='content_hash'}),'Predecessor checksum')
        if name=='m4-bounded-value':require(h==carry.PLAN['predecessor'],'Predecessor identity')
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
    require(v['status']=='passed' and v['sources_unchanged'] is True and v['source_files']==carry.sources(),'Source-bound regression')
    require(len(v['results'])==3,'Regression roster');logs=[]
    for row,name in zip(v['results'],('predecessor.log','bounded.log','continuation.log')):
        require(row['exit_code']==0 and row['log']==name and row['sha256']==file_hash(folder/name),'Regression log')
        logs.append((folder/name).read_text())
    node=list(map(int,re.findall(r'ℹ pass (\d+)',logs[0])))
    counts=[list(map(int,re.findall(r'Ran (\d+) tests? in',log))) for log in logs]
    require(node==[237,3] and counts==[SUITES,[21],[13]],'Regression coverage')
    require([log.count('\nOK\n') for log in logs]==[len(SUITES),1,1],'Regression success markers')
    return {'node':240,'python_predecessor':236,'python_bounded':21,'python_continuation':13,'total':510}


def seal(directory):
    out=Path(directory).resolve();require(out==ROOT/'artifacts/runs/m5-continuation-v1','Production directory')
    prior=predecessors();tests=regression();admitted=read(out/'admission.json')
    require(admitted==carry.admission()==read(REVIEW/'preproduction.json')['admission'],'Production source/runtime freeze')
    require(read(REVIEW/'exit.json')['exit_code']==0 and not (out/'failure.json').exists(),'Production completion')
    reference=json.loads(subprocess.check_output(['node','packages/continuation/audit.mjs',str(out/'reference')],cwd=ROOT,text=True))
    require(reference=={'status':'passed','unpatched_reference_games':144},'Full unpatched reference replay')
    audited=parity.audit(out/'reference');require(audited==read(out/'parity.json'),'Full source-bound parity/carry replay')
    outcome=read(REVIEW/'delegate/outcome.json')
    require(outcome['verdict']=='accepted' and outcome['evidence_sha256']==file_hash(REVIEW/'delegate-verification.md'),'Delegate checks')
    for name in ('claude-review.md','claude-request.json','review-dispositions.md'):
        require((REVIEW/name).is_file(),'Missing review: '+name)
    for name,sha in read(REVIEW/'claude-request.json')['files_sha256'].items():require(file_hash(ROOT/name)==sha,'Reviewed file drift')
    sources={**carry.sources(),'docs/M5_CONTINUATION.md':file_hash(ROOT/'docs/M5_CONTINUATION.md')}
    evidence={str((ROOT/'artifacts/reports'/(p['name']+'.json')).relative_to(ROOT)):p['receipt_sha256'] for p in prior}
    for folder in (out,REVIEW,*sorted((ROOT/'artifacts/runs').glob('m5-continuation-preflight-*'))):
        evidence.update({str(p.relative_to(ROOT)):file_hash(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    receipt={'format':'continuation-receipt-1','status':'complete-bounded-simulator-continuation',
             'run':str(out.relative_to(ROOT)),'predecessors':prior,'reference':reference,'parity':audited,'tests':tests,
             'source_files':sources,'evidence_files':evidence,'max_turns':60,'learning_updates':0,'learner_continuation':'not implemented',
             'full_g2':'open','full_g3':'open','full_g4':'open','strength':'unqualified',
             'final_reserved_data_used':False,'public_server_used':False,'prime_used':False}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m5-continuation-'+receipt['content_hash'])
    for kind,files in [('source',sources),('evidence',evidence)]:
        for name,sha in files.items():
            require(file_hash(ROOT/name)==sha,'Live drift');write_once(archive/kind/name,(ROOT/name).read_bytes())
            require(file_hash(archive/kind/name)==sha,'Archive drift')
    require(carry.admission()==admitted,'Final source/runtime binding')
    write_once(archive/'receipt.json',wire(receipt)+b'\n');write_once(ROOT/'artifacts/reports/m5-continuation.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],'parity':audited,'reference':reference,'tests':tests}))


if __name__=='__main__':seal(sys.argv[1])
