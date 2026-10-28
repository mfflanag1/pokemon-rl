"""Verify the exact M7 experiment and archive a reproducible receipt."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_live_train import evidence as e, checkpoint as cp, verify, learner as l
from pokemon_live_train.workflow import REVIEW, QUAL, SCREEN
from pokemon_switch_engine.scalar import require


def once(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Immutable receipt/archive differs: '+str(path))
    else:
        with path.open('xb') as f:f.write(data)


def main():
    frozen=e.read(REVIEW/'preproduction.json');require(frozen['admission']==e.admission(),'Production admission drift')
    for name,sha in frozen['review_and_regression'].items():require(e.file_hash(ROOT/name)==sha,'Preproduction evidence drift')
    regression=e.read(REVIEW/'regression/verification.json')
    require(regression['status']=='passed' and regression['source_files']==cp.sources() and all(x['exit_code']==0 for x in regression['results']),'Source-bound regression')
    for row in regression['results']:require(e.file_hash(REVIEW/'regression'/row['log'])==row['sha256'],'Regression log drift')
    review=e.read(REVIEW/'review-verification.json')
    required={k:v for k,v in cp.sources().items() if k.startswith(('python/pokemon_live_train/','contracts/live-training-v1/','tests/live-training/')) or k in ('scripts/seal-live-training.py','scripts/verify-live-training.py')}
    require(review['blocking_findings_open']==0 and review['independent_completed'] is True and review['sources']==required,'Independent review scope/drift')
    for name,out,module in (('qualification',QUAL,'qualify'),('screen',SCREEN,'screen')):
        exit=e.read(REVIEW/(name+'-exit.json'))
        expected=[str(ROOT/'.venv/bin/python'),'-m','pokemon_live_train.'+module,str(out)]+([str(QUAL)] if name=='screen' else [])
        require(exit['exit_code']==0 and exit['command']==expected and exit['report_sha256']==e.file_hash(out/'report.json') and exit['log_sha256']==e.file_hash(REVIEW/(name+'.log')),'Launcher binding')
    audits={'qualification':verify.qualification(QUAL),'screen':verify.screen(SCREEN,QUAL)}
    source={**cp.sources(),'docs/M7_LIVE_TRAINING.md':e.file_hash(ROOT/'docs/M7_LIVE_TRAINING.md')}
    evidence={}
    for folder in (QUAL,SCREEN,REVIEW):
        evidence.update({str(p.relative_to(ROOT)):e.file_hash(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    receipt={'format':'live-training-receipt-1','status':'completed','admission_hash':e.digest(frozen['admission']),
        'audit':audits,'source_files':source,'evidence_files':evidence,'review':review,
        'regression_results':regression['results'],'qualification_report_sha256':e.file_hash(QUAL/'report.json'),
        'screen_report_sha256':e.file_hash(SCREEN/'report.json'),**l.PLAN['scope']}
    receipt['content_hash']=e.digest(receipt);archive=ROOT/'artifacts/manifests'/('m7-live-training-'+receipt['content_hash'])
    for kind,files in (('source',source),('evidence',evidence)):
        for name,sha in files.items():
            require(e.file_hash(ROOT/name)==sha,'Live hash drift');once(archive/kind/name,(ROOT/name).read_bytes())
            require(e.file_hash(archive/kind/name)==sha,'Archive hash drift')
    require(e.admission()==frozen['admission'],'Final admission drift')
    once(archive/'receipt.json',e.wire(receipt)+b'\n');once(ROOT/'artifacts/reports/m7-live-training.json',e.wire(receipt)+b'\n')
    print(json.dumps({'status':'completed','content_hash':receipt['content_hash'],'audit':audits}))


if __name__=='__main__':main()
