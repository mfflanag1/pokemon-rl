"""Seal G1 canaries and conditional restricted-slice evidence after verification."""
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,wire,file_hash,provenance,load
from pokemon_learner.core import require
from pokemon_reference.corpus import read_json,verify_seal
from pokemon_reference.dataset_gate import ActorDataset,checked_file
from pokemon_slice.verify import validate_plan

REVIEW=ROOT/'artifacts/reviews/m1-g1'
LEARN=ROOT/'artifacts/runs/m1-g1-v2'
REPEAT=ROOT/'artifacts/runs/m1-g1-v3'
SLICE=ROOT/'artifacts/runs/m1-slice-v1'


def read(path):return json.loads(path.read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,f'Refusing changed sealed file: {path}')
    else:
        with path.open('xb') as f:f.write(data)


def main():
    log=(REVIEW/'npm-test.log').read_text()
    n=re.search(r'^ℹ tests (\d+)$',log,re.M);require(n is not None,'Missing Node test summary');node=int(n[1])
    require(re.search(rf'^ℹ pass {node}$',log,re.M) is not None and re.search(r'^ℹ fail 0$',log,re.M) is not None,'Node tests failed')
    suites=[int(n) for n in re.findall(r'^Ran (\d+) tests? in ',log,re.M)]
    require(len(suites)==8 and log.count('\nOK\n')==8 and 'FAILED (' not in log,'Python tests incomplete')
    tests={'node_passed':node,'python_passed':sum(suites),'python_suite_counts':suites,'log_sha256':file_hash(REVIEW/'npm-test.log')}
    learning=read(LEARN/'report.json');body={k:v for k,v in learning.items() if k!='content_hash'}
    require(learning['content_hash']==digest(body) and learning['status']=='passed','Learning report')
    plan=read(ROOT/'contracts/learner-v1/canaries.json')
    require(learning['provenance']==provenance() and learning['plan_hash']==digest(plan),'Learner source/runtime drift')
    require({(r['task'],r['seed']) for r in learning['results']}=={(t,s) for t in ['bandit','delayed'] for s in plan['seeds']} and len(learning['results'])==6,'Learning cases')
    for row in learning['results']:
        prefix=f"{row['task']}-{row['seed']}";checkpoint=LEARN/f'{prefix}.checkpoint.json'
        trainer=load(checkpoint,expected_plan=plan)
        require(trainer.probabilities()==row['probabilities'] and min(row['probabilities'])>=.99,'Learning threshold/checkpoint mismatch')
        require(trainer.decisions==row['decisions']==plan[row['task']]['sample_budget'],'Fixed learning budget')
        for suffix in ['checkpoint.json','trace.json']:
            require((LEARN/f'{prefix}.{suffix}').read_bytes()==(REPEAT/f'{prefix}.{suffix}').read_bytes(),'Full-run repeat differs')
    reference=verify_seal(read_json(SLICE/'reference.json'));parity=verify_seal(read_json(SLICE/'parity.json'))
    validate_plan(read_json(SLICE/'plan.json'))
    require(parity['reference_hash']==reference['content_hash'] and parity['battles']==10000 and parity['disagreements']==0 and parity['damage_enumeration']==32,'Restricted parity scope/count')
    for name,expected in parity['source_hashes'].items():checked_file(ROOT,name,expected)
    r7=verify_seal(read_json(ROOT/'artifacts/reports/r7-sprint.json'))
    old_data=ROOT/r7['dataset_directory'];admission=ActorDataset(old_data,expected_hash=r7['dataset_hash'],purpose='training')
    checked_file(old_data,'teams/final.json',admission.document['collection_files']['teams/final.json'])
    require(admission.document['validation']['final_episodes']==0,'Final data changed')
    source_files=[]
    for directory in ['packages','python','contracts','scripts','tests','docs','tools/mechanics-deps']:
        source_files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and not {'node_modules','__pycache__'} & set(p.parts))
    source_files.extend(ROOT/name for name in ['README.md','package.json','package-lock.json','showdown.pin.json','requirements-contract.txt','requirements-learner.txt','.node-version','.gitignore'])
    source_files=sorted(set(source_files))
    evidence_files=sorted(set([p for directory in [LEARN,REPEAT,SLICE,REVIEW] for p in directory.rglob('*') if p.is_file()] +
        [ROOT/'artifacts/reports/r6-mechanics.json',ROOT/'artifacts/reports/r7-sprint.json',ROOT/'artifacts/runs/m1-g1-v1/report.json']))
    sources={str(p.relative_to(ROOT)):file_hash(p) for p in source_files}
    evidence={str(p.relative_to(ROOT)):file_hash(p) for p in evidence_files}
    report={'version':'m1-g1-completion-1','status':'complete-for-declared-scope',
        'g1':'passed: synthetic CPU whole-episode canaries','m1_g2':'open: JAX port, chance samplers, public-history/information safety',
        'learning':learning,'tests':tests,'repeat':'all six checkpoints and six training traces byte-identical',
        'slice':parity,'slice_reference_capture_ms':reference['elapsed_ms'],'final_reservation_unchanged':True,
        'previous_r7_hash':r7['content_hash'],'source_files':sources,'evidence_files':evidence,
        'review_dispositions':'artifacts/reviews/m1-g1/review-dispositions.md',
        'limitations':['Tabular synthetic learner; no Pokemon playing-strength claim.',
            'Whole-episode checkpoint boundary only; cut formula is hand-tested, not an online cut collector.',
            'Conditional CPU transition/own-request parity; no independent chance or public-history parity.',
            'No JAX, GPU or Prime resources; final evaluation reservation remains unused.']}
    report['content_hash']=digest(report);content=wire(report)+b'\n'
    archive=ROOT/'artifacts/manifests'/('m1-g1-'+report['content_hash'])
    for group,files,hashes in [('source',source_files,sources),('evidence',evidence_files,evidence)]:
        for p in files:
            rel=str(p.relative_to(ROOT));dest=archive/group/rel
            write_once(dest,p.read_bytes());require(file_hash(dest)==hashes[rel],'Archive hash mismatch')
    write_once(archive/'completion.json',content)
    report_path=ROOT/'artifacts/reports/m1-g1.json';write_once(report_path,content)
    print(json.dumps({'report':str(report_path),'archive':str(archive),'content_hash':report['content_hash'],
        'tests':tests,'learning_runs':6,'parity_battles':parity['battles'],'sources':len(sources),'evidence':len(evidence)},sort_keys=True))

if __name__=='__main__':main()
