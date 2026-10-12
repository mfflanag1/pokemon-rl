"""Copied-evidence sealer negatives; never mutate production or predecessor evidence."""
from pathlib import Path
import copy,json,runpy,shutil,tempfile
from pokemon_learner.checkpoint import wire,file_hash,digest
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'artifacts/runs/m3-switch-engine-v1'
S=runpy.run_path(str(ROOT/'scripts/seal-switch-engine.py'))
receipt=ROOT/'artifacts/reports/m3-switch-engine.json'
before=receipt.read_bytes() if receipt.exists() else None
production_hashes={str(p.relative_to(RUN)):file_hash(p) for p in RUN.rglob('*') if p.is_file()}
results=[]

def edit(path,fn):
    x=json.loads(path.read_bytes());fn(x);path.write_bytes(wire(x)+b'\n')

def first_seed(path,fn):
    p=path/'learner/seed-101/report.json';edit(p,fn)
    edit(path/'learner/report.json',lambda r:r['seeds'][0].update(sha256=file_hash(p)))

cases=[
 ('failure marker','Failed run cannot seal',lambda p:(p/'failure.json').write_text('{}'),'seal'),
 ('source drift','Frozen source/runtime',lambda p:edit(p/'preregistration.json',lambda x:x['admission']['source_files'].clear()),'seal'),
 ('runtime drift','Frozen source/runtime',lambda p:edit(p/'preregistration.json',lambda x:x['admission']['runtime'].update(jax='wrong')),'seal'),
 ('plan drift','Frozen source/runtime',lambda p:edit(p/'preregistration.json',lambda x:x['plan'].update(fresh_games=1)),'seal'),
 ('incomplete status','Frozen source/runtime',lambda p:edit(p/'report.json',lambda x:x.update(status='incomplete')),'seal'),
 ('parity binding','Report binding',lambda p:edit(p/'report.json',lambda x:x.update(parity_sha256='0'*64)),'seal'),
 ('learner binding','Report binding',lambda p:edit(p/'report.json',lambda x:x.update(learner_sha256='0'*64)),'seal'),
 ('reserved-data flag','Scope',lambda p:edit(p/'learner/report.json',lambda x:x.update(final_evaluation_used=True)),'audit_learning'),
 ('missing update','Fixed update budget',lambda p:first_seed(p,lambda x:x['records'].pop()),'audit_learning'),
 ('counter linkage','Update counter/fingerprint chain',lambda p:first_seed(p,lambda x:x['records'][0].update(next_game=33)),'audit_learning'),
 ('forged original-next audit','Saved replay audit',lambda p:first_seed(p,lambda x:x['original_next_update'][0]['original_next_update'].update(buffer='0'*64)),'audit_learning'),
]
for name,expected,mutate,method in cases:
    with tempfile.TemporaryDirectory(prefix='switch-negative-',dir=ROOT/'artifacts/runs') as d:
        p=Path(d)
        for name_ in ('preregistration.json','report.json'):shutil.copyfile(RUN/name_,p/name_)
        (p/'reference').mkdir();shutil.copyfile(RUN/'reference/parity.json',p/'reference/parity.json')
        shutil.copytree(RUN/'learner',p/'learner')
        mutate(p)
        try:S[method](p)
        except ValueError as e:
            assert expected in str(e),(name,str(e));results.append({'case':name,'status':'rejected','reason':str(e)})
        else:raise AssertionError('Negative accepted: '+name)
assert (receipt.read_bytes() if receipt.exists() else None)==before,'Negative wrote a receipt'
assert production_hashes=={str(p.relative_to(RUN)):file_hash(p) for p in RUN.rglob('*') if p.is_file()},'Production evidence drift'
print(json.dumps({'status':'pass','checks':results,'production_unchanged':True}))
