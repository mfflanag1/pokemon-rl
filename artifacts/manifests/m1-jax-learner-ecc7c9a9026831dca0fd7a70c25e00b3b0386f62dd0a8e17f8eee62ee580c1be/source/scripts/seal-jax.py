"""Seal the requested JAX slice evidence without promoting the still-open G2 gate."""
import json
from pathlib import Path
import re
import sys
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_slice.engine import require
from pokemon_reference.corpus import read_json,verify_seal
from pokemon_reference.dataset_gate import ActorDataset,checked_file
from pokemon_jax.campaign import provenance,load_references

REVIEW=ROOT/'artifacts/reviews/m1-g2'
RUN=ROOT/'artifacts/runs/m1-g2-v4'
PLAYER=ROOT/'artifacts/runs/m1-g2-player-v1'


def read(path):return json.loads(path.read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Refusing to replace sealed file: '+str(path))
    else:
        with path.open('xb') as f:f.write(data)


def main():
    log=(REVIEW/'npm-test.log').read_text()
    match=re.search(r'^ℹ tests (\d+)$',log,re.M);require(match is not None,'Missing Node summary');node=int(match[1])
    require(re.search(rf'^ℹ pass {node}$',log,re.M) and re.search(r'^ℹ fail 0$',log,re.M),'Node regression failure')
    suites=[int(x) for x in re.findall(r'^Ran (\d+) tests? in ',log,re.M)]
    require(len(suites)==9 and log.count('\nOK\n')==9 and 'FAILED (' not in log,'Python regression incomplete')
    tests={'node_passed':node,'python_passed':sum(suites),'python_suite_counts':suites,'log_sha256':file_hash(REVIEW/'npm-test.log')}
    result=read(RUN/'report.json');admission=read(RUN/'admission.json');plan=read(ROOT/'contracts/jax-slice-v1/campaign.json')
    require(result['content_hash']==digest({k:v for k,v in result.items() if k!='content_hash'}),'Run result seal')
    require(result['admission_hash']==digest(admission) and admission['plan']==plan,'Admission/plan link')
    require(admission['source_hashes']==provenance(),'Run source drift')
    require(result['status']=='passed' and result['full_g2']=='open','Incorrect promotion status')
    old,new=load_references();require(admission['reference_hashes']==[old['content_hash'],new['content_hash']],'Reference drift')
    require(result['conditional']['battles']==10000 and result['player']['battles']==1000,'Reference counts')
    require(result['conditional']['disagreements']==result['player']['disagreements']==0 and result['damage_enumeration']==32,'Parity gate')
    chance=result['chance'];require(chance['status']=='passed' and not chance['failures'],'Chance failure')
    require(chance['test_count']==plan['family_test_count'] and set(chance['tests'])==set(plan['populations']),'Chance family')
    cutoff=plan['family_alpha']/plan['family_test_count'];require(chance['bonferroni_cutoff']==cutoff,'Chance threshold')
    for rows in chance['tests'].values():
        require(len(rows)==plan['tests_per_population']==len({r['name'] for r in rows}),'Missing chance cells')
        for row in rows:
            expected=float(binomtest(row['count'],row['n'],row['probability']).pvalue) if row['n'] else 1.0
            require(row['p_value']==expected and expected>=cutoff,'Chance evidence/decision mismatch')
    # Preserve predecessor code and final-data reservation rather than reclassifying them.
    g1=read(ROOT/'artifacts/reports/m1-g1.json')
    require(g1['content_hash']==digest({k:v for k,v in g1.items() if k!='content_hash'}),'G1 receipt')
    prefixes=('python/pokemon_reference/','python/pokemon_learner/','python/pokemon_slice/','packages/reference-node/','packages/slice/','contracts/slice-v1/')
    for name,h in g1['source_files'].items():
        if name.startswith(prefixes) or name in ('requirements-learner.txt','package-lock.json'):checked_file(ROOT,name,h)
    r7=verify_seal(read_json(ROOT/'artifacts/reports/r7-sprint.json'))
    dataset=ActorDataset(ROOT/r7['dataset_directory'],expected_hash=r7['dataset_hash'],purpose='training')
    checked_file(dataset.directory,'teams/final.json',dataset.document['collection_files']['teams/final.json'])
    require(dataset.document['validation']['final_episodes']==0,'Final outcomes were collected')
    require((REVIEW/'review-dispositions.md').is_file() and (REVIEW/'claude-review.md').is_file(),'Review evidence missing')
    sources=[]
    for directory in ['packages','python','contracts','scripts','tests','docs','tools/mechanics-deps']:
        sources.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and not {'node_modules','__pycache__'} & set(p.parts))
    sources.extend(ROOT/n for n in ['README.md','package.json','package-lock.json','showdown.pin.json','requirements-contract.txt','requirements-learner.txt','requirements-jax.txt','.node-version','.gitignore'])
    sources=sorted(set(sources))
    directories=[REVIEW,RUN,PLAYER,ROOT/'artifacts/runs/m1-g2-v1',ROOT/'artifacts/runs/m1-g2-v2',ROOT/'artifacts/runs/m1-g2-v3',ROOT/'artifacts/runs/m1-slice-v1']
    evidence=sorted(set([p for d in directories for p in d.rglob('*') if p.is_file()]+[ROOT/'artifacts/reports/m1-g1.json',ROOT/'artifacts/reports/r7-sprint.json']))
    source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in sources};evidence_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in evidence}
    report={'version':'m1-g2-slice-completion-1','status':'complete-for-requested-slice','full_g2':'open','profile':result['profile'],
        'campaign_directory':str(RUN.relative_to(ROOT)),'campaign_hash':result['content_hash'],'campaign':result,
        'runtime':admission['runtime'],'tests':tests,'final_reservation_unchanged':True,'previous_g1_hash':g1['content_hash'],
        'source_files':source_hashes,'evidence_files':evidence_hashes,'review_dispositions':'artifacts/reviews/m1-g2/review-dispositions.md',
        'limitations':['Exactly fixed public-set Rattata duel; no official-format promotion.',
            '10,000 conditional transition cases, but only 1,000 fresh complete player-view cases.',
            'Fixed public sets cannot satisfy G2 hidden-set mutation strata; pending/latent chance probes are restricted evidence.',
            'CPU kernel-only benchmark; no trained Pokemon policy or complete collect/update performance claim.',
            'No GPU/Prime resources or final-evaluation outcomes used.'],
        'next':'JAX autodiff learner parity against G1, observation-only restricted collector, boundary resume and complete-cycle CPU profiling.'}
    report['content_hash']=digest(report);content=wire(report)+b'\n';archive=ROOT/'artifacts/manifests'/('m1-g2-'+report['content_hash'])
    for group,files,hashes in [('source',sources,source_hashes),('evidence',evidence,evidence_hashes)]:
        for p in files:
            rel=str(p.relative_to(ROOT));dest=archive/group/rel;write_once(dest,p.read_bytes());require(file_hash(dest)==hashes[rel],'Archive copy mismatch')
    write_once(archive/'completion.json',content);write_once(ROOT/'artifacts/reports/m1-g2.json',content)
    print(json.dumps({'report':str(ROOT/'artifacts/reports/m1-g2.json'),'archive':str(archive),'content_hash':report['content_hash'],'tests':tests,'source_files':len(sources),'evidence_files':len(evidence)},sort_keys=True))


if __name__=='__main__':main()
