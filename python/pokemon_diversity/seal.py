"""Seal the fixed F18 comparison only after complete replay and bound checks."""
import argparse,hashlib,re
from pathlib import Path
from pokemon_full_train.storage import ROOT,atomic_json,canonical,digest,admission
from .common import read,validate_plan,admit
from .checks import COMMANDS
DOCS=['README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md','docs/REVIEW_RECORD.md','docs/F18_DATA_COVERAGE.md']
TIMING=['elapsed_seconds','additional_collection_seconds','probe_collection_seconds','update_seconds_including_compile','peak_parent_rss_mib']

def check(run,review):
    m=read(run/'inputs/manifest.json');r=read(run/'report.json');v=read(review/'replay/verification.json');t=read(review/'checks.json');p=m['plan'];h=digest(run/'inputs/manifest.json')
    validate_plan(p)
    if r['status']!='passed' or r['plan']!=p or r['input_hash']!=h or r['updates']!=12288 or r['presentations']!=1572864 or r['new_games']!=1792 or r['sealed_original_controls_matched']!=3 or r['sampler_parity_draws']!=12 or r['value_head_weights_unchanged'] is not True:
        raise ValueError('Production report binding')
    if any(r[k] for k in ['final_reserved_data_used','public_server_used','prime_used']) or r['original_training_games']!=512 or r['expanded_training_games']!=2048:
        raise ValueError('Production scope mismatch')
    for name,count,split in [('additional',1536,'train'),('probe',256,'development')]:
        x=r[name]
        if x!=read(run/name/'inventory.json') or x['games']!=count or x['split']!=split or x['terminals']+x['cutoffs']!=count or x['rows']+x['rejected_choices']!=x['decisions']:
            raise ValueError('Collected inventory mismatch')
    if set(r['seeds'])!={'7103','8209','9311'} or r['all_seed_gate']!=all(x['gate']['passed'] for x in r['seeds'].values()):
        raise ValueError('Seed gate inventory mismatch')
    if read(run/'scientific.json')!={k:x for k,x in r.items() if k not in TIMING}:
        raise ValueError('Scientific/report mismatch')
    expected={'status':'passed','report_sha256':digest(run/'report.json'),'scientific_sha256':digest(run/'scientific.json'),'input_hash':h,'new_games':1792,'updates':12288,'presentations':1572864,'checkpoint_comparisons':18,'original_next_update_recoveries':6,'sealed_original_controls_matched':3,'all_seed_gate':r['all_seed_gate']}
    if v!=expected or (run/'scientific.json').read_bytes()!=(review/'replay/scientific.json').read_bytes():
        raise ValueError('Exhaustive replay binding')
    if t['status']!='passed' or t['sources']!=m['sources'] or t['runtime']!=m['runtime'] or len(t['runs'])!=len(COMMANDS) or t['tests_passed']!=sum(x[2] for x in COMMANDS):
        raise ValueError('Tested source/runtime mismatch')
    for item,(_,command,count) in zip(t['runs'],COMMANDS):
        text=(ROOT/item['log']).read_text()
        if item['command']!=command or item['tests']!=count or item['exit_code']!=0 or item['count_verified'] is not True or digest(ROOT/item['log'])!=item['sha256'] or not re.search(rf'Ran {count} tests? in ',text) or '\nOK\n' not in text:
            raise ValueError('Test log binding')
    for stage,path,args in [('production',run/'report.json',['--run',str(run)]),('verification',review/'replay/verification.json',['--run',str(review/'replay'),'--original',str(run)])]:
        x=read(review/(stage+'-exit.json'))
        if x['exit_code']!=0 or x['sha256']!=digest(path) or x['command']!=[str(ROOT/'.venv/bin/python'),'-B','-m','pokemon_diversity.campaign',*args]:
            raise ValueError('Launcher command/exit binding')
    if read(review/'workflow-exit.json')['exit_code']!=0:
        raise ValueError('Workflow failure')
    build=read(review/'build-result.json')
    if build['command']!=['node','build'] or build['cwd']!='vendor/pokemon-showdown' or build['exit_code']!=0 or digest(ROOT/build['log'])!=build['sha256']:
        raise ValueError('Pinned simulator build mismatch')
    return m,r,v,t

def seal(run,review):
    m,r,v,t=check(run,review);admission(run,m);admit(m['plan'])
    for name in DOCS:
        saved=review/'current-docs'/name;saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():
            if saved.read_bytes()!=(ROOT/name).read_bytes():raise ValueError('Completion document changed')
        else:saved.write_bytes((ROOT/name).read_bytes())
    evidence={str(f.relative_to(ROOT)):digest(f) for base in [run,review] for f in sorted(base.rglob('*')) if f.is_file()}
    receipt={'schema_version':'f18-data-coverage-receipt-v1','status':'passed','run':str(run.relative_to(ROOT)),
        'report_sha256':digest(run/'report.json'),'input_hash':digest(run/'inputs/manifest.json'),
        'predecessor_receipt':m['plan']['predecessor_receipt'],'predecessor_sha256':m['plan']['predecessor_sha256'],
        'source_files':m['sources'],'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),
        'evidence_files':evidence,'verification':v,'tests_passed':t['tests_passed'],
        'all_seed_gate':r['all_seed_gate'],'seed_gates':{s:x['gate'] for s,x in r['seeds'].items()},
        'updates':r['updates'],'new_games':r['new_games'],'additional':r['additional'],'probe':r['probe'],
        'public_server_used':False,'final_reserved_data_used':False,'prime_used':False,'interpretation':m['plan']['interpretation']}
    receipt['content_hash']=hashlib.sha256(canonical(receipt)).hexdigest()
    path=ROOT/'artifacts/reports/f18-data-coverage.json';archive=ROOT/'artifacts/manifests'/('f18-data-coverage-'+receipt['content_hash'])/'receipt.json'
    for target in [path,archive]:
        if target.exists():
            if target.read_bytes()!=canonical(receipt)+b'\n':raise ValueError('Existing receipt changed')
        else:atomic_json(target,receipt)
    return {'receipt':str(path),'sha256':digest(path),'content_hash':receipt['content_hash']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    a=p.parse_args();print(seal(a.run.resolve(),a.review.resolve()),flush=True)
