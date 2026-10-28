"""Seal F15 after exhaustive fresh replay, original checkpoint recovery and tests."""
import argparse
import hashlib
import re
from pathlib import Path
from pokemon_full_train.storage import ROOT,atomic_json,admission,canonical,digest
from .common import read,admit
from .campaign import validate_plan
from .checks import COMMANDS

DOCS=['README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md','docs/REVIEW_RECORD.md','docs/F15_FULL_IMITATION.md']

def check_records(run,review,manifest,report,verification,tests):
    h=digest(run/'inputs/manifest.json');plan=manifest['plan']
    if report['status']!='passed' or report['plan']!=plan or report['input_hash']!=h or report['counts']!=plan['expected'] or report['interpretation']!=plan['interpretation']:raise ValueError('Production binding invalid')
    scientific=read(run/'scientific.json')
    if scientific!={k:v for k,v in report.items() if k not in ['elapsed_seconds','update_seconds_including_compile','peak_parent_rss_mib']}:raise ValueError('Scientific/report mismatch')
    if verification['status']!='passed' or verification['report_sha256']!=digest(run/'report.json') or verification['scientific_sha256']!=digest(run/'scientific.json') or verification['input_hash']!=h or verification['all_seed_gates']!=report['all_seed_gates']:raise ValueError('Replay binding invalid')
    expected={**plan['expected'],'teacher_datasets':2,'teacher_probe_endpoints':12,'rl_cohorts':192,'checkpoint_comparisons':21,'original_next_update_recoveries':9}
    if verification['checks']!=expected:raise ValueError('Missing exhaustive replay checks')
    if (review/'replay/scientific.json').read_bytes()!=(run/'scientific.json').read_bytes():raise ValueError('Scientific replay differs')
    for stage,artifact,args in [
        ('production',run/'report.json',['--run',str(run)]),
        ('verification',review/'replay/verification.json',['--run',str(review/'replay'),'--original',str(run)])]:
        r=read(review/(stage+'-exit.json'))
        if r['exit_code']!=0 or r['artifact']!=str(artifact.relative_to(ROOT)) or r['sha256']!=digest(artifact) or r['command']!=[str(ROOT/'.venv/bin/python'),'-m','pokemon_imitation.campaign',*args]:raise ValueError('Launcher exit/command/artifact binding')
    if tests['status']!='passed' or tests['sources']!=manifest['sources'] or tests['runtime']!=manifest['runtime'] or len(tests['runs'])!=len(COMMANDS):raise ValueError('Tested source mismatch')
    for item,(_,command,count) in zip(tests['runs'],COMMANDS):
        if item['command']!=command or item['exit_code']!=0 or item['tests']!=count or item['count_verified'] is not True or digest(ROOT/item['log'])!=item['sha256']:raise ValueError('Test log binding invalid')
        text=(ROOT/item['log']).read_text()
        if not re.search(rf'Ran {count} tests? in ',text) or '\nOK\n' not in text:raise ValueError('Wrong passing test count')
    if tests['tests_passed']!=sum(c for _,_,c in COMMANDS):raise ValueError('Wrong test total')


def seal(run,review):
    manifest=read(run/'inputs/manifest.json');validate_plan(manifest['plan']);admission(run,manifest);admit(manifest['plan'])
    report=read(run/'report.json');verification=read(review/'replay/verification.json');tests=read(review/'checks.json')
    check_records(run,review,manifest,report,verification,tests)
    for name in DOCS:
        saved=review/'current-docs'/name;saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():
            if saved.read_bytes()!=(ROOT/name).read_bytes():raise ValueError('Completion documentation changed')
        else:saved.write_bytes((ROOT/name).read_bytes())
    evidence={str(p.relative_to(ROOT)):digest(p) for base in [run,review] for p in sorted(base.rglob('*')) if p.is_file()}
    receipt={'schema_version':'f15-full-imitation-receipt-v1','status':'passed','run':str(run.relative_to(ROOT)),
        'input_hash':digest(run/'inputs/manifest.json'),'report_sha256':digest(run/'report.json'),
        'source_files':manifest['sources'],'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),
        'evidence_files':evidence,'verification':verification['checks'],'tests_passed':tests['tests_passed'],
        'counts':report['counts'],'training':report['training'],'evaluation':report['evaluation'],
        'all_seed_gates':report['all_seed_gates'],
        'seed_gates':{s:{k:v['analysis'][k] for k in report['all_seed_gates']} for s,v in report['seeds'].items()},
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,'interpretation':report['interpretation']}
    receipt['content_hash']=hashlib.sha256(canonical(receipt)).hexdigest()
    path=ROOT/'artifacts/reports/f15-full-imitation.json';archive=ROOT/'artifacts/manifests'/('f15-full-imitation-'+receipt['content_hash'])/'receipt.json'
    for target in [path,archive]:
        if target.exists():
            if target.read_bytes()!=canonical(receipt)+b'\n':raise ValueError('Existing receipt differs')
        else:atomic_json(target,receipt)
    return {'receipt':str(path),'content_hash':receipt['content_hash'],'sha256':digest(path)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    a=p.parse_args();print(seal(a.run.resolve(),a.review.resolve()),flush=True)
