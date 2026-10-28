"""Seal completed F12 evidence only after full independent replay and source checks."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from pokemon_full_train.storage import ROOT,admission,atomic_json,digest,canonical
from .common import read,validate_plan,admit
from .checks import COMMANDS

DOCS=['README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md','docs/REVIEW_RECORD.md','docs/F12_CURRICULUM_LEARNING.md']


def check_records(run,review,manifest,report,verification,tests):
    h=digest(run/'inputs/manifest.json');plan=manifest['plan']
    if report['status']!='passed' or report['plan']!=plan or report['input_hash']!=h or report['counts']!=plan['expected_counts'] or report['interpretation']!=plan['interpretation']:
        raise ValueError('Production report binding invalid')
    if (verification['status']!='passed' or verification['report_sha256']!=digest(run/'report.json') or
        verification['input_hash']!=h or verification['all_seed_curriculum_benefit']!=report['all_seed_curriculum_benefit']):
        raise ValueError('Independent verification binding invalid')
    for key,count in {**plan['expected_counts'],'updates':288,'stage_checkpoints':18,'original_next_update_recoveries':6}.items():
        if verification['checks'][key]!=count:raise ValueError('Missing exhaustive verification: '+key)
    for stage,module,artifact,args in [
        ('production','pokemon_curriculum.campaign',run/'report.json',['--run',str(run)]),
        ('verification','pokemon_curriculum.verify',review/'verification/verification.json',['--run',str(run),'--evidence',str(review/'verification')])]:
        exit_record=read(review/(stage+'-exit.json'))
        if (exit_record['exit_code']!=0 or exit_record['artifact']!=str(artifact.relative_to(ROOT)) or
            exit_record['sha256']!=digest(artifact) or
            exit_record['command']!=[str(ROOT/'.venv/bin/python'),'-m',module,*args]):
            raise ValueError('Launcher command/artifact/exit binding invalid')
    if tests['status']!='passed' or tests['sources']!=manifest['sources'] or tests['runtime']!=manifest['runtime'] or len(tests['runs'])!=len(COMMANDS):
        raise ValueError('Tested source mismatch')
    for item,(_,command,count) in zip(tests['runs'],COMMANDS):
        if (item['command']!=command or item['exit_code']!=0 or item['tests']!=count or
            item['count_verified'] is not True or digest(ROOT/item['log'])!=item['sha256']):
            raise ValueError('Executed test command/log binding invalid')
        text=(ROOT/item['log']).read_text()
        if not re.search(rf'Ran {count} tests? in ',text) or '\nOK\n' not in text:raise ValueError('Test log does not show passing count')
    if tests['tests_passed']!=sum(c for _,_,c in COMMANDS):raise ValueError('Wrong test total')


def seal(run,review):
    manifest=read(run/'inputs/manifest.json');validate_plan(manifest['plan']);admission(run,manifest);admit(manifest['plan'])
    report=read(run/'report.json');verification=read(review/'verification/verification.json');tests=read(review/'checks.json')
    check_records(run,review,manifest,report,verification,tests)
    for name in DOCS:
        saved=review/'current-docs'/name;saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():
            if saved.read_bytes()!=(ROOT/name).read_bytes():raise ValueError('Completion documentation changed')
        else:saved.write_bytes((ROOT/name).read_bytes())
    evidence={str(p.relative_to(ROOT)):digest(p) for base in [run,review] for p in sorted(base.rglob('*')) if p.is_file()}
    receipt={'schema_version':'f12-curriculum-learning-receipt-v1','status':'passed','run':str(run.relative_to(ROOT)),
        'input_hash':digest(run/'inputs/manifest.json'),'report_sha256':digest(run/'report.json'),
        'source_files':manifest['sources'],'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),
        'evidence_files':evidence,'verification':verification['checks'],'tests_passed':tests['tests_passed'],
        'counts':report['counts'],'training':report['training'],'evaluation':report['evaluation'],
        'all_seed_curriculum_benefit':report['all_seed_curriculum_benefit'],
        'seed_benefit':{s:v['analysis']['curriculum_benefit'] for s,v in report['seeds'].items()},
        'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,'interpretation':report['interpretation']}
    receipt['content_hash']=hashlib.sha256(canonical(receipt)).hexdigest()
    path=ROOT/'artifacts/reports/f12-curriculum.json';archive=ROOT/'artifacts/manifests'/('f12-curriculum-'+receipt['content_hash'])/'receipt.json'
    for target in [path,archive]:
        if target.exists():
            if target.read_bytes()!=canonical(receipt)+b'\n':raise ValueError('Existing receipt differs')
        else:atomic_json(target,receipt)
    return {'receipt':str(path),'content_hash':receipt['content_hash'],'sha256':digest(path)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    a=p.parse_args();print(json.dumps(seal(a.run.resolve(),a.review.resolve())),flush=True)
