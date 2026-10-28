"""Content-addressed completion receipt after independent replay and documentation."""
import argparse
import json
from pathlib import Path
from pokemon_full_train.storage import ROOT, admission, atomic_json, digest
from .campaign import predecessor
from .core import fingerprint
from .checks import COMMANDS

DOCS=['README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md','docs/REVIEW_RECORD.md','docs/F11_BATTLE_DECISIONS.md']


def seal(run, review):
    manifest=json.loads((run/'inputs/manifest.json').read_text()); admission(run,manifest); predecessor(manifest['plan'])
    report=json.loads((run/'report.json').read_text())
    verification=json.loads((review/'verification/verification.json').read_text())
    tests=json.loads((review/'checks.json').read_text())
    if (verification['status']!='passed' or verification['report_sha256']!=digest(run/'report.json') or
        verification['input_hash']!=fingerprint(manifest) or tests['status']!='passed' or tests['sources']!=manifest['sources']):
        raise ValueError('Verification or tested source binding invalid')
    if len(tests['runs'])!=len(COMMANDS): raise ValueError('Missing test runs')
    for item,(_,command,count) in zip(tests['runs'],COMMANDS):
        if (item['command']!=command or item['exit_code']!=0 or item['tests']!=count or
            item['count_verified'] is not True or digest(ROOT/item['log'])!=item['sha256']):
            raise ValueError('Executed test command/log binding invalid')
    if tests['tests_passed']!=sum(c for _,_,c in COMMANDS): raise ValueError('Wrong test total')
    for name in DOCS:
        saved=review/'current-docs'/name; saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():
            if saved.read_bytes()!=(ROOT/name).read_bytes(): raise ValueError('Completion documentation changed')
        else: saved.write_bytes((ROOT/name).read_bytes())
    evidence={str(p.relative_to(ROOT)):digest(p) for base in [run,review] for p in sorted(base.rglob('*')) if p.is_file()}
    branches=report['branches']
    receipt={'schema_version':'f11-battle-audit-receipt-v1','status':'passed','run':str(run.relative_to(ROOT)),
        'input_hash':fingerprint(manifest),'report_sha256':digest(run/'report.json'),
        'source_files':manifest['sources'],'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),
        'evidence_files':evidence,'verification':verification['checks'],'tests_passed':tests['tests_passed'],
        'counts':{'factual_games':sum(c['counts']['games'] for c in report['factual']),
                  'learner_decisions':sum(c['counts']['learner_decisions'] for c in report['factual']),
                  'disagreements':sum(c['counts']['disagreements'] for c in report['factual']),
                  'conditional_branches':len(branches),
                  'improved':sum(b['terminal_reward_delta'] is not None and b['terminal_reward_delta']>0 for b in branches),
                  'worsened':sum(b['terminal_reward_delta'] is not None and b['terminal_reward_delta']<0 for b in branches),
                  'unchanged':sum(b['terminal_reward_delta']==0 for b in branches),
                  'unresolved_cutoff':sum(b['terminal_reward_delta'] is None for b in branches),
                  'fresh_train_games':sum(c['counts']['games'] for c in report['training']),
                  'scenarios':sum(c['scenario_count'] for c in report['training'])},
        'optimizer_updates':0,'strength_claim':False,'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,
        'interpretation':report['interpretation']}
    receipt['content_hash']=fingerprint(receipt)
    path=ROOT/'artifacts/reports/f11-battle-decisions.json'
    archive=ROOT/'artifacts/manifests'/('f11-battle-decisions-'+receipt['content_hash'])/'receipt.json'
    for target in [path,archive]:
        if target.exists():
            if json.loads(target.read_text())!=receipt: raise ValueError('Existing receipt differs')
        else: atomic_json(target,receipt)
    return {'receipt':str(path),'content_hash':receipt['content_hash'],'sha256':digest(path)}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--review',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(seal(a.run.resolve(),a.review.resolve())),flush=True)
