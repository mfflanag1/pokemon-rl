"""Bind the completed fixed evaluation, exhaustive replay and source-bound checks."""
import argparse,hashlib,re
from pathlib import Path
from pokemon_full_train.storage import ROOT,atomic_json,canonical,digest,admission
from .common import read,validate_plan,admit
from .checks import COMMANDS
from .campaign import TIMING
DOCS=['README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md',
    'docs/REVIEW_RECORD.md','docs/F21_SHARED_BATTLE.md','docs/OBJECTIVE_REVIEW_2026_10.md']

def check(run,review):
    m=read(run/'inputs/manifest.json');r=read(run/'report.json');v=read(review/'replay/verification.json');t=read(review/'checks.json');p=m['plan'];h=digest(run/'inputs/manifest.json')
    validate_plan(p)
    if r['status']!='passed' or r['plan']!=p or r['input_hash']!=h or r['games']!=6912 or r['optimizer_updates']!=0 or r['checkpoint_loads']!=9:
        raise ValueError('Production report binding')
    if any(r[k] for k in ['final_reserved_data_used','public_server_used','prime_used','promotion_authorized']):
        raise ValueError('Production scope mismatch')
    if set(r['seeds'])!={'7103','8209','9311'} or r['all_seed_battle_benefit']!=all(s['analysis']['battle_benefit'] for s in r['seeds'].values()):
        raise ValueError('Seed gate inventory')
    if r['all_seed_shared_mastery']!=all(s['analysis']['familiar_mastery']['shared']['passed'] for s in r['seeds'].values()):
        raise ValueError('Mastery inventory')
    for s in r['seeds'].values():
        if set(s['panels'])!={'flat','shared','reference'} or set(s['identities'])!=set(s['panels']) or set(s['analysis']['comparisons'])!={a[0] for a in p['comparisons']}:
            raise ValueError('Missing arm or contrast')
        for arm,panel in s['panels'].items():
            if set(panel)!=set(p['opponents']):raise ValueError('Missing opponent')
            for g in panel.values():
                if g['games']!=256 or g['terminals']+g['cutoffs']!=256 or g['learner_wins']+g['learner_losses']+g['true_ties']!=g['terminals']:
                    raise ValueError('Incomplete outcome panel')
    if r['evaluation']['games']!=6912 or r['evaluation']['terminals']+r['evaluation']['cutoffs']!=6912 or r['roles']['disallowed_overlap']!=0:
        raise ValueError('Invalid total inventory')
    if read(run/'scientific.json')!={k:x for k,x in r.items() if k not in TIMING}:
        raise ValueError('Scientific/report mismatch')
    expected={'status':'passed','input_hash':h,'report_sha256':digest(run/'report.json'),
        'scientific_sha256':digest(run/'scientific.json'),'games':6912,'checkpoint_loads':9,
        'actor_decisions':r['actor_decisions'],'all_decisions':r['evaluation']['decisions'],
        'optimizer_updates':0,'all_seed_battle_benefit':r['all_seed_battle_benefit']}
    if v!=expected or (run/'scientific.json').read_bytes()!=(review/'replay/scientific.json').read_bytes():
        raise ValueError('Full replay binding')
    if t['status']!='passed' or t['sources']!=m['sources'] or t['runtime']!=m['runtime'] or len(t['runs'])!=len(COMMANDS) or t['tests_passed']!=sum(x[2] for x in COMMANDS):
        raise ValueError('Tested source/runtime mismatch')
    for item,(_,command,count) in zip(t['runs'],COMMANDS):
        text=(ROOT/item['log']).read_text()
        if item['command']!=command or item['tests']!=count or item['exit_code']!=0 or item['count_verified'] is not True or digest(ROOT/item['log'])!=item['sha256'] or not re.search(rf'Ran {count} tests? in ',text) or '\nOK\n' not in text:
            raise ValueError('Test log binding')
    for stage,path,args in [('production',run/'report.json',['--run',str(run)]),('verification',review/'replay/verification.json',['--run',str(review/'replay'),'--original',str(run)])]:
        x=read(review/(stage+'-exit.json'))
        if x['exit_code']!=0 or x['sha256']!=digest(path) or x['command']!=[str(ROOT/'.venv/bin/python'),'-B','-m','pokemon_shared_battle.campaign',*args]:
            raise ValueError('Launcher command/exit binding')
    if read(review/'workflow-exit.json')['exit_code']!=0:raise ValueError('Workflow failure')
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
    receipt={'schema_version':'f21-shared-battle-receipt-v1','status':'passed','run':str(run.relative_to(ROOT)),
        'report_sha256':digest(run/'report.json'),'input_hash':digest(run/'inputs/manifest.json'),
        'predecessor_receipt':m['plan']['predecessor_receipt'],'predecessor_sha256':m['plan']['predecessor_sha256'],
        'reference_receipt':m['plan']['reference_receipt'],'reference_sha256':m['plan']['reference_sha256'],
        'source_files':m['sources'],'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),
        'evidence_files':evidence,'verification':v,'tests_passed':t['tests_passed'],
        'all_seed_battle_benefit':r['all_seed_battle_benefit'],'all_seed_shared_mastery':r['all_seed_shared_mastery'],
        'seed_gates':{s:x['analysis']['battle_checks'] for s,x in r['seeds'].items()},
        'evaluation':r['evaluation'],'actor_decisions':r['actor_decisions'],'unavailable_choices':r['unavailable_choices'],
        'optimizer_updates':0,'promotion_authorized':False,'public_server_used':False,
        'final_reserved_data_used':False,'prime_used':False,'interpretation':m['plan']['interpretation']}
    receipt['content_hash']=hashlib.sha256(canonical(receipt)).hexdigest()
    path=ROOT/'artifacts/reports/f21-shared-battle.json';archive=ROOT/'artifacts/manifests'/('f21-shared-battle-'+receipt['content_hash'])/'receipt.json'
    for target in [path,archive]:
        if target.exists():
            if target.read_bytes()!=canonical(receipt)+b'\n':raise ValueError('Existing receipt changed')
        else:atomic_json(target,receipt)
    return {'receipt':str(path),'sha256':digest(path),'content_hash':receipt['content_hash']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    a=p.parse_args();print(seal(a.run.resolve(),a.review.resolve()),flush=True)
