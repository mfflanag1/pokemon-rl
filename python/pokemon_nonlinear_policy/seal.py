"""Seal only a complete fixed PPO comparison with exhaustive fresh replay."""
import argparse,hashlib,re
from pathlib import Path
from pokemon_full_train.storage import ROOT,atomic_json,canonical,digest,admission
from .common import read,validate_plan,admit
from .checks import COMMANDS
from .campaign import TIMING
DOCS=['README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md',
      'docs/REVIEW_RECORD.md','docs/F29_NONLINEAR_POLICY.md']

def check(run,review):
    m=read(run/'inputs/manifest.json');r=read(run/'report.json');p=m['plan'];h=digest(run/'inputs/manifest.json')
    v=read(review/'replay/verification.json');t=read(review/'checks.json');validate_plan(p)
    if r['status']!='passed' or r['plan']!=p or r['input_hash']!=h or r['counts']!=p['expected_counts']:
        raise ValueError('Production report binding')
    if any(r[k] for k in ['final_reserved_data_used','public_server_used','prime_used','promotion_authorized']):
        raise ValueError('Forbidden production scope')
    if set(r['seeds'])!={'7103','8209','9311'} or set(r['all_seed_learning'])!=set(p['arms']):
        raise ValueError('Seed/arm inventory')
    if r['all_seed_feature_benefit']!=all(s['analysis']['feature_benefit']['passed'] for s in r['seeds'].values()):
        raise ValueError('Primary tactical feature gate linkage')
    if r['all_seed_base_power_benefit']!=all(s['analysis']['base_power_benefit']['passed'] for s in r['seeds'].values()):
        raise ValueError('Secondary base-power gate linkage')
    for arm in p['arms']:
        if r['all_seed_learning'][arm]!=all(s['analysis']['learning'][arm]['passed'] for s in r['seeds'].values()):
            raise ValueError('Learning gate linkage')
    for s in r['seeds'].values():
        if set(s['arms'])!=set(p['arms']) or set(s['analysis']['comparisons'])!=set(p['comparisons']):
            raise ValueError('Missing arm/contrast')
        endpoints=[]
        for arm,a in s['arms'].items():
            if set(a['endpoints'])!={'0','131072','262144'} or a['optimizer_steps']!=2048:
                raise ValueError('Endpoint/update inventory')
            expected={'self_play':131072,'tactical':131072}
            if a['selected_rows_by_opponent']!=expected:raise ValueError('Opponent row mixture linkage')
            endpoints.extend(a['endpoints'].values())
            for ep in a['endpoints'].values():
                if ep['selfplay_value']['physical_games']!=32:
                    raise ValueError('Self-play diagnostic inventory')
        for ep in endpoints:
            if set(ep['panels'])!=set(p['opponents']):raise ValueError('Opponent inventory')
            for cell in ep['panels'].values():
                if cell['games']!=256 or cell['terminals']+cell['cutoffs']!=256 or cell['learner_wins']+cell['learner_losses']+cell['true_ties']!=cell['terminals']:
                    raise ValueError('Battle outcome inventory')
    for key,n in [('evaluation',13824),('selfplay_diagnostics',576)]:
        x=r[key]
        if x['games']!=n or x['terminals']+x['cutoffs']!=n:raise ValueError('Total outcome inventory')
    if r['roles']['disallowed_overlap']!=0:raise ValueError('Role overlap')
    if read(run/'scientific.json')!={k:x for k,x in r.items() if k not in TIMING}:
        raise ValueError('Scientific/report mismatch')
    expected={'status':'passed','input_hash':h,'report_sha256':digest(run/'report.json'),
        'scientific_sha256':digest(run/'scientific.json'),'counts':r['counts'],
        'training':r['training'],'evaluation':r['evaluation'],'selfplay_diagnostics':r['selfplay_diagnostics']}
    if v!=expected or (run/'scientific.json').read_bytes()!=(review/'replay/scientific.json').read_bytes():
        raise ValueError('Exhaustive replay binding')
    if t['status']!='passed' or t['sources']!=m['sources'] or t['runtime']!=m['runtime'] or len(t['runs'])!=len(COMMANDS) or t['check_groups_passed']!=sum(x[2] for x in COMMANDS):
        raise ValueError('Tested source/runtime mismatch')
    for item,(_,cmd,count,marker) in zip(t['runs'],COMMANDS):
        text=(ROOT/item['log']).read_text()
        passed=bool(re.search(rf'Ran {count} tests? in ',text) and '\nOK\n' in text) if marker=='unittest' else bool(re.search(rf'^# tests {count}$',text,re.M) and re.search(r'^# fail 0$',text,re.M) and re.search(r'^# skipped 0$',text,re.M) and re.search(r'^# cancelled 0$',text,re.M))
        if item['command']!=cmd or item['check_groups']!=count or item['exit_code']!=0 or item['verified'] is not True or not passed or digest(ROOT/item['log'])!=item['sha256']:
            raise ValueError('Qualification log binding')
    for stage,path,args in [('production',run/'report.json',['--run',str(run)]),
            ('verification',review/'replay/verification.json',['--run',str(review/'replay'),'--original',str(run)])]:
        x=read(review/(stage+'-exit.json'))
        if x['exit_code']!=0 or x['sha256']!=digest(path) or x['command']!=[str(ROOT/'.venv/bin/python'),'-B','-m','pokemon_nonlinear_policy.campaign',*args]:
            raise ValueError('Launcher command/exit binding')
    pf=read(review/'preflight/report.json')
    if pf['status']!='passed' or pf['sources']!=m['sources'] or pf['runtime']!=m['runtime']:
        raise ValueError('Preflight source/runtime mismatch')
    expected={(seed,arm) for seed in p['seeds'] for arm in p['arms']}
    if len(pf['records'])!=6 or {(x['seed'],x['arm']) for x in pf['records']}!=expected:
        raise ValueError('Preflight host/arm inventory')
    for x in pf['records']:
        cfg=p['arm_budgets'][x['arm']]
        if not x['original_update_replayed'] or x['selected_rows']!=cfg['selected_rows'] or x['optimizer_steps']!=cfg['optimizer_steps_per_cohort']:
            raise ValueError('Preflight original update')
    if read(review/'workflow-exit.json')['exit_code']!=0:raise ValueError('Workflow failure')
    b=read(review/'build-result.json')
    if b['command']!=['node','build'] or b['cwd']!='vendor/pokemon-showdown' or b['exit_code']!=0 or digest(ROOT/b['log'])!=b['sha256']:
        raise ValueError('Pinned simulator build mismatch')
    return m,r,v,t

def seal(run,review):
    m,r,v,t=check(run,review);admission(run,m);admit(m['plan'])
    for name in DOCS:
        target=review/'current-docs'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=(ROOT/name).read_bytes():raise ValueError('Completion document drift')
        else:target.write_bytes((ROOT/name).read_bytes())
    evidence={str(f.relative_to(ROOT)):digest(f) for base in [run,review] for f in sorted(base.rglob('*')) if f.is_file()}
    receipt={'schema_version':'f29-nonlinear-policy-receipt-v1','status':'passed','run':str(run.relative_to(ROOT)),
        'report_sha256':digest(run/'report.json'),'input_hash':digest(run/'inputs/manifest.json'),
        'predecessor_receipt':m['plan']['f28_receipt'],'predecessor_sha256':m['plan']['f28_sha256'],
        'diagnostic_receipt':m['plan']['f27_receipt'],'diagnostic_sha256':m['plan']['f27_sha256'],
        'source_files':m['sources'],'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),
        'evidence_files':evidence,'verification':v,'check_groups_passed':t['check_groups_passed'],
        'counts':r['counts'],'training':r['training'],'evaluation':r['evaluation'],
        'selfplay_diagnostics':r['selfplay_diagnostics'],'all_seed_learning':r['all_seed_learning'],
        'all_seed_feature_benefit':r['all_seed_feature_benefit'],
        'all_seed_base_power_benefit':r['all_seed_base_power_benefit'],
        'reused_predecessor_checks':{'f25':88,'f26':18,'f27':13,'f28':15},
        'seed_gates':{s:x['analysis'] for s,x in r['seeds'].items()},
        'promotion_authorized':False,'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,
        'interpretation':m['plan']['interpretation']}
    receipt['content_hash']=hashlib.sha256(canonical(receipt)).hexdigest()
    path=ROOT/'artifacts/reports/f29-nonlinear-policy.json'
    archive=ROOT/'artifacts/manifests'/('f29-nonlinear-policy-'+receipt['content_hash'])/'receipt.json'
    for target in [path,archive]:
        if target.exists():
            if target.read_bytes()!=canonical(receipt)+b'\n':raise ValueError('Existing receipt changed')
        else:atomic_json(target,receipt)
    return {'receipt':str(path),'sha256':digest(path),'content_hash':receipt['content_hash']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    a=p.parse_args();print(seal(a.run.resolve(),a.review.resolve()),flush=True)
