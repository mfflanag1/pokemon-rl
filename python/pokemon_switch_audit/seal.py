"""Bind completed F17 audit/canary/replay and current handoff without promotion."""
import argparse,hashlib,re
from pathlib import Path
from pokemon_full_train.storage import ROOT,canonical,digest,atomic_json,admission
from .canary import read,admit
from .checks import COMMANDS
DOCS=['README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md','docs/REVIEW_RECORD.md','docs/F17_SWITCH_REPRESENTATION.md']

def check(run,review):
    m=read(run/'inputs/manifest.json');r=read(run/'report.json');v=read(review/'replay/verification.json');t=read(review/'checks.json');p=m['plan']
    if r['status']!='passed' or r['plan']!=p or r['input_hash']!=digest(run/'inputs/manifest.json') or r['updates']!=12288 or r['presentations']!=1572864 or r['new_games']!=0:raise ValueError('Canary report binding')
    if read(run/'scientific.json')!={k:v for k,v in r.items() if k not in ['elapsed_seconds','peak_parent_rss_mib']}:raise ValueError('Scientific report binding')
    if v!={'status':'passed','report_sha256':digest(run/'report.json'),'scientific_sha256':digest(run/'scientific.json'),'updates':12288,'presentations':1572864,'checkpoint_comparisons':18,'original_next_update_recoveries':6,'all_seed_gate':r['all_seed_gate']}:raise ValueError('Replay binding')
    if (run/'scientific.json').read_bytes()!=(review/'replay/scientific.json').read_bytes():raise ValueError('Replay scientific bytes')
    if t['status']!='passed' or t['sources']!=m['sources'] or t['runtime']!=m['runtime'] or t['tests_passed']!=28:raise ValueError('Tested source/runtime binding')
    for item,(name,cmd,count,marker) in zip(t['runs'],COMMANDS):
        if item['command']!=cmd or item['tests']!=count or item['marker']!=marker:raise ValueError('Test command/count binding')
        text=(ROOT/item['log']).read_text()
        if item['exit_code']!=0 or digest(ROOT/item['log'])!=item['sha256'] or item['marker'] not in text:raise ValueError('Test log binding')
    if len(t['runs'])!=4 or sum(item['tests'] for item in t['runs'])!=28:raise ValueError('Test count binding')
    for stage,path,args in [('production',run/'report.json',['--run',str(run)]),('verification',review/'replay/verification.json',['--run',str(review/'replay'),'--original',str(run)])]:
        x=read(review/(stage+'-exit.json'))
        if x['exit_code']!=0 or x['sha256']!=digest(path) or x['command']!=[str(ROOT/'.venv/bin/python'),'-B','-m','pokemon_switch_audit.canary',*args]:raise ValueError('Launcher stage binding')
    if read(review/'workflow-exit.json')['exit_code']!=0:raise ValueError('Workflow exit')
    audit=ROOT/'artifacts/runs/f17-switch-audit-v1';a=read(audit/'report.json')
    if a['status']!='passed' or a['counts']!={'train':{'games':512,'rows':26278,'terminals':512,'cutoffs':0},'probe':{'games':128,'rows':6533,'terminals':128,'cutoffs':0}}:raise ValueError('Audit counts')
    for rel,sha in read(run/'audit-inputs.json').items():
        if digest(ROOT/rel)!=sha:raise ValueError('Audited input drift')
    if read(review/'audit-exit.json')['exit_code']!=0 or read(review/'audit-exit.json')['sha256']!=digest(audit/'report.json'):raise ValueError('Audit exit binding')
    return m,r,v,t

def seal(run,review):
    m,r,v,t=check(run,review);admission(run,m);admit(m['plan'])
    for name in DOCS:
        saved=review/'current-docs'/name;saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():
            if saved.read_bytes()!=(ROOT/name).read_bytes():raise ValueError('Document changed')
        else:saved.write_bytes((ROOT/name).read_bytes())
    evidence={str(p.relative_to(ROOT)):digest(p) for base in [run,review,ROOT/'artifacts/runs/f17-switch-audit-v1'] for p in sorted(base.rglob('*')) if p.is_file()}
    receipt={'schema_version':'f17-switch-representation-receipt-v1','status':'passed','run':str(run.relative_to(ROOT)),'report_sha256':digest(run/'report.json'),'audit_report_sha256':digest(ROOT/'artifacts/runs/f17-switch-audit-v1/report.json'),'predecessor_receipt':m['plan']['predecessor_receipt'],'predecessor_sha256':m['plan']['predecessor_sha256'],'source_files':m['sources'],'source_snapshot_root':str((run/'inputs/source').relative_to(ROOT)),'evidence_files':evidence,'verification':v,'tests_passed':t['tests_passed'],'all_seed_gate':r['all_seed_gate'],'seed_gates':{s:x['gate'] for s,x in r['seeds'].items()},'updates':r['updates'],'new_games':0,'public_server_used':False,'final_reserved_data_used':False,'prime_used':False,'scope':m['plan']['scope']}
    receipt['content_hash']=hashlib.sha256(canonical(receipt)).hexdigest()
    target=ROOT/'artifacts/reports/f17-switch-representation.json';archive=ROOT/'artifacts/manifests'/('f17-switch-representation-'+receipt['content_hash'])/'receipt.json'
    for p in [target,archive]:
        if p.exists():
            if p.read_bytes()!=canonical(receipt)+b'\n':raise ValueError('Receipt changed')
        else:atomic_json(p,receipt)
    return {'receipt':str(target),'sha256':digest(target),'content_hash':receipt['content_hash']}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--review',type=Path,required=True);x=a.parse_args();print(seal(x.run.resolve(),x.review.resolve()))
