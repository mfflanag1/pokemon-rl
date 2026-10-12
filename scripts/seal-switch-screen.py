"""Audit every stored decision and seal the preregistered development result."""
import argparse
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_switch_engine import learner
from pokemon_switch_engine.scalar import require
from pokemon_switch_screen import checkpoint as cp
from pokemon_switch_screen.provenance import PLAN,PREDECESSOR,admission,read,roster
from pokemon_switch_screen.campaign import analyze,cell_name,checkpoint_name

REVIEW=ROOT/'artifacts/reviews/m3-switch-screen'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13,6,17,17,13]
write_once=runpy.run_path(str(ROOT/'scripts/seal-switch-engine.py'))['write_once']


def audit(directory,*,smoke=False):
    out=Path(directory).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Run path')
    require(not list(out.glob('*.partial')),'Partial write evidence')
    r=read(out/'report.json');a=read(out/'preregistration.json')
    require(r['content_hash']==digest({k:v for k,v in r.items() if k!='content_hash'}),'Report checksum')
    require(r['status']=='passed' and r['mode']==('smoke' if smoke else 'development'),'Run status/mode')
    require(r['full_g2']==r['full_g3']==r['full_g4']=='open' and r['prime_used'] is False
            and r['final_evaluation_used'] is False and r['public_server_used'] is False
            and r['public_ladder_strength']=='unqualified' and r['final_reservation_unchanged'] is True,'Scope flags')
    require(a==admission() and r['admission_hash']==digest(a),'Preregistration/source/runtime/reservation drift')
    if smoke:require(r['preproduction_freeze_sha256'] is None,'Smoke freeze status')
    else:
        freeze=REVIEW/'preregistration.json';f=read(freeze)
        require(r['preproduction_freeze_sha256']==file_hash(freeze) and f['admission']==a and f['frozen_at']<r['started_at'],'Pre-production freeze binding')
    seeds,updates,pairs=roster(smoke);training=read(out/'training.json')
    require(training==r['training'] and [t['seed'] for t in training]==seeds,'Training roster')
    for t in training:
        seed=t['seed'];require([c['update'] for c in t['checkpoints']]==updates,'Checkpoint roster')
        previous=0
        for c in t['checkpoints']:
            u=c['update'];require(c['file']==checkpoint_name(seed,u) and file_hash(out/c['file'])==c['sha256'],'Checkpoint file binding')
            state=cp.load(out/c['file'],seed,a)
            require(int(state.version)==u and int(state.step)==2*u and int(state.next_game)==c['games']==32*u,'Checkpoint counters')
            require(all(type(c[k]) is int and c[k]>=0 for k in ('decisions','terminal','truncated')) and
                    c['terminal']+c['truncated']==c['games'] and previous<=c['decisions']<=28*c['games'],'Training counts')
            previous=c['decisions']
            require(cp.fingerprint(state)==c['state'] and cp.result_fingerprints(cp.transaction(state,seed))==c['next'],'Original-next-update fingerprints')
            if u==0:require(cp.fingerprint(state)==cp.fingerprint(learner.start(seed)) and c['decisions']==0,'Fresh initialization identity')
    index=read(out/'matches-index.json');names=[cell_name(s,u,o) for s in seeds for u in updates for o in PLAN['opponents']]
    require([x['file'] for x in index]==names,'Complete evaluation roster')
    for x in index:require(x['games']==2*pairs and file_hash(out/x['file'])==x['sha256'],'Evaluation file checksum')
    reference=json.loads(subprocess.check_output(['node','--input-type=module','-e',
        "import {reference} from './packages/switch-reference/profile.mjs';console.log(JSON.stringify(reference()));"],cwd=ROOT,text=True))
    require(reference==read(out/'reference.json') and reference['reference']['source_dirty'] is False,'Pinned reference runtime')
    actual=analyze(out,a,smoke);require(actual==r['analysis'],'Complete tape/report mismatch')
    require(actual['games']==len(names)*2*pairs==actual['terminal']+actual['truncated'],'Outcome accounting')
    reference_replay=json.loads(subprocess.check_output(['node','packages/switch-screen/audit.mjs',str(out)],cwd=ROOT,text=True))
    require(reference_replay=={'status':'passed','games':actual['games'],'unpatched_reference':True},'Fresh complete reference replay')
    return {'status':'verified','mode':r['mode'],'games':actual['games'],'checkpoint_replays':len(seeds)*len(updates),
            'reference_game_replays':reference_replay['games'],
            'report_sha256':file_hash(out/'report.json'),'all_seed_screen_passed':actual['all_seed_screen_passed']}


def seal(directory):
    out=Path(directory).resolve();audited=audit(out);a=admission();r=read(out/'report.json')
    v=read(REVIEW/'verification.json');log=(REVIEW/'npm-test.log').read_text()
    require(v['status']=='passed' and v['exit_code']==0 and v['sources_unchanged'] is True
            and v['source_files']==a['source_files'] and v['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Source-bound regression')
    node=list(map(int,re.findall(r'ℹ pass (\d+)',log)));counts=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(node==[237,3] and counts==SUITES and log.count('\nOK\n')==len(SUITES),'Regression coverage')
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md','delegate/outcome.json','preflight.json','preregistration.json'):
        require((REVIEW/name).is_file(),'Missing review/preflight: '+name)
    require(read(REVIEW/'preregistration.json')['admission']==read(out/'preregistration.json'),'Pre-production preregistration')
    outcome=read(REVIEW/'delegate/outcome.json');require(outcome['verdict']=='accepted' and outcome['evidence_sha256']==file_hash(REVIEW/'delegate-verification.md'),'Delegate verification')
    preflight=read(REVIEW/'preflight.json');require(preflight['audit']==audit(ROOT/preflight['run'],smoke=True),'Preflight evidence')
    sources={**a['source_files'],**{str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]}}
    evidence={'artifacts/reports/m3-switch-soak.json':file_hash(ROOT/'artifacts/reports/m3-switch-soak.json')}
    for folder in (out,REVIEW,*sorted((ROOT/'artifacts/runs').glob('m3-switch-screen-smoke-*'))):
        evidence.update({str(p.relative_to(ROOT)):file_hash(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    receipt={'format':'switch-screen-receipt-1','status':'complete-restricted-development-screen','run':str(out.relative_to(ROOT)),
             'predecessor':PREDECESSOR,'audit':audited,'report':r,'tests':{'node':sum(node),'python':sum(counts),'suite_counts':counts},
             'source_files':sources,'evidence_files':evidence,'full_g2':'open','full_g3':'open','full_g4':'open',
             'public_ladder_strength':'unqualified','public_server_used':False,'final_evaluation_used':False,'prime_used':False}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m3-switch-screen-'+receipt['content_hash'])
    for kind,files in [('source',sources),('evidence',evidence)]:
        for name,h in files.items():
            require(file_hash(ROOT/name)==h,'Live source/evidence drift');write_once(archive/kind/name,(ROOT/name).read_bytes())
            require(file_hash(archive/kind/name)==h,'Archived source/evidence drift')
    write_once(archive/'receipt.json',wire(receipt)+b'\n');write_once(ROOT/'artifacts/reports/m3-switch-screen.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],'audit':audited,'tests':receipt['tests']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--audit-smoke',action='store_true');args=p.parse_args()
    if args.audit_smoke:print(json.dumps(audit(args.directory,smoke=True)))
    else:seal(args.directory)
