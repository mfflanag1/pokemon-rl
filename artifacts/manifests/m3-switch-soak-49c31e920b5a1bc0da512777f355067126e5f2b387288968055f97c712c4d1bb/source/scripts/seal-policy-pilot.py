"""Verify the complete development tape and bind it to source, regression and review."""
import argparse
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_pilot.provenance import admission,roster,source_manifest
from pokemon_pilot.campaign import analyze,cell_name
from pokemon_showdown.campaign import sealed
from pokemon_showdown.loop import Executables,check_report
from pokemon_showdown.checkpoint import load
from pokemon_soak.campaign import read,fingerprint
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_learner.core import require

REVIEW=ROOT/'artifacts/reviews/m2-policy-pilot'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13]
helpers=runpy.run_path(str(ROOT/'scripts/seal-binary-policy.py'))
write_once=helpers['write_once'];node_pass_count=helpers['node_pass_count']


def main(directory,audit_smoke=False):
    out=Path(directory).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Run path')
    r=sealed(read(out/'report.json'));a=read(out/'preregistration.json');plan=a['plan']
    require(r['status']=='passed' and r['mode']==('smoke' if audit_smoke else 'development'),'Run mode/status')
    require(r['full_g2']==r['full_g3']=='open' and r['prime_used'] is False
            and r['final_evaluation_used'] is False and r['strategic_strength_claim'] is False
            and r['final_reservation_unchanged'] is True,'Scope flags')
    require(a==admission() and digest(a)==r['admission_hash'],'Admission/source/runtime drift')
    seeds,updates,pairs=roster(plan,audit_smoke);exe=Executables()
    training=read(out/'training.json');require(training==r['training'] and [t['seed'] for t in training]==seeds,'Training roster')
    for t in training:
        require([c['update'] for c in t['checkpoints']]==updates,'Checkpoint roster')
        for c in t['checkpoints']:
            expected=f"checkpoint-{t['seed']}-{c['update']}.json"
            require(c['file']==expected and file_hash(out/expected)==c['sha256'],'Checkpoint identity')
            runner=load(out/expected,exe)
            require(runner.seed==t['seed'] and int(runner.learner.version)==c['update']
                    and int(runner.learner.next_game)==c['games']==64*c['update']
                    and runner.decisions==c['decisions'],'Training counters')
            metrics=runner.run()
            require(fingerprint(runner.learner)==c['next_learner_fingerprint']
                    and fingerprint(metrics)==c['next_metrics_fingerprint'],'Original next-update fingerprints')
    index=read(out/'matches-index.json')
    names=[cell_name(s,u,o) for s in seeds for u in updates for o in plan['opponents']]
    require([x['file'] for x in index]==names,'Evaluation roster')
    for x in index:
        require(x['games']==2*pairs and file_hash(out/x['file'])==x['sha256'],'Evaluation file checksum')
    actual=analyze(out,exe,plan,audit_smoke)
    require(actual==r['analysis'],'Persisted report/complete action tape disagreement')
    require(actual['games']==len(names)*2*pairs,'Evaluation count')
    reference=read(out/'reference.json')
    current=json.loads(subprocess.check_output(['node','--input-type=module','-e',
        "import {describe} from './packages/reference-node/engine.mjs'; console.log(JSON.stringify(describe()));"],cwd=ROOT,text=True))
    require(reference==current and reference['commit']=='942cda71d14bf0618bd57991708a2e2734b6daba'
            and reference['source_dirty'] is False,'Pinned Showdown manifest')
    if audit_smoke:
        print(json.dumps({'status':'smoke-evidence-verified','games':actual['games'],'promotion':False}));return
    v=read(REVIEW/'verification.json');log=(REVIEW/'npm-test.log').read_text();manifest=source_manifest()
    require(v['status']=='passed' and v['exit_code']==0 and v['sources_unchanged'] is True and
            v['source_files']==manifest and v['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Regression source binding')
    counts=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(counts==SUITES and node_pass_count(log)==68 and log.count('\nOK\n')==len(SUITES),'Regression coverage')
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md','delegate/outcome.json'):
        require((REVIEW/name).is_file(),'Missing review '+name)
    outcome=read(REVIEW/'delegate/outcome.json')
    require(outcome['verdict']=='accepted' and outcome['evidence_sha256']==file_hash(REVIEW/'delegate-verification.md'),'Delegate outcome binding')
    sources=dict(manifest)
    for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:sources[str(p.relative_to(ROOT))]=file_hash(p)
    history=sorted((ROOT/'artifacts/runs').glob('m2-policy-pilot-*'))
    evidence={'artifacts/reports/m1-policy-soak.json':file_hash(ROOT/'artifacts/reports/m1-policy-soak.json')}
    for d in [*history,REVIEW]:
        for p in sorted(d.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:evidence[str(p.relative_to(ROOT))]=file_hash(p)
    receipt={'format':'restricted-policy-pilot-receipt-1','status':'complete-development-evaluation',
        'full_g2':'open','full_g3':'open','prime_used':False,'final_evaluation_used':False,
        'strategic_strength_claim':False,'development_screen_passed':actual['all_seed_screen_passed'],
        'run':str(out.relative_to(ROOT)),'campaign':r,'predecessor':a['predecessor'],
        'tests':{'node':68,'python':sum(counts),'suite_counts':counts},'source_files':sources,'evidence_files':evidence}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m2-policy-pilot-'+receipt['content_hash'])
    for kind,files in [('source',sources),('evidence',evidence)]:
        for p,h in files.items():
            require(file_hash(ROOT/p)==h,'Live file drift');write_once(archive/kind/p,(ROOT/p).read_bytes())
            require(file_hash(archive/kind/p)==h,'Archived file drift')
    write_once(archive/'receipt.json',wire(receipt)+b'\n')
    write_once(ROOT/'artifacts/reports/m2-policy-pilot.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],'games':actual['games'],
                      'development_screen_passed':actual['all_seed_screen_passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--audit-smoke',action='store_true')
    args=p.parse_args();main(args.directory,args.audit_smoke)
