"""Fail-closed, source-bound receipt for the selected combination's two-hour run."""
import argparse
import json
from pathlib import Path
import re
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_soak.campaign import admission,read,fingerprint,counters
from pokemon_soak.gates import validate_summary
from pokemon_showdown.campaign import sealed
from pokemon_showdown.loop import Executables
from pokemon_showdown.checkpoint import load
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require

REVIEW=ROOT/'artifacts/reviews/m1-policy-soak'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6]
HELPERS=runpy.run_path(str(ROOT/'scripts/seal-binary-policy.py'))
write_once=HELPERS['write_once'];node_pass_count=HELPERS['node_pass_count']


def validate_telemetry(rows,soak,plan,*,smoke=False):
    target=plan['smoke']['seconds'] if smoke else plan['qualification_seconds']
    interval=plan['smoke']['telemetry_seconds'] if smoke else plan['telemetry_seconds']
    count=target//interval
    require(len(rows)==soak['telemetry_rows'] and count-1<=len(rows)<=count,'Telemetry coverage')
    initial=soak['initial_counters'];previous=None
    for i,row in enumerate(rows,1):
        require(row['scheduled_seconds']==i*interval,'Telemetry schedule')
        require(row['scheduled_seconds']<=row['elapsed_seconds']<=row['scheduled_seconds']+plan['maximum_host_gap_seconds'],
                'Telemetry delay')
        require(row['elapsed_seconds']<=soak['elapsed_seconds'] and 0<row['cycles']<=soak['cycles'],'Telemetry endpoint')
        require(row['games']==64*row['cycles'] and 4*row['games']<=row['decisions']<=8*row['games']
                and row['decisions']-row['games']<=row['executed_moves']<=row['decisions'],'Telemetry counters')
        for field,total in [('updates','cycles'),('next_game','games'),('decisions','decisions'),('executed_moves','executed_moves')]:
            require(row['counters'][field]==initial[field]+row[total] and row[total]<=soak[total],'Telemetry boundary linkage')
        require(row['unexpected_compilations']==0 and row['maximum_host_gap_seconds']<=plan['maximum_host_gap_seconds'],
                'Telemetry stability')
        require(0<row['memory']['peak_process_rss_bytes']<=soak['peak_process_rss_bytes'],'Memory telemetry')
        if previous:
            require(all(row[k]>previous[k] for k in ('elapsed_seconds','cycles','decisions','executed_moves')),'Telemetry order')
        previous=row
    require(rows[-1]['elapsed_seconds']>=target-interval,'Final telemetry missing')


def verify_replays(run,replays,exe,names,seed):
    require([p['file'] for p in replays]==names,'Replay roster')
    boundaries=[]
    for item in replays:
        path=run/item['file'];require(file_hash(path)==item['sha256'] and item['original_vs_restored_exact'] is True,'Replay evidence')
        restored=load(path,exe);require(restored.seed==item['seed']==seed and counters(restored)==item['counters'],'Replay boundary')
        boundaries.append(counters(restored));metrics=restored.run()
        require(fingerprint(restored.learner)==item['next_learner_fingerprint']
                and fingerprint(metrics)==item['next_metrics_fingerprint'],'Persisted original next-cycle fingerprints')
    return boundaries


def historical_report(directory):
    path=directory/'report.json'
    if not path.is_file():return {'status':'missing-report'}
    doc=read(path)
    return {'status':doc.get('status','unknown'),'file_sha256':file_hash(path),
            'content_hash':sealed(doc)['content_hash'] if 'content_hash' in doc else None}


def main(directory,audit_smoke=False):
    run=Path(directory).resolve();require(run.is_relative_to(ROOT/'artifacts/runs'),'Run directory')
    r=sealed(read(run/'report.json'));a=read(run/'admission.json');plan=a['plan'];s=r['soak']
    if audit_smoke:
        require(r['status']=='passed' and r['mode']=='smoke' and r['two_hour_qualification'] is False,'Smoke audit scope')
    else:validate_summary(r,plan)
    target=plan['smoke']['seconds'] if audit_smoke else plan['qualification_seconds']
    interval=plan['smoke']['telemetry_seconds'] if audit_smoke else plan['telemetry_seconds']
    checkpoint=plan['smoke']['checkpoint_seconds'] if audit_smoke else plan['checkpoint_seconds']
    require(target<=s['elapsed_seconds']<=target+plan['maximum_host_gap_seconds'],'Timed window bounds')
    require(a==admission() and digest(a)==r['admission_hash'],'Admission/source/runtime drift')
    rows=[json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
    validate_telemetry(rows,s,plan,smoke=audit_smoke)
    require(read(run/'progress.json')==rows[-1],'Progress/telemetry linkage')
    require(sum(s['latency_histogram']['counts'])==s['cycles'],'Latency coverage')
    require(s['decisions_per_elapsed_second']==s['decisions']/s['elapsed_seconds'],'Elapsed throughput')
    periodic=len(range(checkpoint,target,checkpoint))
    replays=read(run/'replays.json');names=['soak-start.json',*[f'soak-boundary-{i:02d}.json' for i in range(1,periodic+1)],'soak-final.json']
    require([p['file'] for p in replays]==names and len(replays)==s['replays'],'Replay roster')
    boundaries=verify_replays(run,replays,Executables(),names,plan['seed'])
    require(boundaries[0]==s['initial_counters'] and boundaries[-1]==s['final_counters'],'Endpoint checkpoints')
    require(all(x['updates']<y['updates'] for x,y in zip(boundaries,boundaries[1:])),'Checkpoint order')
    # Check periodic snapshot counters against bracketing minute telemetry, not just their order.
    for i,b in enumerate(boundaries[1:-1],1):
        j=i*checkpoint//interval
        before=rows[j-1];after=rows[j] if j<len(rows) else rows[-1]
        require(before['counters']['updates']<=b['updates']<=after['counters']['updates'],'Checkpoint schedule linkage')
    if audit_smoke:
        print(json.dumps({'status':'smoke-evidence-verified','two_hour_qualification':False,'replays':len(replays)}))
        return
    v=read(REVIEW/'verification.json');log=(REVIEW/'npm-test.log').read_text()
    manifest=runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()
    require(v['status']=='passed' and v['exit_code']==0 and v['sources_unchanged'] is True
            and v['source_files']==manifest and v['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Regression binding')
    counts=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(node_pass_count(log)==65 and counts==SUITES and log.count('\nOK\n')==len(SUITES),'Regression coverage')
    suite_paths=re.findall(r"'discover', '-s', '(tests/[^']+)'",(ROOT/'scripts/test.mjs').read_text())
    require(len(suite_paths)==len(counts) and suite_paths[-1]=='tests/soak','Test suite identities')
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md','delegate/outcome.json'):
        require((REVIEW/name).is_file(),'Missing review '+name)
    outcome=read(REVIEW/'delegate/outcome.json')
    require(outcome['verdict']=='accepted' and outcome['evidence_sha256']==file_hash(REVIEW/'delegate-verification.md'),'Delegate verification')
    sources=dict(manifest)
    for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:sources[str(path.relative_to(ROOT))]=file_hash(path)
    evidence={'artifacts/reports/m1-showdown-chance.json':file_hash(ROOT/'artifacts/reports/m1-showdown-chance.json')}
    history=sorted((ROOT/'artifacts/runs').glob('m1-policy-soak-*'))
    for directory in dict.fromkeys([run,REVIEW,*history]):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:evidence[str(path.relative_to(ROOT))]=file_hash(path)
    receipt={'format':'m1-policy-soak-receipt-1','status':'complete-for-declared-scope','two_hour_qualification':True,
             'full_g2':'open','full_g3':'open','prime_used':False,'strength_evaluation':False,
             'campaign':r,'run':str(run.relative_to(ROOT)),'predecessor':a['predecessor'],
             'tests':{'node':65,'python':sum(counts),'suite_counts':counts,'python_suites':dict(zip(suite_paths,counts))},
             'campaign_history':{str(d.relative_to(ROOT)):historical_report(d) for d in history},
             'source_files':sources,'evidence_files':evidence}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m1-policy-soak-'+receipt['content_hash'])
    for kind,files in (('source',sources),('evidence',evidence)):
        for path,sha in files.items():
            require(file_hash(ROOT/path)==sha,'Archive input drift');write_once(archive/kind/path,(ROOT/path).read_bytes())
            require(file_hash(archive/kind/path)==sha,'Archive output drift')
    write_once(archive/'receipt.json',wire(receipt)+b'\n')
    write_once(ROOT/'artifacts/reports/m1-policy-soak.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],'tests':receipt['tests'],
                      'elapsed_seconds':s['elapsed_seconds'],'decisions':s['decisions']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--audit-smoke',action='store_true')
    args=parser.parse_args();main(args.directory,args.audit_smoke)
