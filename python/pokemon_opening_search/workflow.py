"""Single-use persistent F33 execution, fresh replay, deterministic receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from pokemon_full_train.storage import ROOT,atomic_json,canonical,digest,admission
from .campaign import PLAN

REVIEW=ROOT/'artifacts/reviews/f33-opening-search'
OUTER=ROOT/'artifacts/reviews'
RUN=ROOT/'artifacts/runs/f33-opening-search-v1'
REPLAY=REVIEW/'replay'
QUAL=REVIEW/'qualification-v2'
RECEIPT=ROOT/'artifacts/reports/f33-opening-search.json'
DOC=ROOT/'docs/F33_OPENING_SEARCH.md'


def read(path):return json.loads(Path(path).read_text())


def measured_doc(report):
    text=DOC.read_text().replace('Status: implementation and engineering qualification; no F33 scientific result yet.',
        'Status: completed; production and exhaustive fresh replay passed. Learning/search benefit is reported separately below.')
    rows=['\n## Measured result\n',f"Production took {report['seconds']:.2f} CPU-backed wall seconds; "
        f"{report['main_games']:,} main evaluation games and {report['audit_games']:,} fixed-position audit rollouts; zero optimizer updates. "
        f"Peak parent RSS was {report['peak_parent_rss_mib']:.1f} MiB (excludes workers).\n",
        '| Seed | Tactical policy wins | Search wins | Search applied | Tactical benefit gate |',
        '|---|---:|---:|---:|---|']
    for seed,result in report['seeds'].items():
        c=result['comparisons']['tactical'];s=result['search']
        rows.append(f"| {seed} | {c['initial']['learner_wins']}/{c['initial']['games']} | "
            f"{c['final']['learner_wins']}/{c['final']['games']} | {s['applied']}/{s['roots']} | {result['gate']['passed']} |")
    rows.extend(['',f"All-seed benefit gate: **{report['all_seed_benefit_gate']}**. "
        'All nine opponent contrasts, unresolved cutoff bounds, terminal-only values, fixed-position rankings, '
        'fallback reasons and latency measurements are retained in the production report. '
        'These opening-only development results do not establish general search strength or public ladder strength.',
        '', 'Receipt: `artifacts/reports/f33-opening-search.json`.'])
    DOC.write_text(text+'\n'.join(rows)+'\n')


def seal():
    production=read(RUN/'report.json');replay=read(REPLAY/'report.json')
    if production['status']!='passed' or replay['status']!='passed' or not replay['fresh_replay']:
        raise ValueError('Successful independent replay required')
    plan=read(PLAN)
    if production['main_games']!=plan['expected_main_games'] or replay['main_games']!=production['main_games'] or replay['audit_games']!=production['audit_games']:
        raise ValueError('Complete sample counts required')
    for stage in ('production','replay'):
        if read(REVIEW/f'{stage}-exit.json')['exit_code']!=0:raise ValueError('Stage exit failure')
    manifest=read(RUN/'inputs/manifest.json');admission(RUN,manifest)
    evidence={}
    for base in (RUN,REVIEW):
        for path in sorted(base.rglob('*')):
            if path.is_file():evidence[str(path.relative_to(ROOT))]=digest(path)
    evidence[str(DOC.relative_to(ROOT))]=digest(DOC)
    body={'schema_version':'f33-opening-search-receipt-v1','status':'passed',
        'run':str(RUN.relative_to(ROOT)),'replay':str(REPLAY.relative_to(ROOT)),
        'report_sha256':digest(RUN/'report.json'),'replay_report_sha256':digest(REPLAY/'report.json'),
        'input_sha256':digest(RUN/'inputs/manifest.json'),'source_files':manifest['sources'],
        'evidence_files':evidence,'counts':{'main_games':production['main_games'],'audit_games':production['audit_games'],
            'optimizer_updates':0,'physical_main_games_including_replay':2*production['main_games'],
            'physical_audit_games_including_replay':2*production['audit_games']},
        'all_seed_benefit_gate':production['all_seed_benefit_gate'],
        'limitations':plan['limits'],'checkpoint_identities':production['checkpoint_identities']}
    content_hash=hashlib.sha256(canonical(body)).hexdigest();receipt={**body,'content_hash':content_hash}
    archive=ROOT/'artifacts/manifests'/('f33-opening-search-'+content_hash)
    data=canonical(receipt)+b'\n'
    if RECEIPT.exists() and RECEIPT.read_bytes()!=data:raise ValueError('Receipt changed on repeat sealing')
    for category,files in [('source',body['source_files']),('evidence',evidence)]:
        for rel,expected in files.items():
            target=archive/category/rel
            if target.exists():
                if digest(target)!=expected:raise ValueError('Archived evidence mismatch')
            else:
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    archive.mkdir(parents=True,exist_ok=True)
    if (archive/'receipt.json').exists() and (archive/'receipt.json').read_bytes()!=data:raise ValueError('Archived receipt mismatch')
    (archive/'receipt.json').write_bytes(data);atomic_json(RECEIPT,receipt)
    return receipt


def verify():
    receipt=read(RECEIPT);body=dict(receipt);content_hash=body.pop('content_hash')
    if hashlib.sha256(canonical(body)).hexdigest()!=content_hash:raise ValueError('Content hash')
    archive=ROOT/'artifacts/manifests'/('f33-opening-search-'+content_hash)
    if RECEIPT.read_bytes()!=(archive/'receipt.json').read_bytes():raise ValueError('Receipt byte equality')
    for category,key in [('source','source_files'),('evidence','evidence_files')]:
        if not receipt[key]:raise ValueError('Empty hash inventory')
        for rel,expected in receipt[key].items():
            if digest(ROOT/rel)!=expected or digest(archive/category/rel)!=expected:raise ValueError('Hash drift: '+rel)
    return {'status':'passed','receipt_sha256':digest(RECEIPT),'content_hash':content_hash,
        'live_and_archived_source_checks':2*len(receipt['source_files']),
        'live_and_archived_evidence_checks':2*len(receipt['evidence_files']),
        'archived_receipt_identical':True}


def stage(name,args):
    with (REVIEW/f'{name}.log').open('xb') as log:
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pokemon_opening_search.campaign',*args],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(REVIEW/f'{name}-exit.json',{'exit_code':result.returncode})
    if result.returncode:raise RuntimeError(name+' failed; evidence preserved')


def main():
    claim=OUTER/'f33-opening-search-workflow-claim.json'
    with claim.open('x') as out:json.dump({'pid':os.getpid(),'started_at':time.time()},out)
    try:
        qualification=read(QUAL/'report.json');approval=read(REVIEW/'launch-admission.json')
        if qualification['status']!='passed' or approval['status']!='admitted' or approval['qualification_sha256']!=digest(QUAL/'report.json'):
            raise ValueError('Qualification/review admission')
        admission(QUAL,read(QUAL/'inputs/manifest.json'))
        if approval['plan_sha256']!=digest(PLAN):raise ValueError('Plan drift after review')
        for rel,expected in approval['review_files'].items():
            if digest(ROOT/rel)!=expected:raise ValueError('Review drift: '+rel)
        stage('production',['--run',str(RUN)])
        stage('replay',['--run',str(REPLAY),'--replay-of',str(RUN)])
        measured_doc(read(RUN/'report.json'))
        seal();first=RECEIPT.read_bytes();check=verify()
        seal()
        if first!=RECEIPT.read_bytes():raise ValueError('Repeat receipt bytes differ')
        second=verify()
        if check!=second:raise ValueError('Repeat integrity result differs')
        atomic_json(OUTER/'f33-opening-search-completion.json',
            {**check,'identical_repeat_sealing':True,'exit_code':0})
        atomic_json(OUTER/'f33-opening-search-workflow-exit.json',{'exit_code':0})
    except BaseException as error:
        atomic_json(OUTER/'f33-opening-search-workflow-failure.json',
            {'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'scientific_completion_claimed':False})
        atomic_json(OUTER/'f33-opening-search-workflow-exit.json',{'exit_code':1})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    if parser.parse_args().verify:print(json.dumps(verify(),sort_keys=True))
    else:main()
