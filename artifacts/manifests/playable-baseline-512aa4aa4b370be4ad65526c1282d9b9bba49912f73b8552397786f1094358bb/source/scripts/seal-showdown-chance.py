"""Seal exact finite-word mappings and fresh public-duel views; G2/G3 stay open."""
import argparse
import json
from pathlib import Path
import re
import runpy
import sys
import jax
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_showdown.campaign import admission,capture_index,word_audit,sealed,CAPTURE,WORDS
from pokemon_showdown.loop import Executables
from pokemon_showdown.checkpoint import load,state
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require

REVIEW=ROOT/'artifacts/reviews/m1-showdown-chance'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9]
OLD=runpy.run_path(str(ROOT/'scripts/seal-binary-policy.py'))
write_once=OLD['write_once']
node_pass_count=OLD['node_pass_count']
TOTAL_KEYS=('battles','turns','events','player_view_comparisons_including_padding','pending_player_comparisons','disagreements')


def read(path):return json.loads(path.read_bytes())


def validate_player(player,rows,index):
    require(player['status']=='passed' and player['fresh_distinct_seeds']==10000
            and player['disjoint_from_prior_reference_seeds'] is True,'Fresh player campaign')
    require(len(rows)==len(index['shards'])==100,'Player shard coverage')
    sums={k:0 for k in TOTAL_KEYS}
    for row,shard in zip(rows,index['shards']):
        require(row['first']==shard['first'] and row['capture_sha256']==shard['sha256']
                and row['battles']==shard['count']==100,'Player shard identity')
        n,t,e=row['battles'],row['turns'],row['events']
        require(2*n<=t<=4*n and 2*t-n<=e<=2*t,'Turn/move accounting')
        require(row['player_view_comparisons_including_padding']==18*n
                and row['pending_player_comparisons']==2*t and row['disagreements']==0,'Player view accounting')
        for k in sums:sums[k]+=row[k]
    require(all(player[k]==v for k,v in sums.items()) and sums['battles']==10000,'Player aggregate')
    chance=player['raw_chance_checks']
    require(set(chance)=={'crit','roll','tie'} and chance['crit']==chance['roll']==sums['events']
            and type(chance['tie']) is int and 0<chance['tie']<=2*sums['turns'],'Raw chance coverage')


def validate_scope(result):
    require(result['status']=='passed' and result['full_g2']==result['full_g3']=='open','Scope/status')
    require(result['two_hour_qualification'] is False and result['prime_used'] is False
            and result['final_reservation_unchanged'] is True,'Qualification scope')
    integration=result['integration']
    require(integration['status']=='passed' and integration['two_hour_qualification'] is False
            and integration['traces']=={'collect':1,'update':1,'cycle':1},'Integration scope/traces')


def main(directory):
    run=Path(directory).resolve();require(run.is_relative_to(ROOT/'artifacts/runs'),'Run directory')
    result=sealed(read(run/'report.json'));admitted=read(run/'admission.json')
    validate_scope(result)
    require(result['admission_hash']==digest(admitted) and admitted==admission(),'Source/runtime drift')
    require(result['mapping']==word_audit(),'Exact word audit')
    _,index=capture_index();rows=read(run/'player-shards.json')
    validate_player(result['player'],rows,index)
    c=admitted['contract'];integration=result['integration'];exe=Executables()
    require([r['seed'] for r in integration['seeds']]==c['integration_seeds'],'Integration seeds')
    for r in integration['seeds']:
        path=run/f"checkpoint-{r['seed']}.json"
        require(r['updates']==c['integration_updates']==8 and r['games']==r['updates']*64
                and r['next_cycle_exact'] is True and file_hash(path)==r['checkpoint_sha256'],'Integration evidence')
        a=load(path,exe);b=load(path,exe)
        require(a.seed==r['seed'] and int(a.learner.version)==r['updates'] and int(a.learner.next_game)==r['games'],'Checkpoint counters')
        ma,mb=a.run(),b.run()
        require(state(a)==state(b),'Restored next-cycle bytes')
        require(all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(ma),jax.tree.leaves(mb))),'Restored next-cycle metrics')
    verification=read(REVIEW/'verification.json')
    manifest=runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()
    require(verification['status']=='passed' and verification['exit_code']==0 and verification['sources_unchanged'] is True
            and verification['source_files']==manifest and verification['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Regression source/log binding')
    log=(REVIEW/'npm-test.log').read_text();suites=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(node_pass_count(log)==65 and suites==SUITES and log.count('\nOK\n')==len(suites),'Regression suite coverage')
    paths=re.findall(r"'discover', '-s', '(tests/[^']+)'",(ROOT/'scripts/test.mjs').read_text())
    require(len(paths)==len(suites) and paths[-1]=='tests/showdown-chance','Test suite identities')
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md','delegate/outcome.json'):
        require((REVIEW/name).is_file(),'Missing review '+name)
    outcome=read(REVIEW/'delegate/outcome.json')
    require(outcome['verdict']=='accepted' and outcome['evidence_sha256']==file_hash(REVIEW/'delegate-verification.md'),
            'Verified delegation outcome')
    sources=dict(manifest)
    # Preserve the selected actual upstream implementation as well as its pin.
    sources.update(read(CAPTURE/'plan.json')['sources'])
    for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:sources[str(path.relative_to(ROOT))]=file_hash(path)
    evidence={str(WORDS.relative_to(ROOT)):file_hash(WORDS)}
    historical=sorted((ROOT/'artifacts/runs').glob('m1-showdown-chance-v*'))
    for directory in dict.fromkeys([run,CAPTURE,REVIEW,*historical]):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:evidence[str(path.relative_to(ROOT))]=file_hash(path)
    receipt={'format':'m1-showdown-chance-receipt-1','status':'complete-for-declared-scope',
        'full_g2':'open','full_g3':'open','two_hour_qualification':False,'prime_used':False,
        'campaign':result,'run':str(run.relative_to(ROOT)),'predecessor':admitted['predecessor'],
        'campaign_history':{str(d.relative_to(ROOT)):sealed(read(d/'report.json'))['content_hash'] for d in historical if (d/'report.json').is_file()},
        'tests':{'node':65,'python':sum(suites),'suite_counts':suites,'python_suites':dict(zip(paths,suites))},
        'source_files':sources,'evidence_files':evidence}
    receipt['content_hash']=digest(receipt)
    archive=ROOT/'artifacts/manifests'/('m1-showdown-chance-'+receipt['content_hash'])
    for kind,files in (('source',sources),('evidence',evidence)):
        for path,sha in files.items():
            data=(ROOT/path).read_bytes();require(file_hash(ROOT/path)==sha,'Archive source drift')
            write_once(archive/kind/path,data)
    write_once(archive/'receipt.json',wire(receipt)+b'\n')
    write_once(ROOT/'artifacts/reports/m1-showdown-chance.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],
                      'source_files':len(sources),'evidence_files':len(evidence),'tests':receipt['tests']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');main(parser.parse_args().directory)
