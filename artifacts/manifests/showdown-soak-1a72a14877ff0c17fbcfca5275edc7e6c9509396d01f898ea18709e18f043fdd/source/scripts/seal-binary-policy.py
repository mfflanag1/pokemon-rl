"""Seal the bounded binary-policy qualification without promoting G2/G3."""
import argparse
import json
import math
from pathlib import Path
import re
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pokemon_binary.qualification import admission
from pokemon_binary.loop import Executables
from pokemon_binary.checkpoint import load
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require

REVIEW=ROOT/'artifacts/reviews/m1-binary-policy'


def read(path):return json.loads(path.read_bytes())


def write_once(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==data,'Changed sealed artifact: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(data)


def node_pass_count(log):
    passed=re.search(r'^(?:#|ℹ) pass (\d+)$',log,re.M)
    failed=re.search(r'^(?:#|ℹ) fail (\d+)$',log,re.M)
    require(passed is not None and failed is not None and int(failed[1])==0,'Node test summary/failures')
    return int(passed[1])


def validate_soak_diagnostics(soak):
    # exp(log(r)) is rounded FP32; permit arithmetic error, never missing support.
    minimum=soak['minimum_exp_behavior_logp'];ratio=soak['maximum_update_ratio_error']
    require(math.isfinite(minimum) and 2**-24*(1-3e-6)<=minimum<=1+3e-6,
            'Behavior probability outside declared law')
    # PPO clipping does not cap the observed ratio. The probability floor does.
    require(math.isfinite(ratio) and 0<=ratio<=2**24+16, 'Update ratio outside declared law')


def main(directory):
    run=Path(directory).resolve();require(run.is_relative_to(ROOT/'artifacts/runs'),'Run directory')
    result=read(run/'report.json');admitted=read(run/'admission.json');q=admitted['contract']['qualification'];p=admitted['plan']
    require(result['content_hash']==digest({k:v for k,v in result.items() if k!='content_hash'}),'Result checksum')
    require(result['status']=='passed' and result['full_g2']==result['full_g3']=='open','Scope/status')
    require(result['prime_used'] is False and result['new_two_hour_soak'] is False and result['strength_evaluation'] is False,'Scope claim')
    require(result['admission_hash']==digest(admitted) and admitted==admission(),'Source/runtime drift')
    require(result['final_reservation_unchanged'] is True and admitted['prior']['prior_gates']['final_reservation_unchanged'] is True,'Final reservation')
    sampling=result['sampling'];n=q['word_probe_rows']+2
    require(sampling['status']=='passed' and sampling['rare_mass_cases']==n and sampling['injected_action_checks']==6*n
            and sampling['integer_thresholds_exact'] is True and sampling['uniform_word_assumption'] is True
            and sampling['sampling_frequency_estimated'] is False and sampling['minimum_mass']==2**-24,'Sampling audit')
    parity=result['parity'];batches=len(p['parity_seeds'])*p['parity_batches_per_seed']
    require(parity['status']=='passed' and parity['batches']==batches and parity['games']==64*batches
            and batches*256<=parity['decisions']<=batches*512 and 0<=parity['max_fraction_of_tolerance']<=1
            and parity['atol']==p['atol']==3e-6 and parity['rtol']==p['rtol']==3e-5,'Collector/fused parity')
    control=result['compile_control']
    require(control['first_call_compilation_events']>0 and control['same_shape_repeat_events']==0
            and result['traces']=={'cycle':1,'collect':1,'update':1},'Compilation controls')
    require([r['seed'] for r in result['seed_runs']]==q['seeds'],'Seed roster')
    for r in result['seed_runs']:
        require(r['updates']==q['updates_per_seed'] and r['games']==r['updates']*64
                and r['games']*4<=r['decisions']<=r['games']*8
                and r['decisions']-r['games']<=r['executed_moves']<=r['decisions'],'Seed accounting')
    soak=result['soak']
    validate_soak_diagnostics(soak)
    require(soak['status']=='passed' and soak['elapsed_seconds']>=soak['target_seconds']==q['soak_seconds']==60,'Short soak')
    require(soak['unexpected_compilations']==0 and soak['warmup_compilation_count']>0,'Soak compilations')
    require(soak['games']==soak['cycles']*64 and soak['games']*4<=soak['decisions']<=soak['games']*8
            and soak['decisions']-soak['games']<=soak['executed_moves']<=soak['decisions'],'Soak accounting')
    require(soak['cycle_seconds']>0 and soak['decisions_per_second']==soak['decisions']/soak['elapsed_seconds']
            and soak['device_memory_headroom_certified'] is False,'Timing/headroom')
    telemetry=[json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
    require(len(telemetry)==soak['telemetry_rows']>=5 and telemetry[-1]['elapsed_seconds']>=50,'Telemetry coverage')
    require(all(a['cycles']<b['cycles'] and a['decisions']<b['decisions'] and a['elapsed_seconds']<b['elapsed_seconds']
                for a,b in zip(telemetry,telemetry[1:])),'Telemetry ordering')
    require(telemetry[-1]['cycles']<=soak['cycles'] and telemetry[-1]['decisions']<=soak['decisions'],'Telemetry endpoint')
    exe=Executables();expected_names=[f'seed-{seed}.json' for seed in q['seeds']]+['soak-start.json','soak-final.json']
    require([r['file'] for r in result['replays']]==expected_names,'Replay roster')
    checkpoints={}
    for item in result['replays']:
        require(item['next_cycle_exact'] is True and file_hash(run/item['file'])==item['sha256'],'Replay evidence')
        runner=load(run/item['file'],exe);require(int(runner.learner.version)==item['version'],'Replay version')
        checkpoints[item['file']]=runner
    for row in result['seed_runs']:
        r=checkpoints[f"seed-{row['seed']}.json"]
        require(r.seed==row['seed'] and int(r.learner.version)==row['updates'] and r.decisions==row['decisions']
                and r.executed_moves==row['executed_moves'],'Seed checkpoint')
    start,end=checkpoints['soak-start.json'],checkpoints['soak-final.json']
    require(start.seed==end.seed==q['soak_seed'] and int(start.learner.version)==p['warmup_cycles']
            and int(end.learner.version)-int(start.learner.version)==soak['cycles']
            and end.decisions-start.decisions==soak['decisions']
            and end.executed_moves-start.executed_moves==soak['executed_moves'],'Soak checkpoint accounting')
    verification=read(REVIEW/'verification.json')
    manifest=runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()
    require(verification['status']=='passed' and verification['exit_code']==0 and verification['sources_unchanged'] is True
            and verification['source_files']==manifest and verification['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Regression source/log binding')
    log=(REVIEW/'npm-test.log').read_text();node=node_pass_count(log)
    suites=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(node==63 and suites==[7,6,15,13,10,15,13,6,25,17,16,12]
            and log.count('\nOK\n')==len(suites),'Regression suite coverage')
    suite_paths=re.findall(r"'discover', '-s', '(tests/[^']+)'",(ROOT/'scripts/test.mjs').read_text())
    require(len(suite_paths)==len(suites) and suite_paths[-1]=='tests/binary-policy','Test suite identities')
    for name in ('claude-review.md','review-dispositions.md','delegate-verification.md'):
        require((REVIEW/name).is_file(),'Missing review '+name)
    sources=dict(manifest)
    for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:sources[str(path.relative_to(ROOT))]=file_hash(path)
    evidence={}
    historical=sorted((ROOT/'artifacts/runs').glob('m1-binary-policy-v*'))
    for directory in dict.fromkeys([run,REVIEW,*historical]):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:evidence[str(path.relative_to(ROOT))]=file_hash(path)
    receipt={'format':'m1-binary-policy-receipt-1','status':'complete-for-declared-scope',
        'full_g2':'open','full_g3':'open','campaign':result,'run':str(run.relative_to(ROOT)),
        'predecessor':admitted['predecessor'],
        'campaign_history':{str(d.relative_to(ROOT)):read(d/'report.json')['content_hash'] for d in historical if (d/'report.json').is_file()},
        'tests':{'node':63,'python':sum(suites),'suite_counts':suites,'python_suites':dict(zip(suite_paths,suites))},
        'source_files':sources,'evidence_files':evidence}
    receipt['content_hash']=digest(receipt)
    archive=ROOT/'artifacts/manifests'/('m1-binary-policy-'+receipt['content_hash'])
    for kind,files in (('source',sources),('evidence',evidence)):
        for path,sha in files.items():
            data=(ROOT/path).read_bytes();require(file_hash(ROOT/path)==sha,'Archive source drift')
            write_once(archive/kind/path,data)
    write_once(archive/'receipt.json',wire(receipt)+b'\n')
    write_once(ROOT/'artifacts/reports/m1-binary-policy.json',wire(receipt)+b'\n')
    print(json.dumps({'status':receipt['status'],'content_hash':receipt['content_hash'],'source_files':len(sources),'evidence_files':len(evidence),'tests':receipt['tests']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');main(parser.parse_args().directory)
