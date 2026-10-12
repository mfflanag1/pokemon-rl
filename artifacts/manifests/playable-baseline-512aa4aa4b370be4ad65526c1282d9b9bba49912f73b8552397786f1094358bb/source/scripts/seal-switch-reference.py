"""Seal the M3 reference scope after source-bound regression and complete audit."""
import argparse
import re
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from pokemon_switch.provenance import admission,read,source_manifest
from pokemon_switch.audit import validate,read_bundle
from pokemon_switch.contract import require
from pokemon_learner.checkpoint import wire,digest,file_hash
helpers=runpy.run_path(str(ROOT/'scripts/seal-binary-policy.py'))
write_once=helpers['write_once'];node_pass_count=helpers['node_pass_count']
REVIEW=ROOT/'artifacts/reviews/m3-switch-reference'
SUITES=[7,6,15,13,10,15,13,6,25,17,16,12,9,6,13,6]


def main(path):
    out=Path(path).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Run path')
    r=read(out/'report.json');a=read(out/'admission.json')
    require(r['status']=='passed' and r['content_hash']==digest({k:v for k,v in r.items() if k!='content_hash'}),'Report checksum/status')
    require(a==admission() and digest(a)==r['admission_hash'],'Source/runtime admission')
    actual=validate(read_bundle(out/'original/fixtures.json.gz'))
    require(all(r[k]==v for k,v in actual.items()),'Complete evidence audit')
    summary=read(out/'original/summary.json')
    require(summary['fixture_hash']==actual['fixture_hash'] and r['coverage']==summary['coverage'],'Producer summary binding')
    require(r['exact_regeneration'] is True and r['final_reservation_unchanged'] is True,'Replay/reservation evidence')
    for file in ('fixtures.json.gz','summary.json'):
        require((out/'original'/file).read_bytes()==(out/'replay'/file).read_bytes(),'Replay bytes')
    require(file_hash(out/'original/fixtures.json.gz')==r['file_sha256'],'Fixture file hash')
    v=read(REVIEW/'verification.json');log=(REVIEW/'npm-test.log').read_text();sources=source_manifest()
    require(v['status']=='passed' and v['exit_code']==0 and v['sources_unchanged'] is True and v['source_files']==sources
            and v['test_log_sha256']==file_hash(REVIEW/'npm-test.log'),'Source-bound regression')
    counts=list(map(int,re.findall(r'Ran (\d+) tests? in',log)))
    require(counts==SUITES and node_pass_count(log)==75 and log.count('\nOK\n')==len(SUITES),'Regression count')
    for name in ('claude-review.md','boundary-review.md','review-dispositions.md','delegate-verification.md'):
        require((REVIEW/name).is_file(),'Review record '+name)
    for p in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:sources[str(p.relative_to(ROOT))]=file_hash(p)
    evidence={'artifacts/reports/m2-policy-pilot.json':file_hash(ROOT/'artifacts/reports/m2-policy-pilot.json')}
    for directory in [*sorted((ROOT/'artifacts/runs').glob('m3-switch-reference-*')),REVIEW]:
        for p in sorted(directory.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:evidence[str(p.relative_to(ROOT))]=file_hash(p)
    receipt={'format':'switch-reference-receipt-1','status':'complete-reference-scope','run':str(out.relative_to(ROOT)),
             'predecessor':a['predecessor'],'full_g2':'open','full_g3':'open','jax_port_implemented':False,
             'training_performed':False,'final_evaluation_used':False,'prime_used':False,
             'report':r,'tests':{'node':75,'python':sum(counts),'suite_counts':counts},'source_files':sources,'evidence_files':evidence}
    receipt['content_hash']=digest(receipt);archive=ROOT/'artifacts/manifests'/('m3-switch-reference-'+receipt['content_hash'])
    for kind,files in [('source',sources),('evidence',evidence)]:
        for path,h in files.items():
            require(file_hash(ROOT/path)==h,'Live file drift');write_once(archive/kind/path,(ROOT/path).read_bytes())
            require(file_hash(archive/kind/path)==h,'Archive drift')
    write_once(archive/'receipt.json',wire(receipt)+b'\n');write_once(ROOT/'artifacts/reports/m3-switch-reference.json',wire(receipt)+b'\n')
    print({'status':receipt['status'],'hash':receipt['content_hash'],'tests':75+sum(counts)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');main(p.parse_args().directory)
