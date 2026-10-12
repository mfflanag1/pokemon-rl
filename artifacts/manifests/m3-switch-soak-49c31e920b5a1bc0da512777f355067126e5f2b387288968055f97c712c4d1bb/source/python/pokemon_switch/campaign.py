"""Produce and independently validate replayable reference fixtures on local CPU."""
import argparse
import json
import subprocess
import traceback
from pathlib import Path
from .provenance import ROOT,admission
from .audit import read_bundle,validate
from .contract import require
from pokemon_learner.checkpoint import wire,digest,file_hash


def run(path):
    out=Path(path).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Artifact path')
    out.mkdir(exist_ok=False);report={'status':'failed'}
    try:
        start=admission();(out/'admission.json').write_bytes(wire(start)+b'\n')
        for name in ('original','replay'):
            with (out/(name+'.log')).open('xb') as log:
                subprocess.run(['node','packages/switch-reference/fixtures.mjs',str(out/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        for file in ('fixtures.json.gz','summary.json'):
            require((out/'original'/file).read_bytes()==(out/'replay'/file).read_bytes(),'Independent initialization replay differs')
        report=validate(read_bundle(out/'original/fixtures.json.gz'))
        report.update(admission_hash=digest(start),file_sha256=file_hash(out/'original/fixtures.json.gz'),
                      exact_regeneration=True,coverage=json.loads((out/'original/summary.json').read_bytes())['coverage'])
        require(admission()==start,'Source/runtime/reserved-data drift')
        report['final_reservation_unchanged']=True
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        report['content_hash']=digest(report);(out/'report.json').write_bytes(wire(report)+b'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');run(p.parse_args().directory)
