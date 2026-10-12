"""Run the complete regression and bind its log to the tested code/configuration.

Documentation and outputs are excluded; implementation, tests, contract fixtures,
dependency locks and this verification script are included. The campaign separately
checks dataset, predecessor and runtime provenance. A changed tree fails verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest():
    paths=[]
    for directory in ('python','packages','contracts','scripts','tests','tools/mechanics-deps'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and not {'node_modules','__pycache__'} & set(p.parts))
    paths.extend(ROOT/p for p in ('package.json','package-lock.json','showdown.pin.json','requirements-contract.txt',
                                  'requirements-learner.txt','requirements-jax.txt','.node-version','.gitignore'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def main(directory):
    out=Path(directory).resolve()
    if not out.is_relative_to(ROOT/'artifacts/reviews'):raise ValueError('Verification directory must be under artifacts/reviews')
    out.mkdir(parents=True,exist_ok=True)
    before=source_manifest();log=out/'npm-test.log'
    # Preserve an earlier run; a new source-bound verification never silently erases it.
    if log.exists():
        previous=out/('npm-test-before-'+sha(log)[:16]+'.log')
        if not previous.exists():previous.write_bytes(log.read_bytes())
    with log.open('wb') as stream:
        result=subprocess.run(['npm','test'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    after=source_manifest();passed=result.returncode==0 and before==after
    report={'format':'source-bound-regression-1','status':'passed' if passed else 'failed',
            'command':['npm','test'],'exit_code':result.returncode,'python':sys.version,
            'node':subprocess.check_output(['node','--version'],text=True).strip(),
            'source_files':before,'sources_unchanged':before==after,'test_log_sha256':sha(log)}
    if before!=after:report['changed_paths']=[p for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)]
    (out/'verification.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'verification':str(out/'verification.json'),'sources':len(before)}))
    if not passed:raise SystemExit(result.returncode or 1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');main(parser.parse_args().directory)
