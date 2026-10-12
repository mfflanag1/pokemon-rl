"""Admission of the new scope against immutable M2 evidence and reserved data."""
import json
import runpy
from pathlib import Path
from pokemon_learner.checkpoint import digest,file_hash
from pokemon_reference.dataset_gate import checked_file
from pokemon_soak.campaign import admission as soak_admission
from .contract import PROFILE,require
ROOT=Path(__file__).resolve().parents[2]
PREDECESSOR='daaa42457c61ba4ae79bbc9a2c7ba53d3ae1c1f376d3e7ab67d265e00803d7c1'
CURRENT={'README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md','docs/REVIEW_RECORD.md','package.json','scripts/test.mjs'}


def read(path):return json.loads(path.read_bytes())
def source_manifest():return runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()


def admission():
    prior=read(ROOT/'artifacts/reports/m2-policy-pilot.json')
    require(prior['content_hash']==PREDECESSOR==digest({k:v for k,v in prior.items() if k!='content_hash'}),'M2 receipt')
    require(PROFILE['predecessor']==PREDECESSOR,'Profile predecessor')
    archive=ROOT/'artifacts/manifests'/('m2-policy-pilot-'+PREDECESSOR)
    require(read(archive/'receipt.json')==prior,'M2 archive receipt')
    for kind,key in [('source','source_files'),('evidence','evidence_files')]:
        for path,h in prior[key].items():
            checked_file(archive/kind,path,h)
            if path not in CURRENT:checked_file(ROOT,path,h)
    runtime=soak_admission()
    require(runtime==read(ROOT/prior['run']/'preregistration.json')['runtime_and_subject'],'Inherited runtime/reservation binding')
    return {'format':'switch-reference-admission-1','profile':PROFILE,'predecessor':PREDECESSOR,
            'inherited_runtime':runtime,'source_files':source_manifest()}
