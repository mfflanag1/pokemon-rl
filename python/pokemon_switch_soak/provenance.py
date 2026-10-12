"""Admit immutable predecessor evidence without rerunning its parity campaign."""
import json
import runpy
from pathlib import Path
from pokemon_learner.checkpoint import digest,file_hash
from pokemon_switch.provenance import CURRENT
from pokemon_switch_engine import checkpoint
from pokemon_switch_engine.scalar import require

ROOT=Path(__file__).resolve().parents[2]
PLAN=json.loads((ROOT/'contracts/switch-soak-v1/plan.json').read_text())
PREDECESSOR='af89e56e72553901b4582a3b6910d3ee4ad87c62450a3025db99e5019898774e'


def read(path):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs)


def admission():
    require(PLAN['predecessor']==PREDECESSOR,'Qualification predecessor')
    path=ROOT/'artifacts/reports/m3-switch-engine.json';prior=read(path)
    require(prior['content_hash']==PREDECESSOR==digest({k:v for k,v in prior.items() if k!='content_hash'}),'Engine receipt')
    archive=ROOT/'artifacts/manifests'/('m3-switch-engine-'+PREDECESSOR)
    require((archive/'receipt.json').read_bytes()==path.read_bytes(),'Engine archive receipt')
    checks=0
    for kind,field in [('source','source_files'),('evidence','evidence_files')]:
        for name,h in prior[field].items():
            require(file_hash(archive/kind/name)==h,'Archived predecessor drift: '+name);checks+=1
            if name not in CURRENT:
                require(file_hash(ROOT/name)==h,'Live predecessor drift: '+name);checks+=1
    inherited=runpy.run_path(str(ROOT/'scripts/seal-switch-engine.py'))['admission']()
    return {'format':'switch-soak-admission-1','plan':PLAN,'predecessor':PREDECESSOR,
        'predecessor_hash_checks':checks,'inherited':inherited,'runtime':checkpoint.runtime(),'source_files':checkpoint.sources()}


def binding(admitted,mode):
    require(mode in ('smoke','qualification'),'Run mode')
    return {'admission_hash':digest(admitted),'plan':PLAN,'mode':mode,
        'sources':admitted['source_files'],'runtime':admitted['runtime']}
