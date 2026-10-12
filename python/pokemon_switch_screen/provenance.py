"""Immutable predecessor admission plus the new preregistered experiment tree."""
from pathlib import Path
from pokemon_learner.checkpoint import digest,file_hash
from pokemon_switch.provenance import CURRENT,source_manifest
from pokemon_switch_soak.provenance import admission as prior_admission,read
from pokemon_switch_engine.checkpoint import runtime
from pokemon_switch_engine.scalar import require,ROSTER_IDS

ROOT=Path(__file__).resolve().parents[2]
PREDECESSOR='49c31e920b5a1bc0da512777f355067126e5f2b387288968055f97c712c4d1bb'
PLAN=read(ROOT/'contracts/switch-screen-v1/plan.json')


def admission():
    require(PLAN['predecessor']==PREDECESSOR,'Screen predecessor')
    p=ROOT/'artifacts/reports/m3-switch-soak.json';r=read(p)
    require(r['content_hash']==PREDECESSOR==digest({k:v for k,v in r.items() if k!='content_hash'})
            and r['two_hour_qualification'] is True,'Qualified predecessor')
    archive=ROOT/'artifacts/manifests'/('m3-switch-soak-'+PREDECESSOR)
    require((archive/'receipt.json').read_bytes()==p.read_bytes(),'Predecessor archive receipt')
    checks=0
    for kind,field in [('source','source_files'),('evidence','evidence_files')]:
        for name,h in r[field].items():
            require(file_hash(archive/kind/name)==h,'Archived predecessor drift: '+name);checks+=1
            if name not in CURRENT:require(file_hash(ROOT/name)==h,'Live predecessor drift: '+name);checks+=1
    inherited=prior_admission();old=read(ROOT/r['run']/'admission.json')
    # Only additive current sources/registry can differ; all old files checked above.
    for section in (inherited,old):
        section.pop('source_files')
        section['inherited'].pop('source_files')
    require(inherited==old,'Inherited runtime/reservation or qualification drift')
    return {'format':'switch-screen-admission-1','plan':PLAN,'predecessor':PREDECESSOR,
            'predecessor_hash_checks':checks,'inherited':inherited,'runtime':runtime(),'source_files':source_manifest()}


def roster(smoke):
    return (PLAN['training_seeds'][:1],[0,2],4) if smoke else (PLAN['training_seeds'],PLAN['checkpoint_updates'],PLAN['pairs_per_cell'])


def spec(seed,opponent,pair,seat,mode):
    require(type(pair) is int and 0<=pair<576 and seat in (0,1),'Pair/seat')
    teams=[ROSTER_IDS[pair//24],ROSTER_IDS[pair%24]]
    if seat:teams.reverse()
    return {'seed':'sodium,'+digest(['m3-switch-screen-v1',mode,seed,opponent,pair]),'teams':teams}
