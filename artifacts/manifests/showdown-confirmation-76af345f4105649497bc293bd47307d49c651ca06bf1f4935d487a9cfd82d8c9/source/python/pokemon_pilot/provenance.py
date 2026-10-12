"""Admission binds the completed soak, its immutable archive, and the new pilot tree."""
import json
import runpy
import platform
from pathlib import Path
from pokemon_soak.campaign import admission as soak_admission,read
from pokemon_showdown.campaign import sealed
from pokemon_learner.checkpoint import file_hash,digest
from pokemon_learner.core import require
from pokemon_reference.dataset_gate import checked_file

ROOT=Path(__file__).resolve().parents[2]
PREDECESSOR='90fd64d35aa7cee6ce6cb6ebb1e33fb0c7817ea8355fd1f336c36d4770cced88'
CURRENT={'README.md','docs/IMPLEMENTATION_PLAN.md','docs/ACCEPTANCE_GATES.md','docs/REVIEW_RECORD.md','package.json','scripts/test.mjs'}


def source_manifest():return runpy.run_path(str(ROOT/'scripts/verify-fixed-loop.py'))['source_manifest']()


def admission():
    plan=read(ROOT/'contracts/pilot-v1/plan.json');require(plan['predecessor']==PREDECESSOR,'Pilot predecessor')
    prior=sealed(read(ROOT/'artifacts/reports/m1-policy-soak.json'))
    require(prior['content_hash']==PREDECESSOR and prior['two_hour_qualification'] is True,'Completed soak')
    archive=ROOT/'artifacts/manifests'/('m1-policy-soak-'+PREDECESSOR)
    require(read(archive/'receipt.json')==prior,'Predecessor archive receipt')
    for kind,key in [('source','source_files'),('evidence','evidence_files')]:
        for path,h in prior[key].items():
            checked_file(archive/kind,path,h)
            if path not in CURRENT:checked_file(ROOT,path,h)
    previous=soak_admission()
    require(previous==read(ROOT/prior['run']/'admission.json'),'Selected contract/runtime drift')
    require(plan['training_seeds']==[101,211,307] and plan['checkpoint_updates']==[0,256,2048]
            and plan['opponents']==['random','damage-greedy','frozen-initial']
            and plan['pairs_per_cell']==1000 and plan['evaluation_batch_games']==40
            and plan['action_key_seed']==1296646193 and plan['bootstrap_replicates']==4096
            and plan['statistics_seed']==8191 and plan['family_alpha']==.05
            and plan['primary_family_size']==6,'Frozen pilot roster/statistics budget')
    return {'plan':plan,'predecessor':PREDECESSOR,'runtime_and_subject':previous,'source_files':source_manifest(),
            'replay_host':{'hostname_architecture_sha256':digest([platform.node(),platform.machine()]),
                           'scope':'Host-name/architecture label plus inherited runtime binding and exact FP32 replay; not hardware attestation or a portability guarantee.'}}


def mechanics_seed(plan,seed,opponent,pair,mode):
    namespace=f"{plan['seed_namespace']}/{mode}/{seed}/{opponent}/{pair}"
    return 'sodium,'+digest(['pokemon-reference-r1',namespace])


def roster(plan,smoke=False):
    return (plan['training_seeds'][:1] if smoke else plan['training_seeds'],
            [0,2] if smoke else plan['checkpoint_updates'],4 if smoke else plan['pairs_per_cell'])
