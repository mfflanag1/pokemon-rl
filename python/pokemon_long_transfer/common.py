"""F10 adds receipt admission and diagnostics; collection/learning stay sealed."""
import hashlib
from pokemon_transfer_v2.common import (read,initialize,collect_update,evaluation,probe_specs,tactical,verified_inventory)
from pokemon_transfer_v2.common import predecessor as f9_inputs
from pokemon_transfer.common import summarize as f9_summarize
from pokemon_full_train.storage import ROOT,canonical,digest
from pokemon_fit.data import require
from .value import summarize_value


def predecessor(plan):
    batch,inventory=f9_inputs(plan)
    path=ROOT/plan['transfer_receipt'];r=read(path);body=dict(r);h=body.pop('content_hash')
    require(h==plan['transfer_content_hash']==hashlib.sha256(canonical(body)).hexdigest(),'F9 completed receipt binding')
    require(path.read_bytes()==(ROOT/'artifacts/manifests'/f'f9-full-transfer-v2-{h}'/'receipt.json').read_bytes(),'F9 completed receipt archive')
    require(r['status']=='completed','F9 did not complete')
    for rel,sha in r['source_files'].items():require(digest(ROOT/rel)==sha and digest(ROOT/r['source_snapshot_root']/rel)==sha,f'F9 source drift: {rel}')
    for rel,sha in r['evidence_files'].items():require(digest(ROOT/rel)==sha,f'F9 evidence drift: {rel}')
    return batch,inventory


def validate_plan(plan):
    require(plan==read(ROOT/'contracts/long-transfer-v1/plan.json'),'Plan differs from fixed registration')
    require(plan['seeds']==[7103,8209,9311] and plan['arms']==['plain','rehearsal'],'Wrong seeds/arms')
    require(plan['updates_per_arm']==128 and plan['selected_rows_per_update']==1024 and plan['minibatch_rows']==256 and plan['epochs']==2,'Changed fixed PPO budget')
    require(plan['workers']==8 and plan['training_chunk_games']==8 and plan['maximum_games_per_update']==64,'Changed collection layout')
    require(plan['coefficients']=={'plain':0.,'rehearsal':.1},'Changed rehearsal')
    require(plan['evaluation_pairs_per_opponent']==256 and plan['evaluation_chunk_pairs']==4 and plan['checkpoint_every_updates']==16,'Changed panel/checkpoints')
    require(plan['opponents']==['random','base_power','tactical'],'Wrong opponent panel')
    require(plan['statistics']['family_cells']==27 and plan['statistics']['resamples']==20000,'Changed inference family')
    require(all(plan[k] is False for k in ['final_reserved_data_used','public_server_used','prime_used']),'Unsupported scope')
    expected={'selected_ppo_rows':786432,'ppo_row_presentations':1572864,'rehearsal_row_presentations':393216,'optimizer_steps':6144,
        'evaluation_games':13824,'tactical_predictions':4608,'mechanics_probe_branches':3584}
    require(plan['expected_counts']==expected,'Wrong expected counts')


def summarize(initial,finals,initial_probe,final_probes,plan):
    result=f9_summarize(initial,finals,initial_probe,final_probes,plan)
    result['value_diagnostics']={endpoint:{opponent:summarize_value(games) for opponent,games in panel.items()}
        for endpoint,panel in {'initial':initial,**finals}.items()}
    return result
