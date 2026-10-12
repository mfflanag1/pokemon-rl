"""Independently compare every recorded reference transition and damage outcome."""
import argparse
from collections import Counter
from pathlib import Path
from pokemon_reference.corpus import read_json, verify_seal, sealed, file_digest
from pokemon_reference.contract import canonical,hash_json
from pokemon_reference.dataset_gate import checked_file
from .engine import ROOT,PROFILE,initial,step,damage,require


def validate_plan(plan):
    require(plan['version']=='m1-slice-campaign-1' and plan['profile']==PROFILE,'Profile/plan version')
    reference=plan['reference'];pin=read_json(ROOT/'showdown.pin.json')
    require(reference['commit']==PROFILE['showdown_commit']==pin['commit'] and reference['source_dirty'] is False,'Showdown pin mismatch')
    require(reference['manifest_id']==hash_json({k:v for k,v in reference.items() if k!='manifest_id'}),'Reference manifest hash')
    for name,expected in plan['sources'].items():checked_file(ROOT,name,expected)
    for name,expected in plan['upstream'].items():checked_file(ROOT/'vendor/pokemon-showdown',name,expected)
    for name,expected in reference['source_hashes'].items():checked_file(ROOT/'vendor/pokemon-showdown',name,expected)
    for name,expected in reference['harness_source_hashes'].items():checked_file(ROOT/'packages/reference-node',name,expected)
    require(reference['build']==read_json(ROOT/'vendor/pokemon-showdown/.reference-build.json'),'Reference build mismatch')
    require(type(plan['count']) is int and plan['count']>0,'Invalid campaign count')


def verify(directory):
    directory=Path(directory);plan=read_json(directory/'plan.json');reference=verify_seal(read_json(directory/'reference.json'))
    require(reference['plan_hash']==hash_json(plan) and plan['profile']==PROFILE,'Profile/plan linkage')
    validate_plan(plan)
    require([c['id'] for c in reference['cases']]==list(range(plan['count'])),'Case completeness')
    require(len({c['seed'] for c in reference['cases']})==plan['count'],'Repeated battle seed')
    expected={(crit,roll) for crit in (False,True) for roll in range(85,101)}
    require({(r['crit'],r['roll']) for r in reference['damage_table']}==expected and len(reference['damage_table'])==32,'Damage enumeration completeness')
    for row in reference['damage_table']:require(damage(row['crit'],row['roll'])==row['damage'],'Enumerated damage mismatch')
    phases=Counter();winners=Counter();events=turns=0
    for case in reference['cases']:
        require(case['seed']=='sodium,'+hash_json(['pokemon-reference-r1',f"m1-slice-reference-v1/{case['id']}"]),'Seed plan mismatch')
        state=initial(PROFILE);require(state.snapshot()==case['frames'][0],'Initial snapshot mismatch')
        require(len(case['frames'])==len(case['turns'])+1,'Frame count')
        for frame,tape in zip(case['frames'][1:],case['turns']):
            require(tape['turn']==state.turn,'Turn mismatch')
            action_hash=hash_json(['m1-slice-actions-1',case['id'],state.turn])
            require(tape['actions']==[int(action_hash[i],16)%2 for i in (0,1)],'Action plan mismatch')
            state=step(state,tape['actions'],tape['events'],tape['tie_first'])
            require(state.snapshot()==frame,f"Transition/request mismatch: case {case['id']}, turn {tape['turn']}")
            events+=len(tape['events']);turns+=1;phases[frame['phase']]+=1
        require(state.winner in (0,1),'Nonterminal case');winners[str(state.winner)]+=1
    report=sealed({'version':'m1-slice-parity-1','profile':PROFILE['id'],'reference_hash':reference['content_hash'],
        'plan_hash':reference['plan_hash'],'battles':len(reference['cases']),'turns':turns,'damage_events':events,
        'damage_enumeration':32,'disagreements':0,'phases':dict(phases),'winners':dict(winners),
        'source_hashes':{str(p.relative_to(ROOT)):file_digest(p) for p in sorted((ROOT/'python/pokemon_slice').glob('*.py'))},
        'scope':'Conditional transition and normalized own-request parity. Oracle-sampled semantic chance. No JAX, independent RNG-distribution, public-history or full G2 certification.'})
    output=directory/'parity.json'
    with output.open('x') as f:f.write(canonical(report)+'\n')
    print(canonical(report));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');verify(p.parse_args().directory)
