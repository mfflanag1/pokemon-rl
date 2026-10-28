"""Independent replay pass before sealing F11; never changes production output."""
from collections import Counter
import argparse
import json
from itertools import product
from pathlib import Path
from pokemon_full_train.storage import ROOT, admission, atomic_json, digest
from .campaign import predecessor, games_for, parameters_for, branch
from .transport import Worker
from .core import Selector, fingerprint, continue_game, reset_scenario


def verify(run, evidence):
    manifest=json.loads((run/'inputs/manifest.json').read_text()); plan=manifest['plan']
    admission(run,manifest); predecessor(plan)
    report=json.loads((run/'report.json').read_text())
    if report['status']!='passed' or report['input_hash']!=fingerprint(manifest): raise ValueError('Bad production report')
    if report['optimizer_updates']!=0 or any(report[k] for k in ['final_reserved_data_used','public_server_used','prime_used']):
        raise ValueError('Scope violation')
    checked=Counter()
    if [(c['seed'],c['arm']) for c in report['factual']]!=list(product(plan['seeds'],plan['arms'])):
        raise ValueError('Wrong factual cell set/order')
    if [c['seed'] for c in report['training']]!=plan['seeds']: raise ValueError('Wrong train seed set/order')
    if report['factual']!=json.loads((run/'factual-report.json').read_text()) or report['branches']!=json.loads((run/'branch-report.json').read_text()):
        raise ValueError('Summary report drift')
    branch_summaries=[]
    for cell in report['factual']:
        seed,arm=cell['seed'],cell['arm']
        if cell!=json.loads((run/'factual'/str(seed)/arm/'report.json').read_text()): raise ValueError('Cell report drift')
        if cell['counts']['games']!=512: raise ValueError('Incomplete factual cell')
        tags,disagreement_tags,families,confusions=Counter(),Counter(),Counter(),Counter()
        selector=Selector(plan['families'],plan['branches_per_family_per_cell']); selected=[]
        n=disagreements=0
        with open(run/'factual'/str(seed)/arm/'decisions.jsonl') as stream:
            for line in stream:
                row=json.loads(line); d=row['diagnostic']; n+=1
                disagreements+=int(d['disagreement']); tags.update(d['tags']); families[d['selected_family']]+=1
                if d['disagreement']:
                    disagreement_tags.update(d['tags']); confusions[d['selected_family']+' -> '+d['recommended_family']]+=1
                selected_tags=selector.select(row['game'],d)
                if selected_tags: selected.append((row['game'],row['step'],selected_tags,d,row['context']))
        if (dict(tags)!=cell['tags'] or dict(disagreement_tags)!=cell['disagreement_tags'] or
            dict(families)!=cell['selected_families'] or dict(confusions)!=cell['family_disagreements'] or
            n!=cell['counts']['learner_decisions'] or disagreements!=cell['counts']['disagreements'] or
            selected!=[(c['game'],c['step'],c['families'],c['diagnostic'],c['context']) for c in cell['candidates']]):
            raise ValueError('Factual aggregate or fixed selection mismatch')
        checked['factual_records_reaggregated']+=n; checked['factual_games']+=512
        parameters=parameters_for(plan,seed,arm); games=games_for(plan,seed,arm)
        with Worker(evidence/'branches'/str(seed)/arm) as worker:
            for i,candidate in enumerate(cell['candidates']):
                original=json.loads((run/'branches'/str(seed)/arm/f'{i:03d}.json').read_text())
                fresh=branch(worker,parameters,games[candidate['game']],candidate,plan)
                if original!=fresh: raise ValueError('Fresh branch replay differs')
                branch_summaries.append({k:original[k] for k in ['candidate','factual_outcome','alternative_outcome','terminal_reward_delta']})
                checked['branches_replayed']+=1
    if branch_summaries!=report['branches']: raise ValueError('Branch summary drift')
    for seed in plan['seeds']:
        parameters=parameters_for(plan,seed,'rehearsal')
        with Worker(evidence/'training'/str(seed)) as worker:
            for episode in range(plan['training_games_per_seed']):
                original=json.loads((run/'training'/str(seed)/f'game-{episode:03d}.json').read_text())
                frame=worker.request('start',**original['start'])
                if frame['provenance']!=original['provenance']: raise ValueError('Train provenance mismatch')
                steps,end=continue_game(worker,parameters,frame,original['actor_seeds'],[0,0],original['learner_side'],plan)
                if steps!=original['steps'] or end['outcome']!=original['outcome']: raise ValueError('Train full-game replay differs')
                checked['training_games_replayed']+=1
            scenarios=json.loads((run/'training'/str(seed)/'scenarios.json').read_text())
            selector=Selector(plan['families'],plan['scenarios_per_family_per_seed']); expected=[]
            for episode in range(plan['training_games_per_seed']):
                original=json.loads((run/'training'/str(seed)/f'game-{episode:03d}.json').read_text())
                for index,step in enumerate(original['steps']):
                    d=step['diagnostics'].get(original['learner_side'])
                    tags=selector.select(episode,d) if d else []
                    if tags: expected.append((episode,index,tags))
            if expected!=[(s['origin']['episode'],s['origin']['step'],s['families']) for s in scenarios]:
                raise ValueError('Train-only deterministic selection mismatch')
            for scenario in scenarios:
                reset_scenario(worker,scenario)
                original=json.loads((run/'training'/str(seed)/f"game-{scenario['origin']['episode']:03d}.json").read_text())
                prefix=[{k:s[k] for k in ['before','command','submitted','after']} for s in original['steps'][:scenario['origin']['step']]]
                if scenario['prefix']!=prefix: raise ValueError('Prefix not from declared train source')
                checked['scenarios_reset']+=1
    if checked['factual_games']!=3072 or checked['training_games_replayed']!=96: raise ValueError('Missing cells')
    if report['training']!=[json.loads((run/'training'/str(s)/'report.json').read_text()) for s in plan['seeds']]:
        raise ValueError('Training summary drift')
    admission(run,manifest); predecessor(plan)
    result={'status':'passed','report_sha256':digest(run/'report.json'),'input_hash':fingerprint(manifest),'checks':dict(checked),
            'scope':'Factual records reaggregated and deterministic selection rechecked; production factual replay inherited. Every single-action branch and fresh train full game replayed independently, every bank prefix reset and bound to its declared train source. No optimizer updates.'}
    atomic_json(evidence/'verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(verify(a.run,a.evidence)),flush=True)
