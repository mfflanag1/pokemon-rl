"""Fixed F11 audit; no optimizer, seed search, reserved data, or public server."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import time
from pokemon_full_train.storage import ROOT, atomic_json, canonical, digest, freeze, admission, load_checkpoint
from pokemon_full_train.learning import derived_seed
from .transport import Worker
from .core import (SIDES, Selector, fingerprint, feature_hash, start_spec, traced_command,
                   actor_seeds, continue_game, record_step, reset_scenario, public_context)
from .core import policy_command, cutoff_boundary


def predecessor(plan):
    path = ROOT/plan['predecessor_receipt']
    if digest(path) != plan['predecessor_receipt_sha256']:
        raise ValueError('F10 receipt changed')
    receipt = json.loads(path.read_text())
    for group in ['source_files', 'evidence_files']:
        for rel, sha in receipt[group].items():
            if digest(ROOT/rel) != sha:
                raise ValueError(f'F10 sealed file changed: {rel}')
    for rel,sha in receipt['source_files'].items():
        if digest(ROOT/receipt['source_snapshot_root']/rel) != sha:
            raise ValueError('F10 archived source drift')
    return receipt


def games_for(plan, seed, arm):
    path = ROOT/plan['predecessor_run']/str(seed)/arm/'final-evaluation.json'
    return json.loads(path.read_text())['games']['tactical']


def parameters_for(plan, seed, arm):
    return load_checkpoint(ROOT/plan['predecessor_run']/str(seed)/arm/'final', plan['predecessor_input_hash'])[0]


def audit_cell(run, plan, seed, arm):
    directory = run/'factual'/str(seed)/arm
    directory.mkdir(parents=True, exist_ok=False)
    selector = Selector(plan['families'], plan['branches_per_family_per_cell'])
    counts, tags, disagreement_tags, families, confusions, candidates = Counter(), Counter(), Counter(), Counter(), Counter(), []
    games = games_for(plan, seed, arm)
    if len(games) != plan['factual_games_per_cell']:
        raise ValueError('F10 cell count changed')
    with Worker(directory/'workers') as worker, open(directory/'decisions.jsonl', 'xb') as out:
        for ordinal, game in enumerate(games):
            if not game['trace']: raise ValueError('Empty factual trace')
            side = game['learner_side']
            frame = worker.request('start', **start_spec(game, plan['evaluation_seed']))
            if frame['provenance'] != game['provenance']:
                raise ValueError('F10 provenance mismatch')
            cursor, step_index, rejections = 0, 0, 0
            while cursor < len(game['trace']):
                command, new_cursor = traced_command(frame, game['trace'], cursor, game['learner_side'])
                before = frame
                frame = worker.request('step', **command)
                rejections += sum(not r['accepted'] for r in frame['submitted'])
                side = game['learner_side']
                if side in command['actions']:
                    d = frame['diagnostics'][side]
                    counts['learner_decisions'] += 1
                    counts['disagreements'] += int(d['disagreement'])
                    tags.update(d['tags'])
                    if d['disagreement']: disagreement_tags.update(d['tags'])
                    families[d['selected_family']] += 1
                    if d['disagreement']: confusions[d['selected_family']+' -> '+d['recommended_family']] += 1
                    chosen_tags = selector.select(ordinal, d)
                    context = public_context(before['players'][side])
                    record = {'game': ordinal, 'step': step_index, 'diagnostic': d, 'context':context}
                    out.write(canonical(record)+b'\n')
                    if chosen_tags:
                        candidates.append({'seed': seed, 'arm': arm, 'game': ordinal, 'step': step_index,
                            'families': chosen_tags, 'selected': command['actions'][side],
                            'alternative': before['tactical_actions'][side],
                            'target_frame_hash': fingerprint(before['players']),
                            'target_feature_hash': feature_hash(before['players'][side]), 'diagnostic': d,'context':context})
                cursor, step_index = new_cursor, step_index+1
            if game['outcome']['kind'] == 'administrative_cutoff':
                if not cutoff_boundary(frame, plan['max_turns']): raise ValueError('Factual cutoff boundary mismatch')
                frame = worker.request('cutoff')
            if frame['outcome'] != game['outcome'] or rejections != game['unavailable_choices']:
                raise ValueError('Factual terminal/feedback mismatch')
            counts['games'] += 1; counts['decisions'] += cursor; counts['unavailable_choices'] += rejections
            counts[frame['outcome']['kind']] += 1
            counts['wins'] += int(frame['outcome']['rewards'][side] == 1)
            if (ordinal+1) % 64 == 0:
                atomic_json(directory/'progress.json', {'games': ordinal+1, 'total': len(games)})
    result = {'seed':seed,'arm':arm,'counts':dict(counts),'tags':dict(tags),
              'disagreement_tags':dict(disagreement_tags),'selected_families':dict(families),
              'family_disagreements':dict(confusions),'selection_counts':dict(selector.counts),'candidates':candidates}
    atomic_json(directory/'report.json', result)
    return result


def branch(worker, parameters, game, candidate, plan):
    frame = worker.request('start', **start_spec(game, plan['evaluation_seed']))
    cursor, counts = 0, [0,0]
    for _ in range(candidate['step']):
        command, cursor = traced_command(frame, game['trace'], cursor, game['learner_side'])
        frame = worker.request('step', **command)
        for i,s in enumerate(SIDES): counts[i] += int(s in command['actions'])
    if fingerprint(frame['players']) != candidate['target_frame_hash']:
        raise ValueError('Branch target mismatch')
    command, _ = traced_command(frame, game['trace'], cursor, game['learner_side'])
    side = game['learner_side']
    if command['actions'][side] != candidate['selected'] or frame['tactical_actions'][side] != candidate['alternative']:
        raise ValueError('Branch intervention changed')
    seeds = actor_seeds(derived_seed('f4-evaluation',candidate['seed'],'tactical'), game['episode'], game['swap'])
    sampled, values = policy_command(parameters,frame,seeds,counts,side,plan['actor_capacity'])
    row = next(r for r in game['trace'][cursor:cursor+len(command['actions'])] if r[0]==side)
    if sampled != command or values[side] != row[3]:
        raise ValueError('Original learner action/value or RNG at branch target differs')
    command['actions'][side] = candidate['alternative']
    after = worker.request('step', **command)
    intervention = record_step(frame, command, {}, after)
    for i,s in enumerate(SIDES): counts[i] += int(s in command['actions'])
    steps, final = continue_game(worker, parameters, after, seeds, counts, side, plan)
    factual, alternate = game['outcome'], final['outcome']
    delta = alternate['rewards'][side]-factual['rewards'][side] if factual['kind']==alternate['kind']=='terminal' else None
    return {'candidate':candidate,'factual_outcome':factual,'alternative_outcome':alternate,
            'terminal_reward_delta':delta,'intervention':intervention,'continuation':steps}


def train_bank(run, plan, seed):
    directory = run/'training'/str(seed)
    directory.mkdir(parents=True, exist_ok=False)
    parameters = parameters_for(plan, seed, 'rehearsal')
    selector = Selector(plan['families'], plan['scenarios_per_family_per_seed'])
    run_seed = derived_seed(plan['training_seed_namespace'], seed)
    scenarios, counts = [], Counter()
    with Worker(directory/'workers') as worker:
        for episode in range(plan['training_games_per_seed']):
            side = SIDES[episode % 2]
            start = dict(run_seed=run_seed,episode_index=episode,split='train',swap=False)
            frame = worker.request('start', **start)
            provenance = frame['provenance']
            seeds = actor_seeds(run_seed, episode, False)
            steps, final = continue_game(worker,parameters,frame,seeds,[0,0],side,plan)
            # Selector has no outcome input and uses only pre-action diagnostics.
            # Selection is deterministic in decision order even though collection
            # completes the game first for efficient serialization.
            counters = [0,0]
            for index, step in enumerate(steps):
                d = step['diagnostics'].get(side)
                tags = selector.select(episode,d) if d else []
                if tags:
                    # Reconstruct target once to retain its public feature hash.
                    target = worker.request('start', **start)
                    for prior in steps[:index]: target = worker.request('step', **prior['command'])
                    if fingerprint(target['players']) != step['before']:
                        raise ValueError('Train target replay mismatch')
                    scenario = {'schema_version':'f11-train-scenario-v1','split':'train','seed':seed,
                        'start':start,'provenance':provenance,'learner_side':side,'actor_seeds':seeds,
                        'counters':list(counters),'families':tags,
                        'prefix':[{k:p[k] for k in ['before','command','submitted','after']} for p in steps[:index]],
                        'target_frame_hash':step['before'],'target_feature_hash':feature_hash(target['players'][side]),
                        'origin':{'episode':episode,'step':index,'policy':'F10 rehearsal final','role':'train-only'}}
                    scenario['content_hash'] = fingerprint(scenario)
                    scenarios.append(scenario)
                for i,s in enumerate(SIDES): counters[i] += int(s in step['command']['actions'])
            record = {'start':start,'provenance':provenance,'learner_side':side,'actor_seeds':seeds,
                      'steps':steps,'outcome':final['outcome']}
            atomic_json(directory/f'game-{episode:03d}.json',record)
            counts['games'] += 1; counts['decisions'] += sum(counters)
            counts[final['outcome']['kind']] += 1
            counts['wins'] += int(final['outcome']['rewards'][side] == 1)
        # Every selected start is reset and fully continued with the same policy
        # and counters, requiring the original suffix exactly. Prefix rows never
        # enter an update; there is no optimizer in this campaign.
        for scenario in scenarios:
            frame, counters = reset_scenario(worker, scenario)
            suffix, final = continue_game(worker,parameters,frame,scenario['actor_seeds'],counters,scenario['learner_side'],plan)
            source = json.loads((directory/f"game-{scenario['origin']['episode']:03d}.json").read_text())
            if suffix != source['steps'][scenario['origin']['step']:] or final['outcome'] != source['outcome']:
                raise ValueError('Curriculum reset failed exact full-battle continuation')
        atomic_json(directory/'scenarios.json',scenarios)
    result = {'seed':seed,'counts':dict(counts),'scenario_count':len(scenarios),
              'family_counts':dict(selector.counts),'exact_suffix_replays':len(scenarios)}
    atomic_json(directory/'report.json',result)
    return result


def run_campaign(run, plan):
    run.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    predecessor(plan)
    manifest = freeze(run,plan)
    with ThreadPoolExecutor(max_workers=plan['audit_workers']) as pool:
        jobs = [pool.submit(audit_cell,run,plan,seed,arm) for seed in plan['seeds'] for arm in plan['arms']]
        audit = [f.result() for f in jobs]
    atomic_json(run/'factual-report.json',audit)
    branch_reports = []
    for cell in audit:
        seed,arm=cell['seed'],cell['arm']
        games=games_for(plan,seed,arm); parameters=parameters_for(plan,seed,arm)
        with Worker(run/'branches'/str(seed)/arm/'workers') as worker:
            for index,candidate in enumerate(cell['candidates']):
                result=branch(worker,parameters,games[candidate['game']],candidate,plan)
                path=run/'branches'/str(seed)/arm/f'{index:03d}.json'
                atomic_json(path,result)
                branch_reports.append({k:result[k] for k in ['candidate','factual_outcome','alternative_outcome','terminal_reward_delta']})
    atomic_json(run/'branch-report.json',branch_reports)
    # Adaptive development curriculum weights; the evidence is explicitly one
    # realized branch, not an estimate of expected action regret. All families
    # retain positive coverage. This recipe is not executed as an RL experiment.
    improved=Counter(t for b in branch_reports if b['terminal_reward_delta'] is not None and b['terminal_reward_delta']>0
                     for t in b['candidate']['families'])
    weights={t:1+improved[t] for t in plan['families']}
    total=sum(weights.values())
    atomic_json(run/'curriculum-recipe.json',{
        'schema_version':'f11-curriculum-recipe-v1','role':'train-only','executed_updates':0,
        'family_weights':{t:w/total for t,w in weights.items()},'positive_conditional_branches':dict(improved),
        'mix_stages':[{'official_start_fraction':.9,'prefix_start_fraction':.1},
                      {'official_start_fraction':.8,'prefix_start_fraction':.2},
                      {'official_start_fraction':.75,'prefix_start_fraction':.25}],
        'sampling':'Choose family with these weights, then uniformly among its bank entries across all seeds; overlapping scenarios are eligible in each family. No future-outcome filtering.',
        'continuation':'Normal full battle after the unsolved handoff; prefix excluded from policy/value rows; true terminals and explicit cutoffs remain distinct.',
        'next_experiment':'Register equal PPO-row/optimizer budgets and all-seed ablation against official-start control before any updates; use fresh development games, keep final reserved.',
        'limitations':'Adaptive to F10 development findings; finite recurring train prefixes, not held-out families; branch-derived weights are a curriculum proposal, not established optimality.'})
    training=[train_bank(run,plan,seed) for seed in plan['seeds']]
    admission(run,manifest)
    predecessor(plan)
    report={'schema_version':'f11-battle-audit-report-v1','status':'passed','input_hash':fingerprint(manifest),
            'elapsed_seconds':time.perf_counter()-begun,'factual':audit,'branches':branch_reports,'training':training,
            'optimizer_updates':0,'final_reserved_data_used':False,'public_server_used':False,'prime_used':False,
            'interpretation':plan['interpretation']}
    atomic_json(run/'report.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads((ROOT/'contracts/battle-audit-v1/plan.json').read_text())
    result=run_campaign(args.run,plan)
    print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
