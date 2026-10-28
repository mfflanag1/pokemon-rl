import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_full.features import encode
from pokemon_full.policy import initialize, forward, distribution
from pokemon_full_train.learning import optimize, derived_seed
from pokemon_full_train.storage import ROOT, atomic_json, freeze, admission, digest, canonical, save_checkpoint
from .collect import Pool, target_action, specs_for, evaluation_specs


def summarize(parameters, records):
    groups = {}
    for panel in ['train', 'variant']:
        for family, contexts in [('switch', range(8)), ('tera', range(8,16))]:
            selected = [r for r in records if r['spec']['panel'] == panel and r['spec']['context'] in contexts]
            xs, masks = zip(*(encode(r['player']) for r in selected))
            logits, value = forward(parameters, jnp.asarray(np.stack(xs)))
            prob, _ = distribution(logits, jnp.asarray(np.stack(masks)))
            prob = np.asarray(prob)
            targets = [target_action(r['player'], r['spec']['context']) for r in selected]
            cells = {}
            for c in contexts:
                ids = [i for i, r in enumerate(selected) if r['spec']['context'] == c]
                cells[str(c)] = {'optimal_mass': float(np.mean([prob[i, targets[i]] for i in ids])),
                    'sampled_correct': sum(selected[i]['action'] == targets[i] for i in ids), 'n': len(ids)}
            groups[f'{panel}/{family}'] = {'n': len(selected), 'optimal_mass': float(np.mean(prob[np.arange(len(selected)),targets])),
                'sampled_correct': sum(r['action'] == t for r,t in zip(selected, targets)),
                'mean_reward': float(np.mean([r['reward'] for r in selected])),
                'local_value_mse': float(np.mean([(r['value']-r['reward'])**2 for r in selected])), 'cells': cells}
    return groups


def learning_gate(initial, final, plan):
    cells = {k: {'gain': final[k]['optimal_mass'] - initial[k]['optimal_mass'],
        'minimum_gain': final[k]['optimal_mass'] - initial[k]['optimal_mass'] >= plan['minimum_probability_gain'],
        'minimum_final': final[k]['optimal_mass'] >= plan['minimum_final_optimal_mass'],
        'every_context_minimum': all(v['optimal_mass'] >= plan['minimum_final_per_context'] for v in final[k]['cells'].values())}
        for k in initial}
    return {'passed': all(v['minimum_gain'] and v['minimum_final'] and v['every_context_minimum'] for v in cells.values()), 'cells': cells}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); args = ap.parse_args()
    run = Path(args.run); run.mkdir(parents=True, exist_ok=False)
    plan = json.loads((ROOT/'contracts/decision-canary-v1/plan.json').read_text())
    manifest = freeze(run, plan); input_hash = digest(run/'inputs/manifest.json')
    started = time.perf_counter(); report = {'status':'running','input_hash':input_hash,'plan':plan,'seeds':{}}
    try:
        with Pool(run/'workers', plan['workers']) as pool:
            for seed in plan['seeds']:
                p = initialize(seed); z = jax.tree.map(jnp.zeros_like, p); opt = (z,z,jnp.int32(0))
                base = run/str(seed); base.mkdir()
                meta = {'seed':seed,'input_hash':input_hash,'cohorts':0}
                initial_ck = save_checkpoint(base/'initial',p,opt,meta)
                eval_specs = evaluation_specs(plan['evaluation_seed'],plan['evaluation_repeats'])
                _, initial_records = pool.collect(p,eval_specs,derived_seed('f6-eval',seed))
                atomic_json(base/'initial-evaluation.json',initial_records)
                initial = summarize(p,initial_records)
                for step in range(plan['updates_per_seed']):
                    t0 = time.perf_counter()
                    if step == plan['updates_per_seed']-1:
                        save_checkpoint(base/'before-last',p,opt,{**meta,'cohorts':step})
                    rows, records = pool.collect(p,specs_for(seed,step,plan['rows_per_update']),derived_seed('f6-train-actor',seed))
                    p,opt,metrics = optimize(p,opt,rows,derived_seed('f6-update',seed,step))
                    atomic_json(base/f'update-{step:03d}.json',{'records':records,'metrics':metrics,
                        'seconds':time.perf_counter()-t0,'optimizer_steps':int(opt[2])})
                    atomic_json(run/'progress.json',{'seed':seed,'updates':step+1,'elapsed':time.perf_counter()-started})
                final_ck = save_checkpoint(base/'final',p,opt,{**meta,'cohorts':plan['updates_per_seed']})
                _, final_records = pool.collect(p,eval_specs,derived_seed('f6-eval',seed))
                atomic_json(base/'final-evaluation.json',final_records)
                final = summarize(p,final_records)
                _, tails = pool.collect(p,evaluation_specs(derived_seed('f6-continuation',seed),1),
                    derived_seed('f6-continuation-actor',seed),continuation=True)
                atomic_json(base/'continuations.json',tails)
                report['seeds'][str(seed)] = {'initial':initial,'final':final,'gate':learning_gate(initial,final,plan),
                    'initial_checkpoint':initial_ck,'final_checkpoint':final_ck,'optimizer_steps':int(opt[2]),
                    'continuation_outcomes':dict(Counter(r['continuation']['frame']['outcome']['kind'] for r in tails))}
                admission(run,manifest)
        report.update(status='completed',seconds=time.perf_counter()-started,
            training_decisions=len(plan['seeds'])*plan['updates_per_seed']*plan['rows_per_update'],
            evaluation_decisions=len(plan['seeds'])*2*2*16*plan['evaluation_repeats'],
            continuation_games=len(plan['seeds'])*32,
            parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(run/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['seeds','plan']}),flush=True)
    except BaseException as e:
        atomic_json(run/'failure.json',{'error':repr(e),'elapsed':time.perf_counter()-started}); raise


if __name__ == '__main__': main()
