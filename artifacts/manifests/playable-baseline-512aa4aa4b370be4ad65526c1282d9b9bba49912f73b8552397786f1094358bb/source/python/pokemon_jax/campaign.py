"""Reproducible restricted JAX transition/view/chance campaign; fresh output only."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_reference.corpus import read_json, verify_seal
from pokemon_reference.contract import hash_json
from pokemon_reference.dataset_gate import checked_file
from pokemon_slice.engine import ROOT, PROFILE, require, MOVES
from pokemon_slice.verify import validate_plan
from pokemon_learner.checkpoint import digest, wire, file_hash
from .engine import initial, step, step_kernel, damage, Chance, State, view_kernel, actor_probe, PlayerView
from .chance import sample, sample_kernel
from .statistics import law_tests
from .views import reduce_view


def provenance():
    files=[*sorted((ROOT/'python/pokemon_jax').glob('*.py')), *sorted((ROOT/'contracts/jax-slice-v1').glob('*')),
           ROOT/'packages/jax-slice/reference.mjs',ROOT/'requirements-jax.txt',
           ROOT/'contracts/slice-v1/profile.json',ROOT/'requirements-learner.txt',ROOT/'package-lock.json',ROOT/'scripts/seal-jax.py']
    for package in ('pokemon_reference','pokemon_learner','pokemon_slice'):
        files.extend(sorted((ROOT/'python'/package).glob('*.py')))
    files.extend(sorted((ROOT/'packages/reference-node').glob('*.mjs')))
    files.append(ROOT/'packages/slice/reference.mjs')
    return {str(p.relative_to(ROOT)):file_hash(p) for p in files}


def load_references():
    old_dir=ROOT/'artifacts/runs/m1-slice-v1';new_dir=ROOT/'artifacts/runs/m1-g2-player-v1'
    old=verify_seal(read_json(old_dir/'reference.json'));old_plan=read_json(old_dir/'plan.json');validate_plan(old_plan)
    require(old['plan_hash']==hash_json(old_plan) and old_plan['count']==len(old['cases']),'Old plan link')
    g1=json.loads((ROOT/'artifacts/reports/m1-g1.json').read_text())
    require(g1['content_hash']==digest({k:v for k,v in g1.items() if k!='content_hash'}),'G1 receipt seal')
    require(old['content_hash']==g1['slice']['reference_hash'],'G1 reference receipt drift')
    new=verify_seal(read_json(new_dir/'reference.json'));plan=read_json(new_dir/'plan.json')
    require(new['version']=='m1-g2-player-reference-1' and plan['version']=='m1-g2-player-campaign-1','Player reference versions')
    require(new['plan_hash']==hash_json(plan) and plan['profile']==PROFILE and plan['exact_opponent_hp'] is True and plan['count']==len(new['cases']),'Player plan link')
    require(plan['reference']==old_plan['reference'],'Pinned oracle drift')
    for name,h in plan['sources'].items():checked_file(ROOT,name,h)
    for rows,namespace,actions in [(old['cases'],'m1-slice-reference-v1','m1-slice-actions-1'),(new['cases'],'m1-g2-player-reference-v1','m1-g2-actions-1')]:
        require([c['id'] for c in rows]==list(range(len(rows))),'Case completeness')
        for case in rows:
            require(case['seed']=='sodium,'+hash_json(['pokemon-reference-r1',f"{namespace}/{case['id']}"]),'Battle seed drift')
            require(2<=len(case['turns'])<=4,'Turn count')
            for turn,tape in enumerate(case['turns'],1):
                h=hash_json([actions,case['id'],turn])
                require(tape['turn']==turn and tape['actions']==[int(h[i],16)%2 for i in (0,1)],'Action tape drift')
    return old,new


def compare_views(state, wires):
    count=0
    for side in (0,1):
        actual=view_kernel(state,side)
        expected=[reduce_view(pair[side]) for pair in wires]
        header=np.array([v['header'] for v in expected],np.int32);history=np.array([v['history'] for v in expected],np.int32)
        require(np.array_equal(actual.header,header) and np.array_equal(actual.history,history),f'Player view mismatch seat {side}')
        # Actor is handed only this normalized semantic view, including its history.
        independent=PlayerView(jnp.asarray(header),jnp.asarray(history))
        for a,b in zip(actor_probe(actual),actor_probe(independent)):require(np.array_equal(a,b),'Actor view/logit mismatch')
        count+=len(expected)
    return count


def parity(cases,with_views=False):
    n=len(cases);state=initial(n);turns=events=views=pending=0
    if with_views:views+=compare_views(state,[c['initial'] for c in cases])
    else:
        for c in cases:require(c['frames'][0]=={'turn':1,'phase':'move','winner':None,'hp':[105,105],'pp':[[56,48],[56,48]],'requests':[{'own_hp':105,'moves':[{'id':m,'pp':p,'maxpp':p,'disabled':False} for m,p in zip(MOVES,[56,48])]}]*2},'Initial reference frame')
    for t in range(4):
        acts=np.zeros((n,2),np.int32);crit=np.zeros((n,2),bool);roll=np.full((n,2),85,np.int32);tie=np.zeros(n,np.int32)
        expected_events=np.full((n,2,5),-1,np.int32)
        for i,c in enumerate(cases):
            if t>=len(c['turns']):continue
            tape=c['turns'][t];acts[i]=tape['actions'];tie[i]=tape['tie_first'] or 0
            require((tape['tie_first'] in (0,1)) if acts[i,0]==acts[i,1] else tape['tie_first'] is None,'Tie semantic scope')
            for k,e in enumerate(tape['events']):
                a=e['actor'];crit[i,a]=e['crit'];roll[i,a]=e['roll'];expected_events[i,k]=[a,MOVES.index(e['move']),int(e['crit']),e['roll'],e['damage']]
            events+=len(tape['events']);turns+=1
        if with_views:
            before=[c['turns'][t]['pending'] if t<len(c['turns']) else c['turns'][-1]['views'] for c in cases]
            views+=compare_views(state,before);pending+=2*sum(t<len(c['turns']) for c in cases)
        state,actual_events=step(state,acts,Chance(jnp.asarray(crit),jnp.asarray(roll),jnp.asarray(tie)))
        require(np.array_equal(actual_events,expected_events),f'Semantic event mismatch turn {t+1}')
        if with_views:
            views+=compare_views(state,[c['turns'][min(t,len(c['turns'])-1)]['views'] for c in cases])
        else:
            host=State(*map(np.asarray,state))
            for i,c in enumerate(cases):
                f=c['frames'][min(t+1,len(c['frames'])-1)];win=int(host.winner[i]);win=None if win==-1 else win
                require(int(host.turn[i])==f['turn'] and host.hp[i].tolist()==f['hp'] and host.pp[i].tolist()==f['pp'] and win==f['winner'],'Transition mismatch')
                require(f['phase']==('move' if win is None else 'terminal'),'Request phase')
                if win is None:
                    expected=[{'own_hp':f['hp'][a],'moves':[{'id':m,'pp':f['pp'][a][j],'maxpp':[56,48][j],'disabled':False} for j,m in enumerate(MOVES)]} for a in (0,1)]
                    require(f['requests']==expected,'Own request mismatch')
                else:require(f['requests']==[],'Terminal request mask')
    require(np.all(np.asarray(state.winner)>=0),'Nonterminal campaign')
    return {'battles':n,'turns':turns,'events':events,'player_view_comparisons_including_padding':views,'pending_player_comparisons':pending,'disagreements':0}


def chance_checks(plan,cases):
    n=plan['sampler_games'];ids=np.arange(n,dtype=np.uint32);turns=ids%4+1
    draws=sample(plan['sampler_seed'],ids,turns);jax.block_until_ready(draws)
    tests={'jax_sampler':law_tests(*map(np.asarray,draws),turns)}
    c=[];r=[];ties=[];present=[]
    for case in cases:
        first=case['turns'][0];require(len(first['events'])==2,'First turn must execute both moves')
        ev=sorted(first['events'],key=lambda e:e['actor']);c.append([e['crit'] for e in ev]);r.append([e['roll'] for e in ev]);ties.append(first['tie_first'] or 0);present.append(first['tie_first'] is not None)
    tests['fresh_showdown_first_turn']=law_tests(c,r,ties,np.ones(len(c),int),present)
    require(sum(map(len,tests.values()))==plan['family_test_count'],'Family size')
    cutoff=plan['family_alpha']/plan['family_test_count']
    diagnostics={pop:{'empty_cells':sum(r['n']==0 for r in rows),
        'cells_unable_to_reject_zero_events':sum(r['zero_count_p_value']>=cutoff for r in rows),
        'duplicate_turn_one_marginals':35 if pop=='fresh_showdown_first_turn' else 0}
        for pop,rows in tests.items()}
    failures=[f'{pop}/{row["name"]}' for pop,rows in tests.items() for row in rows if row['p_value']<cutoff]
    return {'status':'passed' if not failures else 'failed','family_alpha':plan['family_alpha'],'bonferroni_cutoff':cutoff,'test_count':plan['family_test_count'],'failures':failures,'tests':tests,'diagnostics':diagnostics,
            'scope':'Finite non-rejection evidence, not a proof of PRNG independence. Fixed first-turn Showdown draws avoid outcome-dependent stopping, but 105 empty and 35 duplicate strata plus rare cells have limited detection power; see expected counts and zero-count diagnostics.'}


def rollout(master,ids):
    state=initial(len(ids))
    def body(s, _):
        chance=sample_kernel(master,ids,s.turn.astype(jnp.uint32))
        actions=jnp.stack([ids%2,(ids//2+s.turn)%2],axis=1).astype(jnp.int32)
        return step_kernel(s,actions,chance)[0],None
    return jax.lax.scan(body,state,None,length=4)[0]


def benchmark(plan):
    kernel=jax.jit(rollout);key=jax.random.key(517,impl='threefry2x32');rows=[]
    for n in plan['benchmark_batch_sizes']:
        ids=jnp.arange(n,dtype=jnp.uint32);start=time.perf_counter();state=kernel(key,ids);jax.block_until_ready(state)
        first=time.perf_counter()-start;times=[]
        for _ in range(plan['benchmark_repetitions']):
            start=time.perf_counter();result=kernel(key,ids);jax.block_until_ready(result);times.append(time.perf_counter()-start)
        require(np.all(np.asarray(state.winner)>=0),'Rollout did not finish')
        for x,y in zip(state,result):require(np.array_equal(x,y),'Rollout repeat drift')
        seconds=float(np.median(times));resolved=int(np.sum(np.asarray(state.turn)))
        rows.append({'batch':n,'first_compile_and_run_seconds':first,'median_seconds':seconds,'games_per_second':n/seconds,'resolved_turns_per_second':resolved/seconds,'submitted_decisions_per_second':2*resolved/seconds,'padded_turn_slots':4*n,'repetitions':len(times)})
    return {'label':plan['benchmark_label'],'rows':rows}


def run(out):
    out=Path(out);out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'contracts/jax-slice-v1/campaign.json').read_text());sources=provenance();old,new=load_references()
    require(len(old['cases'])==plan['conditional_reference_games'] and len(new['cases'])==plan['player_reference_games'],'Frozen campaign counts')
    runtime={'python':platform.python_version(),'platform':platform.platform(),'backend':jax.default_backend(),'x64_enabled':jax.config.x64_enabled,'devices':[str(d) for d in jax.devices()],
             'packages':{p:importlib.metadata.version(p) for p in ['jax','jaxlib','numpy','scipy','ml_dtypes','opt_einsum']}}
    require(runtime['backend']=='cpu','CPU campaign only')
    for name in ('requirements-learner.txt','requirements-jax.txt'):
        for line in (ROOT/name).read_text().splitlines():
            if '==' in line:
                package,version=line.strip().split('==')
                require(runtime['packages'].get(package)==version,'Dependency pin drift: '+package)
    admission={'version':'m1-g2-run-admission-1','plan':plan,'source_hashes':sources,'reference_hashes':[old['content_hash'],new['content_hash']],'runtime':runtime}
    (out/'admission.json').write_bytes(wire(admission)+b'\n')
    start=time.perf_counter();result={'version':'m1-g2-slice-results-1','admission_hash':digest(admission),'full_g2':'open','profile':PROFILE['id']}
    for row in old['damage_table']:require(int(damage(row['crit'],row['roll']))==row['damage'],'Damage table')
    result['damage_enumeration']=32
    result['conditional']=parity(old['cases']);print(json.dumps(result['conditional']),flush=True)
    result['player']=parity(new['cases'],True);print(json.dumps(result['player']),flush=True)
    result['chance']=chance_checks(plan,new['cases']);print(json.dumps({k:v for k,v in result['chance'].items() if k!='tests'}),flush=True)
    result['benchmark']=benchmark(plan);result['elapsed_seconds']=time.perf_counter()-start
    require(provenance()==sources,'Campaign source drift')
    result['status']='passed' if result['chance']['status']=='passed' else 'failed';result['content_hash']=digest(result)
    (out/'report.json').write_bytes(wire(result)+b'\n')
    require(result['status']=='passed','Chance gate failed; preserve run and investigate')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out');run(parser.parse_args().out)
