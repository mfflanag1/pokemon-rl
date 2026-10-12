"""Fresh sharded player-view parity and exact finite-word qualification on local CPU."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import time
import subprocess
import traceback
import jax
import jax.numpy as jnp
import numpy as np
from pokemon_binary.qualification import admission as prior_admission
from pokemon_slice.engine import PROFILE
from pokemon_jax.campaign import parity
from pokemon_learner.checkpoint import digest,wire,file_hash
from pokemon_learner.core import require
from pokemon_reference.dataset_gate import checked_file
from .mapping import kernel
from .loop import ROOT,Executables,start,plan,check_report
from .checkpoint import Runner,provenance,save,load,state

CAPTURE=ROOT/'artifacts/runs/m1-showdown-player-v1'
WORDS=ROOT/'artifacts/runs/m1-showdown-words-v1.json'
CONTRACT_PATH=ROOT/'contracts/showdown-chance-v1/contract.json'


def read(path):return json.loads(path.read_bytes())
def write(path,obj):path.write_bytes(wire(obj)+b'\n')
def sealed(doc):
    require(doc['content_hash']==digest({k:v for k,v in doc.items() if k!='content_hash'}),'Content hash')
    return doc


def capture_index():
    c=read(CONTRACT_PATH);p=read(CAPTURE/'plan.json');i=sealed(read(CAPTURE/'index.json'))
    require(p['version']=='showdown-player-capture-1' and p['contract']==c,'Capture contract')
    require(i['version']=='showdown-player-index-1' and i['status']=='complete' and i['plan_hash']==digest(p),'Capture plan seal')
    count=c['fresh_capture']['count'];size=c['fresh_capture']['shard_size']
    require(count==10000 and size==100 and i['count']==count and len(i['shards'])==count//size,'Capture count')
    for number,s in enumerate(i['shards']):
        require(s['file']==f'cases-{number:04d}.jsonl.gz' and s['first']==number*size and s['count']==size,'Shard identity')
        checked_file(CAPTURE,s['file'],s['sha256'])
    expected={'packages/showdown-chance/reference.mjs','packages/showdown-chance/word-oracle.mjs',
        'contracts/showdown-chance-v1/contract.json','packages/jax-slice/reference.mjs','packages/slice/reference.mjs',
        'contracts/slice-v1/profile.json','contracts/jax-slice-v1/view.md',
        *[f'packages/reference-node/{n}.mjs' for n in ('engine','provenance','observations')],
        *[f'vendor/pokemon-showdown/sim/{n}.ts' for n in ('prng','battle','battle-actions','battle-queue','pokemon','side')]}
    require(set(p['sources'])==expected,'Capture source keys')
    for path,sha in p['sources'].items():checked_file(ROOT,path,sha)
    require(p['reference']['commit']==c['showdown_commit'] and p['reference']['source_dirty'] is False,'Upstream pin')
    return p,i


def admission():
    c=read(CONTRACT_PATH);receipt=sealed(read(ROOT/'artifacts/reports/m1-binary-policy.json'))
    require(receipt['content_hash']==c['predecessor'],'Binary-policy receipt pin')
    for path,sha in receipt['source_files'].items():
        if path.startswith(('python/','contracts/')) or path in ('scripts/seal-binary-policy.py','requirements-jax.txt','requirements-learner.txt','requirements-contract.txt'):
            checked_file(ROOT,path,sha)
    p,i=capture_index();oracle=sealed(read(WORDS))
    require(oracle['version']=='showdown-word-oracle-1' and oracle['reference']==p['reference'],'Word/reference identity')
    for path,sha in oracle['sources'].items():checked_file(ROOT,path,sha)
    prior=prior_admission()
    require(p['profile']==PROFILE and PROFILE['id']==c['profile'],'Profile')
    current=json.loads(subprocess.check_output(['node','--input-type=module','-e',
        "import {describe} from './packages/reference-node/engine.mjs';console.log(JSON.stringify(describe()));"],cwd=ROOT))
    require(p['reference']==current,'Current pinned reference manifest')
    require(jax.default_backend()=='cpu' and not jax.config.x64_enabled,'CPU FP32 qualification')
    return {'contract':c,'loop_plan':plan(),'provenance':provenance(),'predecessor':receipt['content_hash'],
            'prior':prior,'capture_index_hash':i['content_hash'],'capture_plan_hash':digest(p),
            'capture_files':{'plan.json':file_hash(CAPTURE/'plan.json'),'index.json':file_hash(CAPTURE/'index.json'),**{s['file']:s['sha256'] for s in i['shards']}},
            'word_file_hash':file_hash(WORDS),'word_content_hash':oracle['content_hash']}


def verify_case(case,expected_id):
    c=read(CONTRACT_PATH);ns=c['fresh_capture']
    require(case['id']==expected_id and case['seed']=='sodium,'+digest(['pokemon-reference-r1',f"{ns['namespace']}/{expected_id}"]),'Fresh battle seed/identity')
    require(2<=len(case['turns'])<=4,'Case turn bound')
    result=copy.deepcopy(case);previous=case['initial'];previous_index=-1;counts={'crit':0,'roll':0,'tie':0}
    for turn,t in enumerate(result['turns'],1):
        h=digest([ns['actions_namespace'],expected_id,turn]);actions=[int(h[k],16)%2 for k in (0,1)]
        require(t['turn']==turn and t['actions']==actions and t['submitted_first']==(expected_id+turn)%2,'Action/phase schedule')
        require(t['pending']==previous,'Pending commitment disclosure');previous=t['views']
        require(1<=len(t['events'])<=2 and len({e['actor'] for e in t['events']})==len(t['events']),'Executed event identity')
        draws=t['chance'];words=[]
        for d in draws:
            require(d['role'] in counts and type(d['word']) is int and 0<=d['word']<2**32,'Chance role/word')
            require(type(d['index']) is int and d['index']>previous_index,'Word consumption order');previous_index=d['index'];words.append(d['word']);counts[d['role']]+=1
        require(len(words)>0,'Missing chance tape');mapped={k:np.asarray(v) for k,v in kernel(jnp.asarray(words,jnp.uint32)).items()}
        grouped={};ties=[];hit_started=False
        for k,d in enumerate(draws):
            if d['role']=='tie':
                require(not hit_started and d['order'] in ([0,1],[1,0]),'Tie context/order')
                value=int(mapped['tie0' if d['order'][0]==0 else 'tie1'][k]);ties.append(value)
                require(type(d['result']) is int and d['result']==value,'Raw tie map')
            else:
                hit_started=True;a=d['actor'];require(type(a) is int and a in (0,1),'Chance actor')
                key=(d['role'],a);require(key not in grouped,'Repeated chance role')
                value=bool(mapped['crit'][k]) if d['role']=='crit' else int(mapped['roll'][k])
                require(d['result']==(value if d['role']=='crit' else 100-value),'Raw critical/roll map')
                grouped[key]=value
        require(bool(ties)==(actions[0]==actions[1]),'Tie eligibility')
        require(t['tie_first']==(ties[-1] if ties else None),'Effective tie result')
        require(len(grouped)==2*len(t['events']),'Chance data for canceled/missing move')
        for e in t['events']:
            a=e['actor'];require(('crit',a) in grouped and ('roll',a) in grouped,'Missing executed move draw')
            require(e['crit']==grouped['crit',a] and e['roll']==grouped['roll',a],'Event chance mismatch')
            e['crit']=grouped['crit',a];e['roll']=grouped['roll',a]
        if ties:require(t['events'][0]['actor']==ties[-1],'Last shuffle does not govern move')
    return result,counts


def word_audit():
    doc=sealed(read(WORDS));c=read(CONTRACT_PATH);rows=doc['rows']
    require(doc['random_words']==c['word_audit']['random_words']==100000,'Word audit budget')
    edges={0,1,2**32-1}
    for n in (24,16,2):
        for k in range(1,n):
            edge=(k*2**32+n-1)//n;edges.update((edge-1,edge,edge+1))
    require(doc['boundary_words']==len(edges) and [r['word'] for r in rows[:len(edges)]]==sorted(edges)
            and len(rows)==100000+len(edges),'Word boundary coverage')
    actual={k:np.asarray(v) for k,v in kernel(jnp.array([r['word'] for r in rows],jnp.uint32)).items()}
    for k in ('bucket24','crit','roll'):require(np.array_equal(actual[k],[r[k] for r in rows]),'Pinned upstream word mismatch')
    for side in (0,1):require(np.array_equal(actual[f'tie{side}'],[r['ties'][side] for r in rows]),'Pinned upstream shuffle mismatch')
    # Analytic bin widths enumerate the whole input partition, not all 2^32 words.
    bounds=[(k*2**32+23)//24 for k in range(25)];bins=np.diff(bounds).tolist()
    require(sum(bins)==2**32 and bins[0]==178956971 and set(bins)=={178956970,178956971},'Critical bucket partition')
    return {'status':'passed','words':len(rows),'boundary_words':len(edges),'output_comparisons':5*len(rows),
            'critical_bucket_counts':bins,'critical_numerator':bins[0],'denominator':2**32,
            'uniform_word_assumption':True,'raw_seed_equivalence':False,'disagreements':0}


def player_parity(out):
    _,index=capture_index();total={k:0 for k in ('battles','turns','events','player_view_comparisons_including_padding','pending_player_comparisons','disagreements')}
    counts={'crit':0,'roll':0,'tie':0};seen=set();rows=[]
    # Exclude every prior conditional and player-view reference seed by construction.
    old={'sodium,'+digest(['pokemon-reference-r1',f'{ns}/{i}']) for ns,n in
         [('m1-slice-reference-v1',10000),('m1-g2-player-reference-v1',1000)] for i in range(n)}
    for shard in index['shards']:
        with gzip.open(CAPTURE/shard['file'],'rt') as stream:cases=[json.loads(line) for line in stream]
        require(len(cases)==shard['count'],'Shard row count');derived=[]
        for i,c in enumerate(cases,shard['first']):
            require(c['seed'] not in seen and c['seed'] not in old,'Duplicate/nonfresh mechanics seed');seen.add(c['seed'])
            case,n=verify_case(c,i);derived.append(case)
            for k in counts:counts[k]+=n[k]
        report=parity(derived,with_views=True)
        for k in total:total[k]+=report[k]
        rows.append({'first':shard['first'],'capture_sha256':shard['sha256'],**report})
        print(json.dumps({'verified_battles':total['battles'],'disagreements':total['disagreements']}),flush=True)
    require(len(seen)==10000 and total['battles']==10000 and total['player_view_comparisons_including_padding']==180000,'Full player campaign coverage')
    write(out/'player-shards.json',rows)
    return {'status':'passed',**total,'raw_chance_checks':counts,'fresh_distinct_seeds':len(seen),
            'disjoint_from_prior_reference_seeds':True,'scope':'Player protocol reducer and deterministic actor probe; conditional transitions driven by decoded upstream chance words. Hidden-set strata remain open.'}


def integration(out):
    c=read(CONTRACT_PATH);exe=Executables();exe.diagnostics();rows=[]
    for seed in c['integration_seeds']:
        runner=Runner(seed,exe)
        for _ in range(c['integration_updates']):runner.run()
        path=out/f'checkpoint-{seed}.json';save(path,runner);other=load(path,exe)
        # Commit next cycle on both; comparison includes exact FP32 wire bytes of state.
        left=runner.run();right=other.run()
        require(state(runner)==state(other),'Exact disk next-state replay')
        for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)):require(np.array_equal(a,b),'Replay metrics')
        rows.append({'seed':seed,'updates':c['integration_updates'],'games':c['integration_updates']*64,
                     'checkpoint_sha256':file_hash(path),'next_cycle_exact':True})
    return {'status':'passed','seeds':rows,'traces':exe.traces,'two_hour_qualification':False}


def main(directory):
    out=Path(directory).resolve();require(out.is_relative_to(ROOT/'artifacts/runs'),'Run output');out.mkdir(exist_ok=False)
    result={'status':'running','full_g2':'open','full_g3':'open','two_hour_qualification':False,'prime_used':False};start_time=time.perf_counter()
    try:
        admitted=admission();write(out/'admission.json',admitted);result['admission_hash']=digest(admitted)
        result['mapping']=word_audit();result['player']=player_parity(out);result['integration']=integration(out)
        require(admission()==admitted,'End admission/source drift');result['final_reservation_unchanged']=True;result['status']='passed'
    except Exception as exc:result.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start_time;result['content_hash']=digest(result);write(out/'report.json',result)
    print(json.dumps({'status':result['status'],'content_hash':result['content_hash'],'player':result['player']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');main(parser.parse_args().directory)
