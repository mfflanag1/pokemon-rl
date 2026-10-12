"""Non-vacuous post-freeze row-isolation probes using completed reference evidence."""
import copy,gzip,json,sys
from pathlib import Path
sys.path.insert(0,'python')
from pokemon_switch_screen import checkpoint as cp
from pokemon_switch_screen.provenance import read
from pokemon_switch_screen.inference import prefix,decide
from pokemon_learner.checkpoint import digest
root=Path.cwd();out=root/'artifacts/runs/m3-switch-screen-v1'
a=read(out/'preregistration.json');params=cp.load(out/'checkpoint-1009-0.json',1009,a).parameters
with gzip.open(out/'matches-1009-0-random.jsonl.gz','rt') as stream:records=[json.loads(next(stream)) for _ in range(50)]
def frame(r):return {**r,'group':0,'views':[prefix(v,n) for v,n in zip(r['final_views'],r['decisions'][0]['view_lengths'])]}
base=frame(records[0]);original=decide(params,params,[base],1009,'random','development')[0];checks=[]
for index in (2,8,16,24,48):
 changed=copy.deepcopy(base);changed['views'][1]=frame(records[index])['views'][1]
 assert digest(changed['views'][1])!=digest(base['views'][1]),'Vacuous opponent mutation'
 assert changed['views'][0]==base['views'][0]
 actual=decide(params,params,[changed],1009,'random','development')[0]
 for field in ('candidate_value','candidate_logp'):assert original[field]==actual[field]
 assert original['actions'][0]==actual['actions'][0]
 checks.append({'other_pair':index//2,'changed_other_view':True,'own_view_unchanged':True,'candidate_exact':True})
# Extra unrelated rows cannot affect the first actor or its RNG address.
extra=decide(params,params,[base,frame(records[24])],1009,'random','development')[0]
assert extra==original
p=root/'artifacts/reviews/m3-switch-screen/actor-isolation-probe.json';p.write_text(json.dumps({'status':'passed','mutations':checks,'batch_composition_exact':True},indent=2)+'\n')
print(p.read_text())
