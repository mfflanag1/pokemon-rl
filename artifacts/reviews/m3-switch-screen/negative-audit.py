"""Copied-evidence probes; original preflight is never edited."""
import copy,gzip,json,runpy,shutil,sys
from pathlib import Path
sys.path.insert(0,'python')
from pokemon_learner.checkpoint import digest,file_hash,wire
from pokemon_switch_screen.provenance import read
ROOT=Path.cwd();source=ROOT/'artifacts/runs/m3-switch-screen-smoke-02'
base=ROOT/'artifacts/runs/m3-switch-screen-negative-probes';base.mkdir(exist_ok=False)
audit=runpy.run_path('scripts/seal-switch-screen.py')['audit'];results=[]
def put(p,x):p.write_bytes(wire(x)+b'\n')
for name in ['smoke-as-production','runtime-drift','reserved-data','missing-cell','counter','checkpoint-next','action-logp','cutoff-as-win','view-prefix']:
 out=base/name;shutil.copytree(source,out);r=read(out/'report.json');smoke=True
 if name=='smoke-as-production':smoke=False
 if name=='runtime-drift':
  a=read(out/'preregistration.json');a['runtime']['x64']=True;put(out/'preregistration.json',a);r['admission_hash']=digest(a)
 if name=='reserved-data':r['final_evaluation_used']=True
 if name=='missing-cell':put(out/'matches-index.json',read(out/'matches-index.json')[:-1])
 if name in ('counter','checkpoint-next'):
  t=read(out/'training.json')
  if name=='counter':t[0]['checkpoints'][-1]['games']+=1
  else:t[0]['checkpoints'][-1]['next']['buffer']='0'*64
  put(out/'training.json',t);r['training']=t
 if name in ('action-logp','cutoff-as-win','view-prefix'):
  ix=read(out/'matches-index.json');p=out/ix[0]['file']
  with gzip.open(p,'rt') as f:rs=[json.loads(l) for l in f]
  if name=='action-logp':rs[0]['decisions'][0]['candidate_logp']+=.125
  elif name=='cutoff-as-win':
   q=next(x for x in rs if x['truncated']);q['truncated']=False;q['winner']='p1'
  else:rs[0]['decisions'][0]['view_lengths'][0]-=1
  with p.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:
   for x in rs:f.write(wire(x)+b'\n')
  ix[0]['sha256']=file_hash(p);put(out/'matches-index.json',ix)
 r['content_hash']=digest({k:v for k,v in r.items() if k!='content_hash'});put(out/'report.json',r)
 try:audit(out,smoke=smoke)
 except (ValueError,AssertionError,KeyError) as e:results.append({'case':name,'rejected':True,'reason':str(e)})
 else:raise AssertionError('Accepted corrupted evidence '+name)
put(ROOT/'artifacts/reviews/m3-switch-screen/negative-audit.json',{'status':'passed','cases':results})
print(json.dumps(results,indent=2))
