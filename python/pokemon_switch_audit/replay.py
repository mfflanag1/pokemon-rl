"""Reconstruct every accepted existing teacher row without new trajectories."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from pokemon_full.features import encode
from pokemon_full_train.storage import ROOT,atomic_json,digest,canonical,source_paths,runtime
from pokemon_battle_audit.core import fingerprint,record_step
from .transport import Worker

def inspect_game(worker, game, arrays, offset):
    frame=worker.request('start',**game['start']);records=[];n=0
    if frame['provenance']!=game['provenance']:raise ValueError('Changed teacher provenance')
    for step in game['steps']:
        if fingerprint(frame['players'])!=step['before']:raise ValueError('Original player view differs')
        before=frame;frame=worker.request('step',**step['command'])
        if record_step(before,step['command'],step['values'],frame)!=step:raise ValueError('Original transition differs')
        for submitted in frame['submitted']:
            if not submitted['accepted']:continue
            side=submitted['side'];p=before['players'][side];action=step['command']['actions'][side]
            identity={'episode':game['episode'],'side':side,'request_id':p['request_id'],'action':action,'phase':p['phase']}
            if identity!=game['row_identity'][n] or action!=before['tactical_actions'][side]:raise ValueError('Teacher identity differs')
            x,m=encode(p);j=offset+n
            if not np.array_equal(x,arrays['features'][j]) or not np.array_equal(m,arrays['mask'][j]) or action!=int(arrays['targets'][j]) or game['episode']!=int(arrays['game_index'][j]):raise ValueError('Original encoded row differs')
            own=next(q for q in p['own'] if q['active']);foe=next(q for q in p['opponent'] if q and q['active'])
            metadata={'game':game['episode'],'phase':p['phase'],'target':action,'own_species':own['species'],'foe_species':foe['species'],
                'destination_species':p['own'][action-8]['species'] if action>=8 else '',
                'feature_hash':hashlib.sha256(x.tobytes()+m.tobytes()).hexdigest()}
            records.append({'identity':identity,'metadata':metadata,'player':p,'advice':before['projection_audit'][side]});n+=1
    if not frame['outcome']:frame=worker.request('cutoff')
    if n!=game['accepted_rows'] or frame['outcome']!=game['outcome']:raise ValueError('Original terminal/row count differs')
    return records

def run(out):
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    sources={str(p.relative_to(ROOT)):digest(p) for p in source_paths()};rt=runtime()
    atomic_json(out/'source-binding.json',{'sources':sources,'runtime':rt})
    inputs={};parts={};counts={}
    for split,base,dataset,count in [('train','artifacts/runs/f15-full-imitation-v1/teacher','artifacts/runs/f16-balanced-imitation-v1/inputs/teacher.npz',512),('probe','artifacts/runs/f16-balanced-imitation-v1/probe','artifacts/runs/f16-balanced-imitation-v1/probe/dataset.npz',128)]:
        with np.load(ROOT/dataset,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        inputs[dataset]=digest(ROOT/dataset);games=[];offsets=[];offset=0
        for i in range(count):
            path=ROOT/base/f'game-{i:04d}.json';inputs[str(path.relative_to(ROOT))]=digest(path);g=json.loads(path.read_text());games.append(g);offsets.append(offset);offset+=g['accepted_rows']
        if offset!=len(arrays['targets']):raise ValueError('Missing source rows')
        def work(slot):
            results=[]
            with Worker(out/split/f'worker-{slot}') as worker:
                for i in range(slot,count,8):
                    records=inspect_game(worker,games[i],arrays,offsets[i]);atomic_json(out/split/f'game-{i:04d}.json',records);results.extend(records)
            return results
        with ThreadPoolExecutor(max_workers=8) as ex:records=[r for batch in ex.map(work,range(8)) for r in batch]
        records.sort(key=lambda r:(r['metadata']['game'],r['identity']['request_id'],r['identity']['side']))
        parts[split]=records;counts[split]={'games':count,'rows':len(records),'terminals':sum(g['outcome']['kind']=='terminal' for g in games),'cutoffs':sum(g['outcome']['kind']=='administrative_cutoff' for g in games)}
        atomic_json(out/'progress.json',{'split':split,'counts':counts,'elapsed_seconds':time.perf_counter()-start})
    from .coverage import summarize
    coverage=summarize([r['metadata'] for r in parts['train']],[r['metadata'] for r in parts['probe']])
    advice={}
    for split,records in parts.items():
        advice[split]={}
        for family in ['all','normal_move','tera_move','voluntary_switch','forced_switch']:
            def belongs(r):
                m=r['metadata'];f='forced_switch' if m['phase']!='move' else 'normal_move' if m['target']<4 else 'tera_move' if m['target']<8 else 'voluntary_switch'
                return family=='all' or f==family
            chosen=[r for r in records if belongs(r)];advice[split][family]={'rows':len(chosen),'disagreements':{name:sum(r['advice'][name]!=r['metadata']['target'] for r in chosen) for name in ['projection','entry','types','boosts','entry_types_boosts']}}
    atomic_json(out/'inputs.json',inputs)
    if any(digest(ROOT/p)!=h for p,h in inputs.items()) or any(digest(ROOT/p)!=h for p,h in sources.items()) or runtime()!=rt:raise ValueError('Source/input/runtime drift')
    report={'status':'passed','scope':'All existing F15 fit and F16 probe teacher rows; adaptive diagnostic, not new held-out evaluation or strength evidence. Public teacher reconstruction is a diagnostic, not an equivalent teacher or actor. Entry/types/boosts restoration covers only those fields; it is not a complete public-history restoration or a ceiling on achievable agreement. Other raw-request/projection differences remain confounded, and no causal attribution follows from residual disagreement.','counts':counts,'coverage':coverage,'advice':advice,'elapsed_seconds':time.perf_counter()-start}
    atomic_json(out/'report.json',report);print(json.dumps({'counts':counts,'advice':advice,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();run(args.out.resolve())
