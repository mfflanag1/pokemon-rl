"""Copied-evidence sealer probes; never modify the admitted smoke or predecessors."""
import json,runpy,shutil,tempfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'python'))
from pokemon_learner.checkpoint import digest,file_hash,wire
m=runpy.run_path(str(ROOT/'scripts/seal-switch-soak.py'));source=ROOT/'artifacts/runs/m3-switch-soak-smoke-v1'
before={p.name:file_hash(p) for p in source.iterdir() if p.is_file()};results=[]
for case in ('smoke-as-production','unfinished','runtime','missing-telemetry','broken-counters','changed-replay','checkpoint-corruption','partial-write','reserved-data'):
    with tempfile.TemporaryDirectory(prefix='switch-soak-negative-',dir=ROOT/'artifacts/runs') as temp:
        out=Path(temp);shutil.copytree(source,out,dirs_exist_ok=True)
        report=json.loads((out/'report.json').read_text())
        def write(name,d): (out/name).write_bytes(wire(d)+b'\n')
        if case=='unfinished':report['status']='running'
        if case=='runtime':
            a=json.loads((out/'admission.json').read_text());a['runtime']['backend']='gpu';write('admission.json',a);report['admission_hash']=digest(a)
        if case=='missing-telemetry':
            rows=(out/'telemetry.jsonl').read_text().splitlines();(out/'telemetry.jsonl').write_text('\n'.join(rows[:-1])+'\n');report['telemetry_sha256']=file_hash(out/'telemetry.jsonl')
        if case=='broken-counters':report['soak']['final_counters']['next_game']+=32
        if case=='changed-replay':
            r=json.loads((out/'replays.json').read_text());r[0]['next']['buffer']='0'*64;write('replays.json',r);report['replays_sha256']=file_hash(out/'replays.json')
        if case=='checkpoint-corruption':
            p=out/'start.json';p.write_bytes(p.read_bytes()+b' ')
        if case=='partial-write':(out/'new.json.partial').write_text('interrupted')
        if case=='reserved-data':report['final_evaluation_used']=True
        report.pop('content_hash');report['content_hash']=digest(report);write('report.json',report)
        try:
            if case=='smoke-as-production':m['seal'](out)
            else:m['audit'](out,smoke=True)
        except ValueError as error:results.append({'case':case,'rejected':str(error)})
        else:raise AssertionError('Unexpected acceptance: '+case)
assert before=={p.name:file_hash(p) for p in source.iterdir() if p.is_file()}
assert not (ROOT/'artifacts/reports/m3-switch-soak.json').exists()
print(json.dumps({'status':'passed','negatives':results,'production_unchanged':True,'receipt_written':False},indent=2))
