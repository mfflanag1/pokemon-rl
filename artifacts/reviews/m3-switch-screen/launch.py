"""Run the frozen development campaign once and preserve launcher outcome."""
import json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
review=ROOT/'artifacts/reviews/m3-switch-screen'
now=lambda:datetime.now(timezone.utc).isoformat()
command=['/usr/bin/caffeinate','-i',str(ROOT/'.venv/bin/python'),'-m','pokemon_switch_screen.campaign',str(ROOT/'artifacts/runs/m3-switch-screen-v1')]
with (review/'launch.json').open('x') as f:json.dump({'command':command,'started_at':now(),'launcher_pid':os.getpid()},f,indent=2)
with (review/'production.log').open('xb') as f:
 result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':'python'},stdout=f,stderr=subprocess.STDOUT)
with (review/'exit.json').open('x') as f:json.dump({'exit_code':result.returncode,'exited_at':now()},f,indent=2)
sys.exit(result.returncode)
