#!/bin/zsh
set -u
cd /Users/maxf/Documents/ChatGPT/Pokemon || exit 1
if [[ -e artifacts/runs/m3-switch-soak-v1 || -e artifacts/reviews/m3-switch-soak/production.log ]]; then
  print -u2 'Production directory or log already exists; refusing duplicate run.'
  exit 1
fi
PYTHONPATH=python /usr/bin/caffeinate -i .venv/bin/python -m pokemon_switch_soak.campaign artifacts/runs/m3-switch-soak-v1 > artifacts/reviews/m3-switch-soak/production.log 2>&1
qualification_exit=$?
/usr/bin/env python3 -c 'import json,sys,datetime; from pathlib import Path; Path("artifacts/reviews/m3-switch-soak/exit.json").write_text(json.dumps({"exit_code":int(sys.argv[1]),"ended_at":datetime.datetime.now(datetime.timezone.utc).isoformat()})+"\n")' "$qualification_exit"
exit "$qualification_exit"
