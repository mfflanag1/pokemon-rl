import {generateTeam, seedFor} from '../reference-node/engine.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {partition, teamKey} from '../dataset/identity.mjs';
import {execFileSync} from 'node:child_process';
import {ROOT} from '../reference-node/provenance.mjs';

export function baselineAdmission() {
  return JSON.parse(execFileSync(`${ROOT}/.venv/bin/python`, [`${ROOT}/scripts/seal-playable-baseline.py`, '--admission'],
    {cwd: ROOT, encoding: 'utf8', maxBuffer: 4_000_000}));
}

// Only the gate sees rejected candidates. Actors and battle logs receive development teams only.
export function developmentTeams(count, namespace) {
  const result = [], seen = new Set();
  for (let attempt = 0; result.length < count && attempt < count * 500; attempt++) {
    const entry = generateTeam(seedFor(`${namespace}/team/${attempt}`));
    const payload = teamArtifact(entry, '0'.repeat(64)).payload;
    if (partition(payload) !== 'development' || seen.has(teamKey(payload))) continue;
    seen.add(teamKey(payload));
    result.push({...entry, partition: 'development', team_key: teamKey(payload)});
  }
  if (result.length !== count) throw new Error('Development team sampling budget exhausted');
  return result;
}
