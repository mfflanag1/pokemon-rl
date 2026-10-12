import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {developmentTeams} from '../showdown-bot/development.mjs';
import {ROOT} from '../reference-node/provenance.mjs';

export const PRIOR_POOLS = ['artifacts/runs/showdown-panel-v1/teams.json', 'artifacts/runs/showdown-panel-smoke-v1/teams.json'];
export function freshTeams(count, namespace) {
  const prior = new Set(PRIOR_POOLS.flatMap(path => JSON.parse(readFileSync(`${ROOT}/${path}`)).map(t => t.team_key)));
  if (namespace !== 'showdown-tactical-confirmation-smoke-v1') {
    for (const t of developmentTeams(4, 'showdown-tactical-confirmation-smoke-v1')) prior.add(t.team_key);
    for (const t of developmentTeams(2, 'showdown-tactical-network-v1')) prior.add(t.team_key);
  }
  const teams = developmentTeams(count, namespace);
  assert(teams.every(t => !prior.has(t.team_key)), 'Fresh pool overlaps the selected panel; fail without replacement');
  return teams;
}
