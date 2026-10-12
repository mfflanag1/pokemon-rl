import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ROOT} from '../reference-node/provenance.mjs';
import {developmentTeams} from '../showdown-bot/development.mjs';
import {Teams, Dex} from '../reference-node/engine.mjs';
import {planFor} from './schedule.mjs';

export const developmentFormat = () => ({name: '[Gen 9] Bot Baseline Development', mod: 'gen9', gameType: 'singles',
  searchShow: false, rated: false, ruleset: [...Dex.formats.get('gen9randombattle').ruleset]});

export const PRIOR_POOLS = [
  'showdown-panel-v1', 'showdown-panel-smoke-v1', 'showdown-confirmation-v1',
  'showdown-confirmation-smoke-v1', 'showdown-tactical-network-v1',
  'showdown-session-v1', 'playable-baseline-network-v2',
].map(name => `artifacts/runs/${name}/teams.json`);

export function freshTeams(smoke = false) {
  const plan = planFor(smoke), excluded = new Set();
  for (const path of PRIOR_POOLS) for (const team of JSON.parse(readFileSync(`${ROOT}/${path}`, 'utf8'))) excluded.add(team.team_key);
  if (!smoke) {
    const pilot = planFor(true);
    for (const team of developmentTeams(pilot.games, pilot.namespace)) excluded.add(team.team_key);
  }
  const teams = developmentTeams(plan.games, plan.namespace);
  assert(teams.every(team => !excluded.has(team.team_key)), 'Exact-team overlap with prior or smoke pool');
  return teams;
}

// The loopback-only private receipt uses the simulator's original side.team,
// unlike the first move request, which can already contain Imposter or consumed items.
export function teamReceiptLine(team) {
  const received = Teams.unpack(Teams.pack(team));
  // Generated Random Battle sets bypass constructed-team legality. Only the
  // simulator's canonical default nickname is applied; no set fields are omitted.
  for (const set of received) if (!set.name || set.name === set.species) set.name = Dex.species.get(set.species).baseSpecies;
  return `|soakteam|${Teams.pack(received)}`;
}
