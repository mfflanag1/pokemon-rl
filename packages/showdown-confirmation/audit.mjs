import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {pathToFileURL} from 'node:url';
import {sha256} from '../reference-node/provenance.mjs';
import {freshTeams} from './teams.mjs';
import {describe} from '../reference-node/engine.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {partition} from '../dataset/identity.mjs';
import {PLAN, gameSpec, gameFile} from './campaign.mjs';
import {ASSUMPTIONS} from '../showdown-panel/policy.mjs';
import {playGame} from '../showdown-panel/game.mjs';
import {counters, countGame} from '../showdown-panel/report.mjs';
import {report as summarize} from './report.mjs';

export async function audit(directory, {write = true} = {}) {
  const read = file => JSON.parse(readFileSync(`${directory}/${file}`, 'utf8'));
  const registration = read('preregistration.json'), plan = structuredClone(PLAN), report = read('report.json');
  assert.equal(typeof registration.smoke, 'boolean'); assert.deepEqual(registration.manifest, describe());
  if (registration.smoke) { plan.pairs = 2; plan.namespace = 'showdown-tactical-confirmation-smoke-v1'; }
  assert.deepEqual(registration.plan, plan); assert.deepEqual(registration.assumptions, ASSUMPTIONS);
  const teams = read('teams.json'); assert.deepEqual(teams, freshTeams(plan.pairs * 2, plan.namespace));
  const splits = teams.map(team => partition(teamArtifact(team, '0'.repeat(64)).payload));
  assert(splits.every(split => split === 'development'), 'Only development teams may enter the panel');
  const totals = counters(plan.policy_ids), records = []; let replayed = 0;
  const omitTiming = ({latency_ms, ...rest}) => rest;
  for (let matchup = 0; matchup < plan.matchups.length; matchup++) for (let pair = 0; pair < plan.pairs; pair++) for (let mirror = 0; mirror < 2; mirror++) {
    const file = gameFile(matchup, pair, mirror), bytes = readFileSync(`${directory}/${file}`);
    const game = JSON.parse(gunzipSync(bytes)), spec = gameSpec(plan, teams, matchup, pair, mirror);
    const replay = await playGame(spec);
    const {decisions: oldDecisions, ...oldRest} = game, {decisions: newDecisions, ...newRest} = replay;
    assert.deepEqual(oldRest, newRest, 'Fresh simulator replay mismatch');
    assert.deepEqual(oldDecisions.map(omitTiming), newDecisions.map(omitTiming), 'Actor inputs/actions mismatch');
    const [a, b] = plan.matchups[matchup];
    records.push({a, b, matchup, pair, mirror, file, sha256: sha256(bytes), score: game.outcome.score[mirror], turns: game.outcome.turns});
    countGame(totals, game); replayed++;
    if (write && replayed % 100 === 0) writeFileSync(`${directory}/audit-progress.json`, JSON.stringify({replayed, total: plan.matchups.length * plan.pairs * 2}));
  }
  assert.deepEqual(report, summarize(plan, records, totals), 'Summary/counter/interval mismatch');
  const result = {status: 'pass', games_replayed: replayed, actor_inputs_and_actions_reproduced: report.decisions,
    report_sha256: sha256(readFileSync(`${directory}/report.json`)),
    preregistration_sha256: sha256(readFileSync(`${directory}/preregistration.json`)),
    teams_sha256: sha256(readFileSync(`${directory}/teams.json`)), final_teams_used: splits.filter(split => split === 'final').length,
    admitted_development_teams: splits.length,
    pairing_seeds_policy_seats_and_intervals_verified: true};
  if (write) {
    writeFileSync(`${directory}/audit-progress.json`, JSON.stringify({replayed, total: replayed}));
    writeFileSync(`${directory}/audit.json`, JSON.stringify(result, null, 2), {flag: 'wx'});
  }
  return result;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw new Error('Usage: audit.mjs RUN_DIR');
  console.log(JSON.stringify(await audit(process.argv[2])));
}
