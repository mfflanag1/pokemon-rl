import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {describe, seedFor} from '../reference-node/engine.mjs';
import {ROOT, sha256} from '../reference-node/provenance.mjs';
import {freshTeams} from './teams.mjs';
import {playGame} from '../showdown-panel/game.mjs';
import {ASSUMPTIONS} from '../showdown-panel/policy.mjs';
import {counters, countGame} from '../showdown-panel/report.mjs';
import {report} from './report.mjs';

export const PLAN = JSON.parse(readFileSync(new URL('./plan.json', import.meta.url), 'utf8'));
export function gameSpec(plan, teams, matchup, pair, mirror) {
  const [a, b] = plan.matchups[matchup], policies = mirror ? [b, a] : [a, b];
  return {teams: [teams[2 * pair], teams[2 * pair + 1]], seed: seedFor(`${plan.namespace}/battle/${pair}`), policies,
    policySeeds: policies.map(id => seedFor(`${plan.namespace}/policy/${id}/${pair}`)), maxDecisions: plan.max_decisions_per_game};
}
export const gameFile = (matchup, pair, mirror) => `game-${matchup}-${String(pair).padStart(4, '0')}-${mirror}.json.gz`;

export async function run(directory, smoke = false) {
  mkdirSync(directory); const records = [], plan = structuredClone(PLAN);
  if (smoke) { plan.pairs = 2; plan.namespace = 'showdown-tactical-confirmation-smoke-v1'; }
  const totals = counters(plan.policy_ids);
  try {
    const provenance = JSON.parse(execFileSync(`${ROOT}/.venv/bin/python`, [`${ROOT}/scripts/seal-showdown-confirmation.py`, '--admission'],
      {cwd: ROOT, encoding: 'utf8', maxBuffer: 4_000_000}));
    const manifest = describe(), teams = freshTeams(plan.pairs * 2, plan.namespace);
    writeFileSync(`${directory}/preregistration.json`, JSON.stringify({plan, smoke, assumptions: ASSUMPTIONS, manifest, provenance,
      frozen_at: new Date().toISOString()}, null, 2), {flag: 'wx'});
    writeFileSync(`${directory}/teams.json`, JSON.stringify(teams), {flag: 'wx'});
    for (let matchup = 0; matchup < plan.matchups.length; matchup++) for (let pair = 0; pair < plan.pairs; pair++) for (let mirror = 0; mirror < 2; mirror++) {
      const game = await playGame(gameSpec(plan, teams, matchup, pair, mirror));
      const file = gameFile(matchup, pair, mirror), bytes = gzipSync(JSON.stringify(game));
      writeFileSync(`${directory}/${file}`, bytes, {flag: 'wx'});
      const [a, b] = plan.matchups[matchup];
      records.push({a, b, matchup, pair, mirror, file, sha256: sha256(bytes), score: game.outcome.score[mirror], turns: game.outcome.turns});
      countGame(totals, game);
      writeFileSync(`${directory}/progress.json`, JSON.stringify({completed: records.length, total: plan.matchups.length * plan.pairs * 2}));
    }
    const result = report(plan, records, totals);
    writeFileSync(`${directory}/report.json`, JSON.stringify(result, null, 2), {flag: 'wx'}); return result;
  } catch (error) {
    writeFileSync(`${directory}/failure.json`, JSON.stringify({status: 'failed', completed: records.length, error: error.stack}, null, 2)); throw error;
  }
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [directory, mode] = process.argv.slice(2);
  if (!directory || mode && mode !== '--smoke') throw new Error('Usage: campaign.mjs NEW_DIR [--smoke]');
  const result = await run(directory, mode === '--smoke');
  console.log(JSON.stringify({status: result.status, games: result.games, decisions: result.decisions, confirmation: result.confirmation}));
}
