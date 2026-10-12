import assert from 'node:assert/strict';
import {readFileSync, writeFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {generateTeam} from '../reference-node/engine.mjs';
import {sha256} from '../reference-node/provenance.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {partition, teamKey} from '../dataset/identity.mjs';
import {playGame} from './campaign.mjs';

const [directory] = process.argv.slice(2);
if (!directory) throw new Error('Usage: node packages/showdown-bot/audit.mjs CAMPAIGN_DIR');
const report = JSON.parse(readFileSync(`${directory}/report.json`));
assert.equal(report.status, 'complete'); assert.equal(report.records.length, report.pairs * 2);
const counters = {decisions: 0, switches: 0, tera: 0, unavailable_retries: 0};
const scores = [], latencies = [], pairs = new Map();
for (const record of report.records) {
  assert.match(record.file, /^game-\d{3}-[01]\.json\.gz$/);
  const bytes = readFileSync(`${directory}/${record.file}`); assert.equal(sha256(bytes), record.sha256);
  const game = JSON.parse(gunzipSync(bytes));
  const teams = game.teams.map(t => {
    const entry = generateTeam(t.seed), payload = teamArtifact(entry, '0'.repeat(64)).payload;
    assert.equal(partition(payload), 'development'); assert.equal(teamKey(payload), t.team_key);
    return {...entry, partition: 'development', team_key: t.team_key};
  });
  const replay = await playGame({teams, seed: game.seed, policies: game.policies, policySeeds: game.policySeeds});
  assert.deepEqual(replay.input, game.input);
  assert.deepEqual(replay.outcome, game.outcome);
  assert.deepEqual(replay.spectator, game.spectator);
  const omitTiming = ({latency_ms, ...rest}) => rest;
  assert.deepEqual(replay.decisions.map(omitTiming), game.decisions.map(omitTiming));
  assert.equal(record.turns, game.outcome.turns);
  assert.deepEqual(game.policies, record.mirror ? ['random', 'heuristic'] : ['heuristic', 'random']);
  const score = !game.outcome.winner ? 0.5 : Number(game.outcome.winner === `Baseline ${record.mirror + 1}`);
  assert.equal(record.score, score); assert.ok(score >= 0 && score <= 1); scores.push(score);
  const identity = teams.map(t => t.team_key);
  if (pairs.has(record.pair)) assert.deepEqual(identity, pairs.get(record.pair));
  else pairs.set(record.pair, identity);
  for (const d of game.decisions) {
    counters.decisions++; counters.switches += Number(d.choice.startsWith('switch '));
    counters.tera += Number(d.choice.endsWith(' terastallize')); counters.unavailable_retries += d.errors.length;
    assert.ok(Number.isFinite(d.latency_ms) && d.latency_ms >= 0);
    if (d.side === `p${record.mirror + 1}`) latencies.push(d.latency_ms);
  }
}
for (const [key, value] of Object.entries(counters)) assert.equal(report[key], value);
assert.equal(report.mean_score, scores.reduce((a, b) => a + b, 0) / scores.length);
latencies.sort((a, b) => a - b);
assert.deepEqual(report.heuristic_latency_ms, {median: latencies[Math.floor(latencies.length * 0.5)],
  p95: latencies[Math.floor(latencies.length * 0.95)], max: latencies.at(-1)});
const result = {status: 'pass', games_replayed: report.records.length, actor_inputs_and_actions_reproduced: counters.decisions,
  report_sha256: sha256(readFileSync(`${directory}/report.json`)), final_teams_used: 0, paired_team_assignment_verified: true};
writeFileSync(`${directory}/audit.json`, JSON.stringify(result, null, 2), {flag: 'wx'});
console.log(JSON.stringify(result));
