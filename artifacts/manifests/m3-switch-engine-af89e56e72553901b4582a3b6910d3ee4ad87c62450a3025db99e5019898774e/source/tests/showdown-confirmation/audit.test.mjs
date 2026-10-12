import test, {before, after} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, readFileSync, writeFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {gunzipSync, gzipSync} from 'node:zlib';
import {run} from '../../packages/showdown-confirmation/campaign.mjs';
import {freshTeams} from '../../packages/showdown-confirmation/teams.mjs';
import {audit} from '../../packages/showdown-confirmation/audit.mjs';

let root, directory, result;
before(async () => {
  root = mkdtempSync(join(tmpdir(), 'pokemon-panel-test-')); directory = join(root, 'smoke');
  result = await run(directory, true);
});
after(() => { if (root) rmSync(root, {recursive: true, force: true}); });

test('fresh official development panel reproduces every input, action, outcome and interval', async () => {
  const check = await audit(directory, {write: false});
  assert.equal(check.games_replayed, 24);
  assert.equal(check.actor_inputs_and_actions_reproduced, result.decisions);
  assert.equal(check.final_teams_used, 0);
  assert.equal(result.matrix.length, 6);
});

async function rejectsChange(file, mutate, compressed = false) {
  const path = join(directory, file), bytes = readFileSync(path);
  const value = JSON.parse(compressed ? gunzipSync(bytes) : bytes.toString());
  mutate(value);
  writeFileSync(path, compressed ? gzipSync(JSON.stringify(value)) : JSON.stringify(value));
  try { await assert.rejects(audit(directory, {write: false}), assert.AssertionError); }
  finally { writeFileSync(path, bytes); }
}

test('replay rejects an altered actor-visible request', async () => {
  await rejectsChange('game-0-0000-0.json.gz', x => { x.decisions[0].actor_input.request.side.pokemon[0].condition = '1/999'; }, true);
});
test('replay rejects altered mechanics seeds and policy seats', async () => {
  await rejectsChange('game-0-0000-0.json.gz', x => { x.seed = x.seed.replace(/.$/, c => c === '0' ? '1' : '0'); }, true);
  await rejectsChange('game-0-0000-0.json.gz', x => { x.policies.reverse(); }, true);
});
test('recount rejects a changed score even if the aggregate is left intact', async () => {
  await rejectsChange('report.json', x => { x.records[0].score = 0.5; });
});
test('recomputed pair bootstrap rejects a changed interval', async () => {
  await rejectsChange('report.json', x => { x.matrix[0].interval95 = [-1, 2]; });
});
test('regenerated development-team admission rejects a changed team', async () => {
  await rejectsChange('teams.json', x => { x[0].team[0].species = 'MissingNo.'; });
});


test('confirmation decision tampering is rejected by recomputation', async () => {
  await rejectsChange('report.json', x => { x.confirmation.status = x.confirmation.status === 'pass' ? 'not-passed' : 'pass'; });
});
test('prior selected-panel teams cannot be reused as a fresh pool', () => {
  assert.throws(() => freshTeams(2, 'showdown-opponent-panel-qualified-v1'), /overlaps/);
});
test('changed runtime manifest and thresholds cannot relabel a campaign', async () => {
  await rejectsChange('preregistration.json', x => { x.manifest.node = 'invented'; });
  await rejectsChange('preregistration.json', x => { x.plan.confirmation.opponent_delta_lower_threshold = -0.9; });
});
