import {mkdirSync, writeFileSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import {pathToFileURL} from 'node:url';
import {performance} from 'node:perf_hooks';
import {BattleStream, describe, seedFor} from '../reference-node/engine.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {sha256} from '../reference-node/provenance.mjs';
import {makePolicy, POLICY_ID} from './policy.mjs';
import {legalChoices, PublicBattle} from './protocol.mjs';
import {baselineAdmission, developmentTeams} from './development.mjs';

export async function playGame({teams, seed, policies = ['heuristic', 'random'], policySeeds, maxDecisions = 20_000}) {
  const projection = new PlayerProjection(), states = [new PublicBattle(), new PublicBattle()], cursors = [0, 0];
  const actors = policies.map((kind, i) => makePolicy(kind, policySeeds?.[i] ?? seedFor(`${seed}/actor/${i}`)));
  const decisions = [], input = []; let outcome = null;
  const stream = new class extends BattleStream {
    constructor() { super({noCatch: true}); }
    pushMessage(type, data) { projection.ingest(type, data); if (type === 'end') outcome = JSON.parse(data); }
  }();
  async function write(text) { input.push(text); await stream.write(text); }
  const start = performance.now();
  try {
    await write(`>start ${JSON.stringify({formatid: 'gen9randombattle', seed})}\n` + teams.map((t, i) =>
      `>player p${i + 1} ${JSON.stringify({name: `Baseline ${i + 1}`, team: t.team})}`).join('\n'));
    while (!outcome) {
      let progressed = false;
      for (let i = 0; i < 2 && !outcome; i++) {
        const side = `p${i + 1}`, player = projection.players[side];
        while (cursors[i] < player.protocol.length) states[i].ingest(player.protocol[cursors[i]++]);
        if (!player.pending || !player.request || player.request.wait) continue;
        if (decisions.length >= maxDecisions) throw new Error('Incomplete game: decision cap reached');
        const request = structuredClone(player.request), view = structuredClone(states[i]), requestId = player.request_id;
        const t0 = performance.now(), choice = actors[i](request, view), latencyMs = performance.now() - t0;
        if (!legalChoices(request).includes(choice)) throw new Error('Policy chose an unavailable request action');
        player.pending = false;
        const before = projection.errors.length;
        await write(`>${side} ${choice}`);
        const errors = projection.errors.slice(before);
        if (errors.some(e => !e.message.startsWith('[Unavailable choice]'))) throw new Error(`Invalid choice: ${JSON.stringify(errors)}`);
        if (errors.length && !player.pending) throw new Error('Unavailable action without refreshed request');
        decisions.push({side, request_id: requestId, turn: view.turn, choice, latency_ms: latencyMs,
          actor_input: {request, public: view}, errors});
        progressed = true;
      }
      if (!progressed && !outcome) throw new Error('No pending player: stalled game');
    }
    if (outcome.winner && !['Baseline 1', 'Baseline 2'].includes(outcome.winner)) throw new Error('Unknown terminal winner');
    const score = [1, 2].map(i => !outcome.winner ? 0.5 : Number(outcome.winner === `Baseline ${i}`));
    return {teams: teams.map(t => ({seed: t.seed, team_key: t.team_key, partition: t.partition})), seed, policies,
      policySeeds: policySeeds ?? null, input, decisions, outcome: {winner: outcome.winner, turns: outcome.turns, score, pokemon_left: outcome.score},
      elapsed_ms: performance.now() - start, spectator: projection.spectator};
  } finally { stream.battle?.destroy(); }
}

export async function runCampaign(directory, pairs = 16, namespace = 'playable-baseline-v1') {
  if (!Number.isSafeInteger(pairs) || pairs < 1 || pairs > 200) throw new Error('pairs must be 1..200');
  mkdirSync(directory); // Preserve failed and successful runs; never overwrite.
  const started = new Date().toISOString(), manifest = describe(), records = [];
  const latencies = [], counters = {decisions: 0, switches: 0, tera: 0, unavailable_retries: 0};
  writeFileSync(`${directory}/admission.json`, JSON.stringify({manifest, namespace, pairs, policies: [POLICY_ID, 'random-move-0.9-no-tera-v1'],
    split: 'development', provenance: baselineAdmission()}, null, 2));
  try {
    const teams = developmentTeams(pairs * 2, namespace);
    for (let pair = 0; pair < pairs; pair++) for (let mirror = 0; mirror < 2; mirror++) {
      const policies = mirror ? ['random', 'heuristic'] : ['heuristic', 'random'];
      const game = await playGame({teams: [teams[pair * 2], teams[pair * 2 + 1]],
        seed: seedFor(`${namespace}/battle/${pair}`), policies,
        policySeeds: [seedFor(`${namespace}/random/${pair}`), seedFor(`${namespace}/random/${pair}`)]});
      const file = `game-${String(pair).padStart(3, '0')}-${mirror}.json.gz`, bytes = gzipSync(JSON.stringify(game));
      writeFileSync(`${directory}/${file}`, bytes, {flag: 'wx'});
      const candidateSeat = mirror;
      records.push({pair, mirror, file, sha256: sha256(bytes), score: game.outcome.score[candidateSeat], turns: game.outcome.turns});
      for (const d of game.decisions) {
        counters.decisions++; counters.switches += Number(d.choice.startsWith('switch '));
        counters.tera += Number(d.choice.endsWith(' terastallize')); counters.unavailable_retries += d.errors.length;
        if (d.side === `p${candidateSeat + 1}`) latencies.push(d.latency_ms);
      }
      writeFileSync(`${directory}/progress.json`, JSON.stringify({completed: records.length, total: pairs * 2, ...counters}));
    }
    latencies.sort((a, b) => a - b);
    const pairScores = Array.from({length: pairs}, (_, p) => records.filter(r => r.pair === p).reduce((a, r) => a + r.score, 0) / 2);
    const mean = pairScores.reduce((a, b) => a + b, 0) / pairs;
    const report = {schema: 'playable-baseline-campaign-v1', status: 'complete', started, finished: new Date().toISOString(),
      namespace, pairs, games: records.length, mean_score: mean, pair_scores: pairScores, ...counters,
      heuristic_latency_ms: {median: latencies[Math.floor(latencies.length * 0.5)], p95: latencies[Math.floor(latencies.length * 0.95)], max: latencies.at(-1)},
      records, interpretation: 'Descriptive development baseline only; no Elo, learner improvement or final-data claim.'};
    writeFileSync(`${directory}/report.json`, JSON.stringify(report, null, 2));
    return report;
  } catch (error) { writeFileSync(`${directory}/failure.json`, JSON.stringify({status: 'failed', error: error.stack, completed: records.length})); throw error; }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [directory, pairs, namespace] = process.argv.slice(2);
  if (!directory) throw new Error('Usage: node packages/showdown-bot/campaign.mjs NEW_OUTPUT_DIR [PAIRS] [NAMESPACE]');
  console.log(JSON.stringify(await runCampaign(directory, pairs === undefined ? 16 : Number(pairs), namespace)));
}
