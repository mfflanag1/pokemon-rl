import {performance} from 'node:perf_hooks';
import {BattleStream} from '../reference-node/engine.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {legalChoices} from '../showdown-bot/protocol.mjs';
import {createActor} from './policy.mjs';

export async function playGame({teams, seed, policies, policySeeds, maxDecisions = 20_000}) {
  const projection = new PlayerProjection(), actors = policies.map((id, i) => createActor(id, policySeeds[i]));
  const cursors = [0, 0], decisions = [], input = []; let outcome = null;
  const stream = new class extends BattleStream {
    constructor() { super({noCatch: true}); }
    pushMessage(type, data) { projection.ingest(type, data); if (type === 'end') outcome = JSON.parse(data); }
  }();
  async function write(text) { input.push(text); await stream.write(text); }
  try {
    await write(`>start ${JSON.stringify({formatid: 'gen9randombattle', seed})}\n` + teams.map((t, i) =>
      `>player p${i + 1} ${JSON.stringify({name: `Baseline ${i + 1}`, team: t.team})}`).join('\n'));
    while (!outcome) {
      let progressed = false;
      for (let i = 0; i < 2 && !outcome; i++) {
        const side = `p${i + 1}`, player = projection.players[side], actor = actors[i];
        while (cursors[i] < player.protocol.length) actor.state.ingest(player.protocol[cursors[i]++]);
        if (!player.pending || !player.request || player.request.wait) continue;
        if (decisions.length >= maxDecisions) throw new Error('Incomplete game: decision cap reached');
        const request = structuredClone(player.request), view = structuredClone(actor.state), requestId = player.request_id;
        // Isolate policy inputs from the reducer, provenance, opponent and saved evidence.
        const t0 = performance.now(), choice = actor.choose(structuredClone(request), structuredClone(view));
        const latencyMs = performance.now() - t0;
        if (!legalChoices(request).includes(choice)) throw new Error('Policy chose an unavailable request action');
        player.pending = false; const before = projection.errors.length;
        await write(`>${side} ${choice}`);
        const errors = projection.errors.slice(before);
        if (errors.some(e => e.side !== side || !e.message.startsWith('[Unavailable choice]'))) throw new Error(`Invalid choice: ${JSON.stringify(errors)}`);
        if (errors.length && !player.pending) throw new Error('Unavailable choice without refreshed request');
        decisions.push({side, policy: policies[i], request_id: requestId, turn: view.turn, choice, latency_ms: latencyMs,
          actor_input: {request, public: view}, errors});
        progressed = true;
      }
      if (!progressed && !outcome) throw new Error('No pending player: stalled game');
    }
    if (outcome.winner && !['Baseline 1', 'Baseline 2'].includes(outcome.winner)) throw new Error('Unknown winner');
    return {teams: teams.map(t => ({seed: t.seed, team_key: t.team_key, partition: t.partition})), seed, policies, policySeeds,
      input, decisions, outcome: {winner: outcome.winner, turns: outcome.turns,
        score: [1, 2].map(i => !outcome.winner ? 0.5 : Number(outcome.winner === `Baseline ${i}`)), pokemon_left: outcome.score},
      spectator: projection.spectator};
  } finally { stream.battle?.destroy(); }
}
