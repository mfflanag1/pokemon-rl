import {Dex} from '../reference-node/engine.mjs';
import {summarizeMatrix} from './statistics.mjs';

export function counters(policyIds) {
  return Object.fromEntries(policyIds.map(id => [id, {decisions: 0, damaging_moves: 0, status_moves: 0,
    forced_switches: 0, voluntary_switches: 0, tera: 0, unavailable_retries: 0, latencies: []}]));
}
export function countGame(totals, game) {
  for (const d of game.decisions) {
    const c = totals[d.policy]; if (!c) throw new Error('Unknown decision policy');
    if (!Number.isFinite(d.latency_ms) || d.latency_ms < 0) throw new Error('Invalid recorded latency');
    c.decisions++; c.latencies.push(d.latency_ms); c.unavailable_retries += d.errors.length;
    if (d.choice.startsWith('switch ')) c[d.actor_input.request.forceSwitch ? 'forced_switches' : 'voluntary_switches']++;
    if (d.choice.startsWith('move ')) {
      const move = d.actor_input.request.active[0].moves[Number(d.choice.split(' ')[1]) - 1];
      c[Dex.moves.get(move.id ?? move.move).category === 'Status' ? 'status_moves' : 'damaging_moves']++;
    }
    c.tera += Number(d.choice.endsWith(' terastallize'));
  }
}
export function report(plan, records, totals) {
  const byPolicy = Object.fromEntries(Object.entries(totals).map(([id, {latencies, ...c}]) => {
    const xs = [...latencies].sort((a, b) => a - b);
    return [id, {...c, latency_ms: {median: xs[Math.floor(xs.length * 0.5)], p95: xs[Math.floor(xs.length * 0.95)],
      p99: xs[Math.floor(xs.length * 0.99)], max: xs.at(-1)}}];
  }));
  return {schema: 'showdown-opponent-panel-report-1', status: 'complete', namespace: plan.namespace,
    games: records.length, pairs_per_matchup: plan.pairs, matrix: summarizeMatrix(records, {policyIds: plan.policy_ids, pairs: plan.pairs,
      bootstrapSamples: plan.bootstrap_samples, seed: plan.bootstrap_seed}), by_policy: byPolicy, records,
    decisions: Object.values(byPolicy).reduce((n, c) => n + c.decisions, 0),
    interpretation: 'Fixed development screen, descriptive paired intervals; no promotion, Elo, full-format learner or final-evaluation claim.'};
}
