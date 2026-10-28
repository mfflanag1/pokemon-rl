import {FullShowdownSession} from '../full-showdown/session.mjs';
import {FullTrainingSession} from '../full-training/session.mjs';
import {scenarioPlan, CONTEXTS} from './scenarios.mjs';
export function hpFraction(p) {
    if (p.condition.includes('fnt')) return 0;
    const [hp, max] = p.condition.split(' ')[0].split('/').map(Number);
    if (!Number.isFinite(hp) || !Number.isFinite(max) || max <= 0) throw new Error('Invalid HP');
    return hp / max;
}
export function publicBaseline(player) {
  const legal = player.mask.flatMap((yes, i) => yes ? [i] : []);
  const moves = legal.filter(a => a < 4);
  return moves.length ? moves.reduce((best, a) => player.active_moves[a].base_power >
    player.active_moves[best].base_power ? a : best) : legal[0];
}
export class DecisionSession extends FullShowdownSession {
  snapshot() { return FullTrainingSession.prototype.snapshot.call(this); }
  async begin(options) {
    if (Object.keys(options).sort().join() !== 'context,episode,panel,run_seed') throw new Error('Unsupported F6 start field');
    this.split = 'train'; this.swap = false;
    this.learnerSide = CONTEXTS[options.context]?.side;
    const plan = scenarioPlan(options);
    const frame = await super.start({episodeIndex: options.episode, runSeed: options.run_seed, plan});
    this.decided = false;
    if (!Object.values(frame.players).every(p => p.phase === 'move' && p.mask.some(Boolean))) throw new Error('Invalid decision boundary');
    const player = frame.players[this.learnerSide];
    const revealed = player.opponent.find(p => p?.active)?.moves ?? [];
    const attack = CONTEXTS[options.context].threat === 'electric' ? 'thunderbolt' : 'surf';
    if (!revealed.includes(attack)) throw new Error('Threat not revealed to learner');
    this.before = frame;
    return frame;
  }
  async decide(action) {
    if (this.decided) throw new Error('Diagnostic already decided');
    this.decided = true;
    const side = this.learnerSide, foe = side === 'p1' ? 'p2' : 'p1';
    const before = this.before;
    const frame = await this.step({actions: {[side]: action, [foe]: publicBaseline(before.players[foe])},
      request_ids: Object.fromEntries(['p1', 'p2'].map(s => [s, before.players[s].request_id]))});
    if (frame.submitted.some(r => !r.accepted)) throw new Error('Unavailable diagnostic action');
    const exposed = frame.players[side].own.find(p => p.active);
    const prior = before.players[side].own.find(p => p.ident === exposed?.ident);
    if (!prior || hpFraction(prior) <= 0) throw new Error('Missing exposed-member prior HP');
    const reward = Math.max(0, Math.min(1, hpFraction(exposed) / hpFraction(prior)));
    return {frame, reward, exposed_ident: exposed.ident, diagnostic_ended: true,
      natural_terminal: frame.outcome?.kind === 'terminal'};
  }
  async continueBattle(maxTurns = 120) {
    if (!this.decided) throw new Error('Decide before continuation');
    let frame = this.snapshot(), decisions = 0;
    while (!frame.outcome) {
      if (Object.values(frame.players).every(p => p.phase === 'move') && frame.players.p1.turn >= maxTurns) {
        frame = this.cutoff(); break;
      }
      if (decisions > 10000) throw new Error('Continuation cap');
      const actions = {}, request_ids = {};
      for (const side of ['p1', 'p2']) if (frame.players[side].mask.some(Boolean)) {
        actions[side] = publicBaseline(frame.players[side]); request_ids[side] = frame.players[side].request_id; decisions++;
      }
      frame = await this.step({actions, request_ids});
    }
    return {frame, decisions};
  }
}
