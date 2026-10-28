import {FullTrainingSession} from '../full-training/session.mjs';
import {TacticalView} from '../showdown-panel/view.mjs';
import {chooseOpponent} from '../showdown-panel/policy.mjs';
import {ACTIONS} from '../full-showdown/actions.mjs';
import {describeDecision} from './describe.mjs';

// Coordinator diagnostics are deliberately outside player observations. Each
// view receives only its own projected public protocol, never engine state.
export class AuditSession extends FullTrainingSession {
  constructor() {
    super();
    this.auditViews = {p1: new TacticalView(), p2: new TacticalView()};
    this.auditCursors = {p1: 0, p2: 0};
  }
  drain() {
    super.drain();
    for (const side of ['p1', 'p2']) {
      const lines = this.projection.players[side].protocol;
      while (this.auditCursors[side] < lines.length) {
        const line = lines[this.auditCursors[side]++];
        if (line && !line.startsWith('|request|')) this.auditViews[side].ingest(line);
      }
    }
  }
  snapshot() {
    const frame = super.snapshot();
    const tactical_actions = {p1: -1, p2: -1};
    for (const side of ['p1', 'p2']) {
      if (!frame.outcome && frame.players[side].mask.some(Boolean)) {
        const request = this.projection.players[side].request;
        if (request.side?.id !== side) throw new Error('Foreign request');
        const choice = chooseOpponent('tactical-v1', structuredClone(request), structuredClone(this.auditViews[side]));
        const action = ACTIONS.indexOf(choice);
        if (action < 0 || !frame.players[side].mask[action]) throw new Error('Illegal advice');
        tactical_actions[side] = action;
      }
    }
    return {...frame, tactical_actions};
  }
  async step(command) {
    const before = this.snapshot();
    const diagnostics = {};
    for (const [side, action] of Object.entries(command.actions ?? {})) {
      if (!['p1', 'p2'].includes(side)) throw new Error('Unknown seat');
      diagnostics[side] = describeDecision(this.projection.players[side].request,
        this.auditViews[side], before.players[side], action);
    }
    return {...await super.step(command), diagnostics};
  }
}
