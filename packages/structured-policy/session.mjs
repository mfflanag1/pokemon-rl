import {WatchedSession} from '../worker-watch/session.mjs';
import {PublicHistory} from './history.mjs';
export {SPLITS, TACTICAL_SIDES} from '../opponent-mix/session.mjs';

/** Preserve frozen v2 players and mechanics; histories are separately projected. */
export class StructuredSession extends WatchedSession {
  constructor() {
    super();
    this.histories = Object.fromEntries(['p1', 'p2'].map(side => [side, new PublicHistory(side)]));
  }

  snapshot() {
    const frame = super.snapshot();
    return {...frame, observable_history: Object.fromEntries(['p1', 'p2'].map(side =>
      [side, this.histories[side].consume(this.projection.players[side].protocol)]))};
  }
}
