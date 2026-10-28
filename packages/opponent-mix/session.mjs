import {TerminalMatchupSession} from '../terminal-matchup/session.mjs';
import {FullTrainingSession, validateEpisodeArgs} from '../full-training/session.mjs';
import {TacticalView} from '../showdown-panel/view.mjs';
export {SPLITS, TACTICAL_SIDES} from '../full-learning/session.mjs';

// F26 admits the same frozen advisor on training teams. Its advice remains
// coordinator-only; projected players and the F25 identity reducer are unchanged.
export class OpponentMixSession extends TerminalMatchupSession {
  async start(options = {}) {
    if (this.started) throw new Error('Session already started');
    if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('Expected start options object');
    const fields = ['episodeIndex', 'runSeed', 'split', 'swap', 'tacticalSide'];
    if (Object.keys(options).some(key => !fields.includes(key))) throw new Error('Unsupported start option');
    const {tacticalSide = null, ...episode} = options;
    const split = episode.split === undefined ? 'train' : episode.split;
    const swap = episode.swap === undefined ? false : episode.swap;
    validateEpisodeArgs(episode.episodeIndex, episode.runSeed, split, swap);
    if (tacticalSide !== null && !['p1', 'p2'].includes(tacticalSide)) throw new Error('Unknown tactical side');
    this.tacticalSide = tacticalSide;
    this.tacticalView = tacticalSide ? new TacticalView() : null;
    this.tacticalCursor = 0;
    return FullTrainingSession.prototype.start.call(this, episode);
  }

  snapshot() {
    return {...super.snapshot(), schema_version: 'opponent-mix-frame-v1'};
  }
}
