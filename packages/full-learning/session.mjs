import {FullTrainingSession, validateEpisodeArgs} from '../full-training/session.mjs';
import {chooseOpponent} from '../showdown-panel/policy.mjs';
import {TacticalView} from '../showdown-panel/view.mjs';
import {ACTIONS} from '../full-showdown/actions.mjs';

export {SPLITS, officialPlan, validateEpisodeArgs, episodeStem, MAX_TEAM_ATTEMPTS} from '../full-training/session.mjs';

// F4 is an additive adapter over the unchanged F2 session: identical episode
// admission, identical mechanics seed, identical projected player frames. The
// only addition is a frozen `tactical-v1` advisor attached to at most one seat
// of a development episode. Its advice is reported beside the frame, never
// inside either player's projection, and never submitted on anyone's behalf.
const SIDES = ['p1', 'p2'];
export const TACTICAL_POLICY = 'tactical-v1';
export const TACTICAL_SIDES = Object.freeze(['p1', 'p2']);
const START_OPTIONS = Object.freeze(['episodeIndex', 'runSeed', 'split', 'swap', 'tacticalSide']);

/** `tacticalSide` is null (no advisor) or one seat, and only on development.
 * Train episodes never carry tactical advice; `final` is already refused by F2. */
export function validateTacticalSide(tacticalSide, split) {
  if (tacticalSide === null || tacticalSide === undefined) return null;
  if (typeof tacticalSide !== 'string' || !TACTICAL_SIDES.includes(tacticalSide)) {
    throw new Error(`Unknown tactical side rejected: ${JSON.stringify(tacticalSide ?? null)}`);
  }
  if (split !== 'development') throw new Error(`Tactical side is only admitted on development episodes, not ${JSON.stringify(split ?? null)}`);
  return tacticalSide;
}

export class FullLearningSession extends FullTrainingSession {
  constructor() {
    super();
    this.tacticalSide = null;
    // Separate public-event view and cursor for the advised seat. It is fed
    // exclusively from that seat's own projected player protocol, never from the
    // spectator channel, the engine or the other seat's private request.
    this.tacticalView = null; this.tacticalCursor = 0;
  }

  async start(options = {}) {
    if (this.started) throw new Error('Session already started');
    if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('Expected start options object');
    const unexpected = Object.keys(options).filter(key => !START_OPTIONS.includes(key));
    if (unexpected.length) throw new Error(`Unsupported start option(s): ${unexpected.sort().join(', ')}`);
    const {tacticalSide = null, ...episode} = options;
    // F2 coordinates are validated first so their refusals keep F2 wording; the
    // tactical seat is then checked against the effective split. Nothing below
    // runs before every refusal: no team is sampled and no engine is created.
    const split = episode.split === undefined ? 'train' : episode.split, swap = episode.swap === undefined ? false : episode.swap;
    validateEpisodeArgs(episode.episodeIndex, episode.runSeed, split, swap);
    const admittedSide = validateTacticalSide(tacticalSide, split);
    this.tacticalSide = admittedSide;
    this.tacticalView = admittedSide ? new TacticalView() : null;
    this.tacticalCursor = 0;
    // Exact F2 mechanics: the parent receives only its own four coordinates.
    return super.start(episode);
  }

  drain() {
    super.drain();
    if (!this.tacticalSide) return;
    const lines = this.projection.players[this.tacticalSide].protocol;
    while (this.tacticalCursor < lines.length) {
      const line = lines[this.tacticalCursor++];
      if (line && !line.startsWith('|request|')) this.tacticalView.ingest(line);
    }
  }

  /** Frozen policy choice for the advised seat, translated through the shared
   * ACTIONS codec and required to be legal under the projected mask the actor
   * sees. There is no fallback policy: an untranslatable or illegal choice is
   * an error, not a silently substituted action. */
  tacticalAction(side, mask) {
    const p = this.projection.players[side];
    if (!p.pending || !p.request || p.request.wait) return -1;
    if (p.request.side?.id !== side) throw new Error('Tactical advisor received a foreign private request');
    const choice = chooseOpponent(TACTICAL_POLICY, structuredClone(p.request), structuredClone(this.tacticalView));
    if (choice === null || choice === 'pass') return -1;
    const index = ACTIONS.indexOf(choice);
    if (index < 0 || !mask[index]) {
      throw new Error(`Tactical choice ${JSON.stringify(choice)} is not legal under the current projected mask`);
    }
    return index;
  }

  snapshot() {
    const frame = super.snapshot();
    const tactical_actions = Object.fromEntries(SIDES.map(side => [side, -1]));
    const side = this.tacticalSide;
    if (side && !frame.outcome && frame.players[side].mask.some(Boolean)) {
      tactical_actions[side] = this.tacticalAction(side, frame.players[side].mask);
    }
    // Same schema, same players, same provenance: advice sits beside the frame.
    return {...frame, tactical_actions};
  }

  cutoff() {
    // F2 keeps the live player observations for value bootstrapping. Advice is
    // coordinator control and must be cleared after the administrative outcome.
    return {...super.cutoff(), tactical_actions: {p1: -1, p2: -1}};
  }
}
