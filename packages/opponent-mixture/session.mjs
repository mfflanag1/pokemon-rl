import {FullTrainingSession, validateEpisodeArgs} from '../full-training/session.mjs';
import {chooseOpponent} from '../showdown-panel/policy.mjs';
import {TacticalView} from '../showdown-panel/view.mjs';
import {ACTIONS} from '../full-showdown/actions.mjs';

export {SPLITS, officialPlan, validateEpisodeArgs, episodeStem, MAX_TEAM_ATTEMPTS} from '../full-training/session.mjs';

// F5 opponent mixture is an additive adapter over the unchanged F2 session,
// copied from the frozen F4 source rather than subclassing it: F4's own
// development-only admission stays intact in its package, untouched and
// unbypassed. The single behavioural difference is that the frozen
// `tactical-v1` advisor may be attached to one seat of a TRAIN episode as well
// as a development episode, so the collecting policy can be mixed with a
// scripted opponent during training. `final` is never admitted: F2 refuses it
// before the tactical seat is even considered, and the helper below refuses it
// independently. Episode admission, mechanics seed, team split/seed sampling
// and projected player frames are exactly F2; advice is reported beside the
// frame, never inside either player's projection, never submitted by the
// session itself.
const SIDES = ['p1', 'p2'];
export const TACTICAL_POLICY = 'tactical-v1';
export const TACTICAL_SIDES = Object.freeze(['p1', 'p2']);
export const TACTICAL_SPLITS = Object.freeze(['train', 'development']);
const START_OPTIONS = Object.freeze(['episodeIndex', 'runSeed', 'split', 'swap', 'tacticalSide']);

/** `tacticalSide` is null (no advisor) or one seat, admitted on train and on
 * development. Anything else - an unknown seat, a non-string, `final` or an
 * unknown split - is refused. The helper never consults the F4 validator, so
 * F4's train refusal is neither reused nor circumvented here. */
export function validateMixtureSide(tacticalSide, split) {
  if (tacticalSide === null || tacticalSide === undefined) return null;
  if (typeof tacticalSide !== 'string' || !TACTICAL_SIDES.includes(tacticalSide)) {
    throw new Error(`Unknown tactical side rejected: ${JSON.stringify(tacticalSide ?? null)}`);
  }
  if (typeof split !== 'string' || !TACTICAL_SPLITS.includes(split)) {
    throw new Error(`Tactical side is only admitted on train or development episodes, not ${JSON.stringify(split ?? null)}`);
  }
  return tacticalSide;
}

/** Strict admission of the full start option object: unknown keys are refused
 * (no plan, policy, curriculum, opponent request or snake_case alias may enter),
 * the four F2 coordinates are validated with F2 wording first, then the seat is
 * checked against the effective split. Returns the F2 episode options and the
 * admitted seat; nothing is sampled or constructed. */
export function validateMixtureStart(options) {
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('Expected start options object');
  const unexpected = Object.keys(options).filter(key => !START_OPTIONS.includes(key));
  if (unexpected.length) throw new Error(`Unsupported start option(s): ${unexpected.sort().join(', ')}`);
  const {tacticalSide = null, ...episode} = options;
  const split = episode.split === undefined ? 'train' : episode.split, swap = episode.swap === undefined ? false : episode.swap;
  validateEpisodeArgs(episode.episodeIndex, episode.runSeed, split, swap);
  return {episode, tacticalSide: validateMixtureSide(tacticalSide, split)};
}

export class OpponentMixtureSession extends FullTrainingSession {
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
    // Every refusal fires before anything below runs: no team is sampled and no
    // engine is created. F2 refusals keep F2 wording (`final`, unknown split,
    // nonboolean swap, bad coordinates); the seat is then checked for train or
    // development.
    const {episode, tacticalSide} = validateMixtureStart(options);
    this.tacticalSide = tacticalSide;
    this.tacticalView = tacticalSide ? new TacticalView() : null;
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
   * sees. Inputs are the seat's OWN private request and the view built from the
   * seat's OWN protocol only; the other seat's request is never read. There is
   * no fallback policy: an untranslatable or illegal choice is an error, not a
   * silently substituted action. */
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
