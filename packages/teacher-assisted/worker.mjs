import readline from 'node:readline';
import {pathToFileURL} from 'node:url';
import {ACTIONS} from '../full-showdown/actions.mjs';
import {createActor} from '../showdown-panel/policy.mjs';
import {SPLITS, TACTICAL_SIDES, StructuredSession} from '../structured-policy/session.mjs';
import {context, mark} from '../worker-watch/diagnostics.mjs';
import {describe} from '../reference-node/engine.mjs';

export class TeacherAssistedSession extends StructuredSession {
  constructor() {
    super();
    this.baselineSide = null;
    this.baselineActor = null;
    this.baselineCursor = 0;
  }

  async start(options = {}) {
    if (!options || typeof options !== 'object' || Array.isArray(options)) throw new Error('Expected start options object');
    const {baselineSide = null, ...inherited} = options;
    if (baselineSide !== null && !['p1', 'p2'].includes(baselineSide)) throw new Error('Unknown baseline side');
    if (baselineSide !== null && inherited.split !== 'development') throw new Error('Baseline is development-only');
    if (baselineSide !== null && inherited.tacticalSide != null) throw new Error('Only one external local policy is allowed');
    this.baselineSide = baselineSide;
    this.baselineActor = baselineSide ? createActor('baseline-v1') : null;
    this.baselineCursor = 0;
    return super.start(inherited);
  }

  drain() {
    super.drain();
    if (!this.baselineSide) return;
    const lines = this.projection.players[this.baselineSide].protocol;
    while (this.baselineCursor < lines.length) {
      const line = lines[this.baselineCursor++];
      if (line && !line.startsWith('|request|')) this.baselineActor.state.ingest(line);
    }
  }

  snapshot() {
    const frame = super.snapshot();
    const baseline_actions = {p1: -1, p2: -1};
    const side = this.baselineSide;
    if (side && !frame.outcome && frame.players[side].mask.some(Boolean)) {
      const player = this.projection.players[side];
      if (!player.pending || player.request?.side?.id !== side) throw new Error('Foreign or nonpending baseline request');
      const choice = this.baselineActor.choose(structuredClone(player.request), structuredClone(this.baselineActor.state));
      const action = ACTIONS.indexOf(choice);
      if (action < 0 || !frame.players[side].mask[action]) throw new Error('Invalid baseline choice');
      baseline_actions[side] = action;
    }
    return {...frame, baseline_actions};
  }

  cutoff() {
    return {...super.cutoff(), baseline_actions: {p1: -1, p2: -1}};
  }
}

async function main() {
const START_FIELDS = Object.freeze(['op', 'id', 'run_seed', 'episode_index', 'split', 'swap', 'tactical_side', 'baseline_side']);
const field = (command, key, fallback) =>
  Object.hasOwn(command, key) && command[key] !== undefined ? command[key] : fallback;

mark('worker_ready');
let session = null;
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  let command;
  context(null);
  try {
    if (line.length > 1_000_000) throw new Error('Oversized command');
    command = JSON.parse(line);
    context(command);
    if (command?.op === 'start') mark('start_received');
    if (!command || typeof command !== 'object' || Array.isArray(command)) throw new Error('Expected a command object');
    let data;
    if (command.op === 'hello') {
      data = {schema_version: 'opponent-mix-worker-v1', observation_extension: 'structured-public-history-v1', reference_commit: describe().commit};
    } else if (command.op === 'start') {
      const unexpected = Object.keys(command).filter(key => !START_FIELDS.includes(key));
      if (unexpected.length) throw new Error(`Unsupported start field(s): ${unexpected.sort().join(', ')}`);
      const split = field(command, 'split', 'train'), swap = field(command, 'swap', false);
      const tacticalSide = field(command, 'tactical_side', null);
      const baselineSide = field(command, 'baseline_side', null);
      if (split === 'final') throw new Error("Split 'final' is never admitted by full-training F2");
      if (typeof split !== 'string' || !SPLITS.includes(split)) throw new Error(`Unknown split rejected: ${JSON.stringify(split ?? null)}`);
      if (typeof swap !== 'boolean') throw new Error('Swap must be a boolean');
      if (tacticalSide !== null && (typeof tacticalSide !== 'string' || !TACTICAL_SIDES.includes(tacticalSide))) {
        throw new Error(`Unknown tactical side rejected: ${JSON.stringify(tacticalSide)}`);
      }
      session?.close(); session = new TeacherAssistedSession();
      data = await session.start({episodeIndex: command.episode_index, runSeed: command.run_seed, split, swap, tacticalSide, baselineSide});
    } else if (command.op === 'step' && session) {
      data = await session.step(command);
    } else if (command.op === 'cutoff' && session) {
      data = session.cutoff();
    } else if (command.op === 'close') {
      session?.close(); session = null; data = {closed: true};
    } else throw new Error('Unknown operation or missing session');
    process.stdout.write(`${JSON.stringify({id: command.id, ok: true, data})}\n`);
    if (command.op === 'start') mark('start_replied');
  } catch (error) {
    mark('command_error');
    session?.close(); session = null;
    process.stdout.write(`${JSON.stringify({id: command?.id ?? null, ok: false, error: error.message})}\n`);
  }
}
session?.close();

}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
