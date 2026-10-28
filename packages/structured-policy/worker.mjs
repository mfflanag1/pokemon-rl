import readline from 'node:readline';
import {SPLITS, TACTICAL_SIDES, StructuredSession} from './session.mjs';
import {context, mark} from '../worker-watch/diagnostics.mjs';
import {describe} from '../reference-node/engine.mjs';

const START_FIELDS = Object.freeze(['op', 'id', 'run_seed', 'episode_index', 'split', 'swap', 'tactical_side']);
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
      if (split === 'final') throw new Error("Split 'final' is never admitted by full-training F2");
      if (typeof split !== 'string' || !SPLITS.includes(split)) throw new Error(`Unknown split rejected: ${JSON.stringify(split ?? null)}`);
      if (typeof swap !== 'boolean') throw new Error('Swap must be a boolean');
      if (tacticalSide !== null && (typeof tacticalSide !== 'string' || !TACTICAL_SIDES.includes(tacticalSide))) {
        throw new Error(`Unknown tactical side rejected: ${JSON.stringify(tacticalSide)}`);
      }
      session?.close(); session = new StructuredSession();
      data = await session.start({episodeIndex: command.episode_index, runSeed: command.run_seed, split, swap, tacticalSide});
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
