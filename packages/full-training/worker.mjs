import readline from 'node:readline';
import {FullTrainingSession, SPLITS} from './session.mjs';
import {describe} from '../reference-node/engine.mjs';

// Line-delimited JSON over stdio only: no sockets, no HTTP, no fetch. Replies
// carry projected frames, never engine objects or private opponent state.
const START_FIELDS = Object.freeze(['op', 'id', 'run_seed', 'episode_index', 'split', 'swap']);
const field = (command, key, fallback) =>
  Object.hasOwn(command, key) && command[key] !== undefined ? command[key] : fallback;

let session = null;
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  let command;
  try {
    if (line.length > 1_000_000) throw new Error('Oversized command');
    command = JSON.parse(line);
    if (!command || typeof command !== 'object' || Array.isArray(command)) throw new Error('Expected a command object');
    let data;
    if (command.op === 'hello') {
      data = {schema_version: 'full-training-worker-v1', reference_commit: describe().commit};
    } else if (command.op === 'start') {
      // Episodes are fully determined by the wire coordinates. There is no
      // curriculum provider and no plan field: both are refused as unknown.
      const unexpected = Object.keys(command).filter(key => !START_FIELDS.includes(key));
      if (unexpected.length) throw new Error(`Unsupported start field(s): ${unexpected.sort().join(', ')}`);
      const split = field(command, 'split', 'train'), swap = field(command, 'swap', false);
      if (split === 'final') throw new Error("Split 'final' is never admitted by full-training F2");
      if (typeof split !== 'string' || !SPLITS.includes(split)) {
        throw new Error(`Unknown split rejected: ${JSON.stringify(split ?? null)}`);
      }
      if (typeof swap !== 'boolean') throw new Error('Swap must be a boolean');
      session?.close(); session = new FullTrainingSession();
      data = await session.start({episodeIndex: command.episode_index, runSeed: command.run_seed, split, swap});
    } else if (command.op === 'step' && session) {
      data = await session.step(command);
    } else if (command.op === 'cutoff' && session) {
      data = session.cutoff();
    } else if (command.op === 'close') {
      session?.close(); session = null; data = {closed: true};
    } else throw new Error('Unknown operation or missing session');
    process.stdout.write(`${JSON.stringify({id: command.id, ok: true, data})}\n`);
  } catch (error) {
    session?.close(); session = null;
    process.stdout.write(`${JSON.stringify({id: command?.id ?? null, ok: false, error: error.message})}\n`);
  }
}
session?.close();
