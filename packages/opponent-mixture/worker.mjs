import readline from 'node:readline';
import {OpponentMixtureSession, SPLITS, validateMixtureSide} from './session.mjs';
import {describe} from '../reference-node/engine.mjs';

// Line-delimited JSON over stdio only: no sockets, no HTTP, no fetch. Replies
// carry projected frames plus the advised seat's tactical action index; engine
// objects and private opponent state never cross this boundary. The wire
// shape is the F4 worker's; the only difference is that `tactical_side` is
// admitted on train episodes as well as development ones (never on final).
const START_FIELDS = Object.freeze(['op', 'id', 'run_seed', 'episode_index', 'split', 'swap', 'tactical_side']);
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
      data = {schema_version: 'opponent-mixture-worker-v1', reference_commit: describe().commit};
    } else if (command.op === 'start') {
      // Episodes are fully determined by the F2 wire coordinates; the only extra
      // field is the optional advised seat. Plans, policies, opponent requests
      // and camelCase aliases are refused as unknown before anything starts.
      const unexpected = Object.keys(command).filter(key => !START_FIELDS.includes(key));
      if (unexpected.length) throw new Error(`Unsupported start field(s): ${unexpected.sort().join(', ')}`);
      const split = field(command, 'split', 'train'), swap = field(command, 'swap', false);
      const tacticalSide = field(command, 'tactical_side', null);
      if (split === 'final') throw new Error("Split 'final' is never admitted by full-training F2");
      if (typeof split !== 'string' || !SPLITS.includes(split)) {
        throw new Error(`Unknown split rejected: ${JSON.stringify(split ?? null)}`);
      }
      if (typeof swap !== 'boolean') throw new Error('Swap must be a boolean');
      validateMixtureSide(tacticalSide, split);
      session?.close(); session = new OpponentMixtureSession();
      data = await session.start({episodeIndex: command.episode_index, runSeed: command.run_seed, split, swap, tacticalSide});
    } else if (command.op === 'step' && session) {
      data = await session.step(command);
    } else if (command.op === 'cutoff' && session) {
      data = session.cutoff();
    } else if (command.op === 'close') {
      session?.close(); session = null; data = {closed: true};
    } else throw new Error('Unknown operation or missing session');
    process.stdout.write(`${JSON.stringify({id: command.id, ok: true, data})}\n`);
  } catch (error) {
    // Any failure (refused start, illegal step, untranslatable advice) tears
    // the session down so no half-configured episode survives the error.
    session?.close(); session = null;
    process.stdout.write(`${JSON.stringify({id: command?.id ?? null, ok: false, error: error.message})}\n`);
  }
}
session?.close();
