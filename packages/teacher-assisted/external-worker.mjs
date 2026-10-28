import readline from 'node:readline';
import {pathToFileURL} from 'node:url';
import {ExternalBenchmarkSession} from '../external-benchmark/session.mjs';
import {describe} from '../reference-node/engine.mjs';
import {choiceFor} from '../full-showdown/actions.mjs';
import {validateEpisodeArgs} from '../full-training/session.mjs';

const fields = {hello: ['op', 'id'], start: ['op', 'id', 'run_seed', 'episode_index', 'swap', 'split', 'external_side', 'tactical_side'],
  step: ['op', 'id', 'actions', 'request_ids'], cutoff: ['op', 'id'], close: ['op', 'id']};

export function validateCommand(command) {
  if (!command || typeof command !== 'object' || Array.isArray(command) || !fields[command.op]) throw new Error('Unknown operation');
  if (Object.keys(command).some(k => !fields[command.op].includes(k))) throw new Error('Unsupported command field');
  if (command.op === 'start') {
    validateEpisodeArgs(command.episode_index, command.run_seed, command.split, command.swap);
    if (!['p1', 'p2'].includes(command.external_side)) throw new Error('External side must be p1 or p2');
    if (command.tactical_side !== null && !['p1', 'p2'].includes(command.tactical_side)) throw new Error('Explicit tactical side or null required');
    if (command.tactical_side === command.external_side) throw new Error('Teacher cannot also be tactical');
  }
  return command;
}

export async function submittedStep(session, command) {
  const submitted = Object.fromEntries(Object.entries(command.actions ?? {}).map(([side, action]) => {
    const player = session.projection.players[side];
    if (!player) throw new Error('Unknown submitting side');
    return [side, {request_id: command.request_ids?.[side], command: choiceFor(action, player.request)}];
  }));
  const frame = await session.step(command);
  return {...frame, submitted: frame.submitted.map(row => ({...row, ...submitted[row.side]}))};
}

export async function main() {
  let session = null;
  for await (const line of readline.createInterface({input: process.stdin, crlfDelay: Infinity})) {
    let command;
    try {
      if (line.length > 1_000_000) throw new Error('Oversized command');
      command = validateCommand(JSON.parse(line));
      let data;
      if (command.op === 'hello') data = {schema_version: 'f35-external-worker-v1', reference_commit: describe().commit, splits: ['train', 'development']};
      else if (command.op === 'start') {
        if (session) throw new Error('Close the current session before starting another');
        session = new ExternalBenchmarkSession(command.external_side);
        data = session.bundle(await session.start({episodeIndex: command.episode_index, runSeed: command.run_seed,
          split: command.split, swap: command.swap, tacticalSide: command.tactical_side}));
      } else if (command.op === 'step' && session) data = session.bundle(await submittedStep(session, command));
      else if (command.op === 'cutoff' && session) data = session.bundle(session.cutoff());
      else if (command.op === 'close') { session?.close(); session = null; data = {closed: true}; }
      else throw new Error('Session missing');
      process.stdout.write(JSON.stringify({id: command.id, ok: true, data}) + '\n');
    } catch (error) {
      session?.close(); session = null;
      process.stdout.write(JSON.stringify({id: command?.id ?? null, ok: false, error: error.message}) + '\n');
    }
  }
  session?.close();
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
