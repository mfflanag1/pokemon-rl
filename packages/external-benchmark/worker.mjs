import readline from 'node:readline';
import {ExternalBenchmarkSession} from './session.mjs';
import {describe} from '../reference-node/engine.mjs';

const fields = {hello: ['op','id'], start: ['op','id','run_seed','episode_index','swap','external_side','tactical_side'],
  step: ['op','id','actions','request_ids'], cutoff: ['op','id'], close: ['op','id']};
let session = null;
for await (const line of readline.createInterface({input: process.stdin, crlfDelay: Infinity})) {
  let command;
  try {
    if (line.length > 1_000_000) throw new Error('Oversized command');
    command = JSON.parse(line);
    if (!command || typeof command !== 'object' || Array.isArray(command) || !fields[command.op]) throw new Error('Unknown operation');
    if (Object.keys(command).some(k => !fields[command.op].includes(k))) throw new Error('Unsupported command field');
    let data;
    if (command.op === 'hello') data = {schema_version:'f34-external-worker-v1',reference_commit:describe().commit};
    else if (command.op === 'start') {
      session?.close(); session = new ExternalBenchmarkSession(command.external_side);
      const frame = await session.start({episodeIndex: command.episode_index, runSeed: command.run_seed,
        split:'development', swap: command.swap ?? false, tacticalSide: command.tactical_side ?? null});
      data = session.bundle(frame);
    } else if (command.op === 'step' && session) data = session.bundle(await session.step(command));
    else if (command.op === 'cutoff' && session) data = session.bundle(session.cutoff());
    else if (command.op === 'close') { session?.close(); session=null; data={closed:true}; }
    else throw new Error('Session missing');
    process.stdout.write(JSON.stringify({id:command.id,ok:true,data})+'\n');
  } catch(error) {
    session?.close(); session=null;
    process.stdout.write(JSON.stringify({id:command?.id??null,ok:false,error:error.message})+'\n');
  }
}
session?.close();
