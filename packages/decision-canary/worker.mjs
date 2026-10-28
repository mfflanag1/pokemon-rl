import readline from 'node:readline';
import {DecisionSession} from './session.mjs';
let session;
for await (const line of readline.createInterface({input: process.stdin, crlfDelay: Infinity})) {
  let c;
  try {
    if (line.length > 1000000) throw new Error('Oversized command');
    c = JSON.parse(line); let data;
    if (c.op === 'hello') data = {schema_version: 'decision-canary-worker-v1'};
    else if (c.op === 'start') {
      const {id, op, ...options} = c;
      session?.close(); session = new DecisionSession(); data = await session.begin(options);
    } else if (c.op === 'decide' && session) data = await session.decide(c.action);
    else if (c.op === 'continue' && session) data = await session.continueBattle();
    else throw new Error('Unsupported F6 operation');
    process.stdout.write(JSON.stringify({id: c.id, ok: true, data}) + '\n');
  } catch (e) {
    session?.close(); session = null;
    process.stdout.write(JSON.stringify({id: c?.id ?? null, ok: false, error: e.message}) + '\n');
  }
}
session?.close();
