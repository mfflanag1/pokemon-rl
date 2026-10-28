import readline from 'node:readline';
import {AuditSession} from '../battle-audit/session.mjs';
import {projectedAdvice} from './projection.mjs';
class SwitchAuditSession extends AuditSession {
  snapshot(){
    const f=super.snapshot();
    if(!f.outcome) f.projection_audit=Object.fromEntries(['p1','p2'].map(s=>[s,projectedAdvice(f.players[s],this.auditViews[s])]));
    return f;
  }
}
let session = null;
for await (const line of readline.createInterface({input: process.stdin, crlfDelay: Infinity})) {
  let c;
  try {
    if (line.length > 1_000_000) throw new Error('Oversized command');
    c = JSON.parse(line);
    let data;
    if (c.op === 'hello') data = {schema_version: 'switch-audit-worker-v1'};
    else if (c.op === 'start') {
      if (Object.keys(c).some(k => !['op','id','run_seed','episode_index','split','swap'].includes(k))) throw new Error('Unsupported start field');
      session?.close(); session = new SwitchAuditSession();
      data = await session.start({runSeed:c.run_seed, episodeIndex:c.episode_index, split:c.split, swap:c.swap});
    } else if (c.op === 'step' && session) data = await session.step(c);
    else if (c.op === 'cutoff' && session) data = session.cutoff();
    else if (c.op === 'close') {session?.close(); session = null; data = {closed:true};}
    else throw new Error('Unknown operation or missing session');
    process.stdout.write(JSON.stringify({id:c.id,ok:true,data})+'\n');
  } catch (e) {
    session?.close(); session = null;
    process.stdout.write(JSON.stringify({id:c?.id ?? null,ok:false,error:e.message})+'\n');
  }
}
session?.close();
