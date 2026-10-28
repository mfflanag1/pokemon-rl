import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import readline from 'node:readline';

function client(path) {
  const process = spawn('node', [path], {stdio: ['pipe','pipe','pipe']});
  const lines = readline.createInterface({input:process.stdout})[Symbol.asyncIterator]();
  let stderr='',id=0;
  process.stderr.on('data', chunk => { stderr+=chunk.toString(); });
  return {async send(command) {
    process.stdin.write(`${JSON.stringify({...command,id:++id})}\n`);
    let timer;
    try {
      const row = await Promise.race([lines.next(),new Promise((_,reject) => {
        timer=setTimeout(()=>reject(new Error('Test worker deadline')),10000);
      })]);
      assert.equal(row.done,false);
      const response=JSON.parse(row.value);assert.equal(response.id,id);return response;
    } finally {clearTimeout(timer);}
  },async close() {
    const exited=new Promise(resolve=>process.once('close',resolve));
    process.stdin.end();process.kill();await exited;
  },diagnostics:()=>stderr.trim().split('\n').filter(Boolean).map(line=>JSON.parse(line))};
}
const start={op:'start',run_seed:1025263001,episode_index:0,split:'development',swap:false,tactical_side:null};

test('watched worker keeps hello/start/step/cutoff/error wire results identical', async()=> {
  const old=client('packages/opponent-mix/worker.mjs'),watch=client('packages/worker-watch/worker.mjs');
  try {
    const same=async command=> {
      const [a,b]=await Promise.all([old.send(command),watch.send(command)]);assert.deepEqual(b,a);return a;
    };
    await same({op:'hello'});
    let frame=(await same(start)).data;
    for(let turn=0;turn<5 && !frame.outcome;turn++) {
      const actions={},request_ids={};
      for(const side of ['p1','p2']) {
        const p=frame.players[side];if(!p.mask.some(Boolean))continue;
        actions[side]=p.mask.findIndex(Boolean);request_ids[side]=p.request_id;
      }
      frame=(await same({op:'step',actions,request_ids})).data;
    }
    if(!frame.outcome && ['p1','p2'].every(s=>frame.players[s].phase==='move' && frame.players[s].mask.some(Boolean)))await same({op:'cutoff'});
    await same({...start,swap:true,tactical_side:'p1'});
    await same({...start,split:'final'});
    await same({...start,extra:'forbidden'});
    await same({op:'close'});
  } finally {await Promise.all([old.close(),watch.close()]);}
  const records=watch.diagnostics();assert.ok(records.length>0);
  for(const r of records) {
    assert.equal(r.schema_version,'worker-watch-v1');assert.ok(r.pid>0 && r.rss_bytes>0);
    assert.ok(Number.isFinite(r.wall_ms) && Number.isFinite(r.monotonic_ms));
    assert.deepEqual(Object.keys(r).sort(),['schema_version','stage','pid','request_id','op','wall_ms','monotonic_ms','rss_bytes','dropped_records'].sort());
  }
  for(const stage of ['worker_ready','start_received','session_start_begin','team_admission_begin','team_admission_done','stream_write_begin','stream_write_done','session_start_done','start_replied','session_close_begin','session_close_done','command_error'])assert.ok(records.some(r=>r.stage===stage),stage);
  assert.equal(old.diagnostics().length,0);
});

async function diagnosticProbe(code) {
  const child=spawn('node',['--input-type=module','-e',code],{stdio:['ignore','pipe','pipe']});
  // Intentionally do not drain stderr.
  let timer;
  try {
    const output=await Promise.race([new Promise(resolve=>child.stdout.once('data',chunk=>resolve(chunk.toString()))),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Blocked by diagnostics')),5000);})]);
    assert.equal(output,'ok\n');
  } finally {clearTimeout(timer);child.kill();await new Promise(resolve=>child.once('close',resolve));}
}
test('bounded diagnostic queue does not block an undrained pipe',async()=> {
  await diagnosticProbe("import {mark} from './packages/worker-watch/diagnostics.mjs';for(let i=0;i<100000;i++)mark('probe');process.stdout.write('ok\\n');setTimeout(()=>{},10000);");
});
test('closed diagnostic fd does not prevent a protocol reply',async()=> {
  await diagnosticProbe("import {closeSync} from 'node:fs';import {mark} from './packages/worker-watch/diagnostics.mjs';closeSync(2);mark('probe');process.stdout.write('ok\\n');setTimeout(()=>{},10000);");
});
