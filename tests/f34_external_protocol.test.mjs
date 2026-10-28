import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ExternalBenchmarkSession} from '../packages/external-benchmark/session.mjs';
import {TerminalMatchupSession} from '../packages/terminal-matchup/session.mjs';
const options = {episodeIndex:0,runSeed:1027341901,split:'development',swap:false};

test('external boundary preserves original player projections and only its channel',async()=>{
  const session=new ExternalBenchmarkSession('p2'),control=new TerminalMatchupSession();
  try {
    const frame=await session.start(options),expected=await control.start(options);
    assert.deepEqual(frame.players,expected.players);
    const source=session.projection.players.p2.protocol.slice();
    session.projection.players.p1.protocol.push('|request|PRIVATE_OTHER_PLAYER_SENTINEL');
    session.projection.ingest('end',JSON.stringify({teams:['PRIVATE_END_SENTINEL']}));
    const input=session.externalInput();
    assert.deepEqual(input.protocol,source);
    assert.equal(input.side,'p2');assert.equal(input.pending,true);
    assert.deepEqual(Object.keys(input).sort(),['schema_version','side','request_id','pending','protocol'].sort());
    assert.equal(JSON.stringify(input).includes('SENTINEL'),false);
    for(const line of input.protocol.filter(x=>x.startsWith('|request|')))
      assert.equal(JSON.parse(line.slice(9)).side.id,'p2');
    assert.deepEqual(session.externalInput().protocol,[]);
  } finally {session.close();control.close();}
});

test('foreign private request and unsplit channel are rejected',()=>{
  for(const line of ['|request|{"side":{"id":"p1"}}','|split|p1']) {
    const session=new ExternalBenchmarkSession('p2');
    try {session.projection.players.p2.protocol.push(line);assert.throws(()=>session.externalInput());}
    finally {session.close();}
  }
});

test('side selection is explicit',()=>assert.throws(()=>new ExternalBenchmarkSession('spectator')));

test('recorded engineering actions exactly replay authoritative mechanics',async()=>{
  const path='artifacts/reviews/f34-policy-reset/external-integration/timing-smoke-001/';
  const result=JSON.parse(readFileSync(path+'result.json'));
  const trace=readFileSync(path+'transcript.jsonl','utf8').trim().split('\n').map(JSON.parse);
  const session=new ExternalBenchmarkSession(result.external_side);
  try {
    let frame=await session.start({...options,tacticalSide:'p1'});
    for(let i=0;i<result.steps.length;i++) {
      assert.deepEqual(frame.players,trace[i].frame.players);
      assert.deepEqual(session.externalInput(),trace[i].external);
      const step=result.steps[i];
      frame=await session.step({actions:step.actions,request_ids:step.request_ids});
    }
    assert.deepEqual(frame.outcome,result.outcome);
    assert.deepEqual(frame.players,trace.at(-1).frame.players);
  } finally {session.close();}
});
