import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from '../../packages/switch-screen/worker.mjs';
import {ROSTER} from '../../packages/switch-reference/profile.mjs';
import {mask} from '../../packages/switch-reference/actions.mjs';
import {hashJSON} from '../../packages/battle-contract/canonical.mjs';
import {replayRecord} from '../../packages/switch-screen/audit.mjs';
const game=id=>({id,teams:[ROSTER[0].id,ROSTER[8].id],seed:'sodium,'+'7'.repeat(64)});

test('Unfinished games cannot be discarded; stale request rejected before writes',async()=>{
 const w=new Worker();try{
  const f=await w.handle({op:'start',games:[game(0)]});
  await assert.rejects(w.handle({op:'start',games:[game(1)]}));
  const s=w.games[0].s,before=s.views();
  await assert.rejects(w.handle({op:'step',decisions:[{id:0,group:0,turn:1,requests:[0,0],actions:[0,0]}]}));
  assert.deepEqual(s.views(),before);assert.equal(f.games[0].truncated,false);
 }finally{for(const {s} of w.games)s.close();}
});
test('Reference replay rejects a changed action behind intact view fingerprints',async()=>{
 const w=new Worker();try{
  const spec=game(0);delete spec.id;let f=await w.handle({op:'start',games:[{id:0,...spec}]});const ds=[];
  for(let i=0;i<12;i++){
   const g=f.games[0],actions=g.views.map(v=>mask(v.request).findIndex((x,j)=>x&&j>=2));
   const d={group:g.group,turn:g.turn,requests:g.views.map(v=>v.request_id),view_lengths:g.views.map(v=>v.protocol.length),view_hashes:g.views.map(hashJSON),actions};
   ds.push(d);f=await w.handle({op:'step',decisions:[{id:0,...d}]});
  }
  const g=f.games[0],r={spec,pair:0,seat:0,decisions:ds,groups:g.group,winner:g.winner,truncated:g.truncated,final_views:g.views};
  await replayRecord(r);r.decisions[0].actions[0]=0;await assert.rejects(replayRecord(r));
 }finally{for(const {s} of w.games)s.close();}
});
test('Twelve switching turns are explicit nonterminal cutoffs',async()=>{
 const w=new Worker();try{
  let f=await w.handle({op:'start',games:[game(0)]});
  for(let i=0;i<12;i++){
   const g=f.games[0],actions=g.views.map(v=>mask(v.request).findIndex((x,j)=>x&&j>=2));
   f=await w.handle({op:'step',decisions:[{id:0,group:g.group,turn:g.turn,requests:g.views.map(v=>v.request_id),actions}]});
  }
  const g=f.games[0];assert.equal(g.truncated,true);assert.equal(g.phase,'joint_move');assert.equal(g.turn,13);assert.equal(g.winner,null);
  assert.equal((await w.handle({op:'finish'})).type,'finished');
 }finally{for(const {s} of w.games)s.close();}
});
