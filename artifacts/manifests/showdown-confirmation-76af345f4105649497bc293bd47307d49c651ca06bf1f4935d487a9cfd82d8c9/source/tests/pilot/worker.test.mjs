import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from '../../packages/pilot/worker.mjs';
import {seedFor} from '../../packages/reference-node/engine.mjs';

test('pilot worker enforces request identity before committing actions',async()=>{
  const w=new Worker();
  try {
    const f=await w.handle({op:'start',games:[{id:0,seed:seedFor('pilot-worker-test-1')}]});
    const decision={id:0,turn:1,requests:f.games[0].views.map(v=>v.request_id),actions:[1,1]};
    await assert.rejects(w.handle({op:'step',decisions:[{...decision,requests:[999,999]}]}));
    assert.deepEqual(w.frame(),f);
    await assert.rejects(w.handle({op:'step',decisions:[{...decision,actions:[true,1]}]}));
    assert.deepEqual(w.frame(),f);
    await w.handle({op:'step',decisions:[decision]});
  } finally {for(const s of w.games)s.destroy();}
});

test('pilot worker cannot discard live games or admit malformed batches',async()=>{
  const w=new Worker();
  try {
    await assert.rejects(w.handle({op:'start',games:[{id:0,seed:'bad'}]}));
    await assert.rejects(w.handle({op:'start',games:[{id:0,seed:seedFor('x')},{id:0,seed:seedFor('y')}]}));
    await w.handle({op:'start',games:[{id:0,seed:seedFor('pilot-worker-test-2')}]});
    await assert.rejects(w.handle({op:'start',games:[{id:1,seed:seedFor('z')}]}));
    await assert.rejects(w.handle({op:'finish'}));
  } finally {for(const s of w.games)s.destroy();}
});

test('both player channels reach a public terminal outcome without oracle internals',async()=>{
  const w=new Worker();
  try {
    let f=await w.handle({op:'start',games:[0,1].map(id=>({id,seed:seedFor(`pilot-worker-test-3/${id}`)}))});
    for(let t=0;t<4 && f.games.some(g=>!g.ended);t++) {
      f=await w.handle({op:'step',decisions:f.games.filter(g=>!g.ended).map(g=>({id:g.id,turn:g.views[0].turn,
        requests:g.views.map(v=>v.request_id),actions:[t%2,1]}))});
    }
    for(const g of f.games) {
      assert.ok(g.ended);assert.ok([0,1].includes(g.winner));
      assert.deepEqual(Object.keys(g).sort(),['ended','id','views','winner']);
      for(const v of g.views)assert.ok(v.protocol.includes(`|win|P${g.winner+1}`));
    }
    assert.equal((await w.handle({op:'finish'})).type,'finished');
  } finally {for(const s of w.games)s.destroy();}
});
