import test from 'node:test';
import assert from 'node:assert/strict';
import {Session,choices,audit} from '../../packages/switch-reference/session.mjs';
import {ROSTER,reference} from '../../packages/switch-reference/profile.mjs';
import {mask,command} from '../../packages/switch-reference/actions.mjs';
import {seedFor} from '../../packages/reference-node/engine.mjs';
import {fixture} from '../../packages/switch-reference/fixtures.mjs';
import {damageGrid} from '../../packages/switch-reference/damage.mjs';
const spec=(key='a')=>({teams:[ROSTER[0].id,ROSTER[8].id],seed:seedFor(`switch-test/${key}`)});

test('profile has 24 closed teams and a non-debug private-HP format',()=>{
  assert.equal(ROSTER.length,24);const r=reference();assert.ok(r.format.rules.some(([r])=>r==='hppercentagemod'));
});
test('stable switch identity survives current-request reordering',async()=>{
  const s=new Session();try{
    let f=await s.start(spec());assert.deepEqual(mask(f.views[0].request),[true,true,false,true]);
    f=await s.step(choices(f,[3,1]));assert.equal(f.views[0].request.side.pokemon[0].ident,'p1: B');
    assert.deepEqual(mask(f.views[0].request),[true,true,true,false]);assert.equal(command(f.views[0].request,2),'switch 2');
    assert.ok(f.views[1].protocol.some(l=>l==='|switch|p1a: B|Squirtle, L50, M|100/100'));
  }finally{s.close();}
});
test('stale, duplicate, extra, invalid and concurrent submissions cannot double commit',async()=>{
  const s=new Session();try{
    const f=await s.start(spec('reject')),before=audit(s),d=choices(f,[0,0]);
    for(const bad of [[{...d[0],request_id:99},d[1]],[d[0],d[0]],[{...d[0],action:true},d[1]],
      [{...d[0],seed:'hidden'},d[1]],[{...d[0],view_hash:'bad'},d[1]],[{...d[0],action:2},d[1]]]){
      await assert.rejects(s.step(bad));assert.deepEqual(audit(s),before);assert.deepEqual(s.frame(),f);
    }
    const pending=s.step(d);await assert.rejects(s.step(d));await pending;
    assert.equal(s.tape.length,2);await assert.rejects(s.step(d));assert.equal(s.tape.length,2);
  }finally{s.close();}
});
test('out-of-profile startup and action phases fail closed',async()=>{
  const s=new Session();try{await assert.rejects(s.start({...spec(),teams:[ROSTER[0].id,'unknown']}));assert.ok(!s.battle);
    const f=await s.start(spec('scope'));const req=f.views[0].request;
    for(const mutate of [r=>r.teamPreview=true,r=>r.active[0].canTerastallize='Grass',r=>r.active[0].trapped=true,
      r=>r.active[0].moves[0].id='struggle',r=>r.active.push(r.active[0]),r=>r.side.pokemon[1].reviving=true]){
      const changed=structuredClone(req);mutate(changed);assert.throws(()=>mask(changed));
    }
    assert.deepEqual(mask({wait:true}),[false,false,false,false]);assert.throws(()=>command({wait:true},0));
  }finally{s.close();}
});
test('natural faint replacement is separate from a turn and waiting side has no action',async()=>{
  const s=new Session();try{
    let f=await s.start(spec('forced'));
    for(let i=0;i<12&&f.phase==='joint_move';i++)f=await s.step(choices(f,[0,0]));
    assert.equal(f.phase,'replacement');assert.equal(f.requested.length,1);
    const seat=f.requested[0],turn=f.turn;assert.deepEqual(mask(f.views[1-seat].request),[false,false,false,false]);
    const action=mask(f.views[seat].request).findIndex(Boolean);assert.ok(action>=2);
    const next=await s.step(choices(f,seat===0?[action,null]:[null,action]));assert.equal(next.phase,'joint_move');
    assert.ok(next.turn===turn||next.turn===turn+1);assert.equal(next.group,f.group+1);
  }finally{s.close();}
});
test('switch loops truncate without creating a terminal outcome, order replay agrees',async()=>{
  const a=await fixture(spec('cutoff'),'switch',0),b=await fixture(spec('cutoff'),'switch',1);
  assert.deepEqual(a.snapshots,b.snapshots);assert.deepEqual(a.outcome,{status:'truncated',winner:null,turn:13});
});
test('conditional damage oracle covers both sides and the exact ability threshold',async()=>{
  const rows=await damageGrid();assert.equal(rows.length,18);assert.equal(rows.reduce((n,r)=>n+r.calls,0),3456);
});
