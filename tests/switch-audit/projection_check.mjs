import assert from 'node:assert/strict';
import {AuditSession} from '../../packages/battle-audit/session.mjs';
import {reconstruct,projectedAdvice} from '../../packages/switch-audit/projection.mjs';
import {requestActions} from '../../packages/full-showdown/actions.mjs';
const session=new AuditSession();
try {
 const f=await session.start({runSeed:1024261501,episodeIndex:0,split:'train',swap:false});
 const p=f.players.p1;let n=0;
 const test=(fn)=>{fn();n++;};
 test(()=>assert.deepEqual(requestActions(reconstruct(p).request).mask,p.mask));
 test(()=>{const before=JSON.stringify(p);projectedAdvice(p,session.auditViews.p1);assert.equal(JSON.stringify(p),before);});
 test(()=>{const q=structuredClone(p);q.opponent.find(m=>m?.active).stats={atk:999};assert.throws(()=>reconstruct(q));});
 test(()=>{const q=structuredClone(p);q.own[0].tera=true;assert.throws(()=>reconstruct(q));});
 test(()=>{const q={...p,battle:{sides:[]}};assert.throws(()=>reconstruct(q));});
 test(()=>{const q={...p,phase:'wait',mask:Array(14).fill(false)};assert.equal(projectedAdvice(q),null);assert.deepEqual(requestActions(reconstruct(q).request).mask,q.mask);});
 test(()=>{const q=structuredClone(p);q.phase='replace';q.active_moves=[];q.mask=q.mask.map((_,i)=>i>=8&&!q.own[i-8].active&&!q.own[i-8].condition.includes('fnt'));assert.ok(q.mask[projectedAdvice(q).projection]);});
 test(()=>{const q=structuredClone(p);q.mask[8]=true;q.own[0].active=true;assert.throws(()=>reconstruct(q));});
 test(()=>{const q=structuredClone(p);q.phase='move';q.turn=5;q.mask=Array(14).fill(false);q.mask[0]=q.mask[1]=true;
  const own=q.own.find(m=>m.active);own.species='Meowth';own.moves=['fakeout','quickattack'];own.tera=null;own.stats.atk=200;
  const foe=q.opponent.find(m=>m?.active);foe.species='Magikarp';foe.condition='100/100';foe.types=null;foe.tera=null;foe.ability=null;foe.item=null;
  q.active_moves=['fakeout','quickattack'].map(id=>({id,pp:10,maxpp:10,disabled:false,base_power:40,type:'Normal',category:'Physical'}));
  const {view}=reconstruct(q);view.active[q.side].enteredTurn=5;
  const advice=projectedAdvice(q,view);assert.equal(advice.projection,1);assert.equal(advice.entry,0);assert.equal(advice.entry_types_boosts,0);
 });
 console.log(`Passed ${n} projection checks`);
} finally {session.close();}
