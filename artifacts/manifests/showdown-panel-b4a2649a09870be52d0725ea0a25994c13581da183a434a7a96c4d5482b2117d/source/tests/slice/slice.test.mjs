import test from 'node:test';
import assert from 'node:assert/strict';
import {admit,team,battleCase,damageTable} from '../../packages/slice/reference.mjs';
test('restricted admission rejects items, abilities, Tera fields and extra team members',()=>{
  admit([team(),team()]);
  for(const edit of [t=>{t[0][0].item='Leftovers'},t=>{t[0][0].ability='Hustle'},t=>{t[0][0].terastallized='Normal'},t=>t[0].push(t[0][0])]){
    const t=[structuredClone(team()),structuredClone(team())];edit(t);assert.throws(()=>admit(t));
  }
});
test('restricted reference replays exactly, cannot Tera, and enumerates damage',async()=>{
  const a=await battleCase(77),b=await battleCase(77);assert.deepEqual(a,b);assert.ok(a.turns.length<=4);
  const rows=await damageTable();assert.equal(rows.length,32);assert.equal(Math.min(...rows.map(x=>x.damage)),33);assert.equal(Math.max(...rows.map(x=>x.damage)),58);
});
