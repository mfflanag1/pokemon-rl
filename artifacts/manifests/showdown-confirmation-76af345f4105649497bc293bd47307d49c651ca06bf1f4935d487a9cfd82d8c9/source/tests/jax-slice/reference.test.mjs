import test from 'node:test';
import assert from 'node:assert/strict';
import {battleCase} from '../../packages/jax-slice/reference.mjs';
import {team,admit} from '../../packages/slice/reference.mjs';

test('both player channels hide pending commitments through terminal',async()=>{
  const c=await battleCase(17);let before=c.initial;
  assert.ok(c.turns.length>=2&&c.turns.length<=4);
  for(const t of c.turns){
    assert.deepEqual(t.pending,before);
    for(let side=0;side<2;side++){
      const v=t.views[side];assert.equal(v.request.side.id,`p${side+1}`);
      assert.ok(v.protocol.filter(l=>l.startsWith('|request|')).every(l=>JSON.parse(l.slice(9)).side.id===`p${side+1}`));
    }
    before=t.views;
  }
  assert.ok(before.every(v=>v.protocol.some(l=>l.startsWith('|win|'))));
});

test('fixed public profile cannot admit hidden set perturbations',()=>{
  for(const patch of [{item:'Leftovers'},{ability:'Guts'},{species:'Pikachu'},{teraType:'Ghost'},{moves:['Protect','Tackle']}]){
    const teams=[team(),team()];Object.assign(teams[1][0],patch);assert.throws(()=>admit(teams));
  }
});
