import test from 'node:test';
import assert from 'node:assert/strict';
import {battleCase} from '../../packages/showdown-chance/reference.mjs';
import {wordCase,boundaryWords} from '../../packages/showdown-chance/word-oracle.mjs';

test('word map equals BigInt arithmetic at every bin edge',()=>{
  for(const w of boundaryWords()){
    const row=wordCase(w);assert.equal(row.bucket24,Number(BigInt(w)*24n/2n**32n));
    assert.equal(row.crit,w<178956971);assert.equal(row.roll,100-(w>>>28));
    assert.deepEqual(row.ties,[w>>>31,1-(w>>>31)]);
  }
  assert.equal(wordCase(178956970).crit,true);assert.equal(wordCase(178956971).crit,false);
});
test('raw-word instrumentation is passive and both pending views remain private',async()=>{
  for(const id of [0,1,17,42]){
    const actual=await battleCase(id),plain=await battleCase(id,{instrument:false});
    const stripped=structuredClone(actual);for(const t of stripped.turns)t.chance=[];
    assert.deepEqual(stripped,plain);
    let before=actual.initial,previousIndex=-1;
    for(const t of actual.turns){
      assert.deepEqual(t.pending,before);before=t.views;
      for(const c of t.chance){
        assert.ok(c.index>previousIndex);previousIndex=c.index;
        if(c.role==='crit')assert.equal(c.result,c.word<178956971);
        else if(c.role==='roll')assert.equal(c.result,c.word>>>28);
        else assert.equal(c.result,c.order[c.word>>>31]);
      }
      for(let a=0;a<2;a++)for(const v of [t.pending[a],t.views[a]]){
        assert.deepEqual(Object.keys(v).sort(),['side','request_id','turn','request','protocol'].sort());
        assert.equal(v.request.side.id,`p${a+1}`);
      }
    }
  }
});
