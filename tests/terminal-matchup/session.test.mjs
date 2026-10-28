import test from 'node:test';
import assert from 'node:assert/strict';
import {TerminalMatchupSession} from '../../packages/terminal-matchup/session.mjs';
import {FullLearningSession} from '../../packages/full-learning/session.mjs';

// Test the additive submission adapter without invoking a battle or private state.
function fixture(revival=false) {
  const s=new TerminalMatchupSession();
  const pokemon=Array.from({length:6},(_,i)=>({ident:`p1: member${i}`,active:i===0,
    condition:revival&&i===1?'0 fnt':'100/100',reviving:revival&&i===0}));
  s.projection.players.p1={pending:true,request_id:7,request:{side:{id:'p1',pokemon},
    ...(revival?{forceSwitch:[true]}:{active:[{moves:[{id:'splash',pp:40}]}]})}};
  s.projection.players.p2={pending:false};
  return s;
}
async function intercepted(s,fn) {
  const original=FullLearningSession.prototype.step;
  FullLearningSession.prototype.step=fn;
  try {await s.step({actions:{p1:9},request_ids:{p1:7}});}
  finally {FullLearningSession.prototype.step=original;s.close();}
}

test('ordinary switch hint is scoped to the submitted request',async()=>{
  const s=fixture();await intercepted(s,async function(){assert.equal(this.views.p1.switchHint,'member1');});
  assert.equal(s.views.p1.switchHint,null);
});

test('Revival Blessing target is not an entrant hint',async()=>{
  const s=fixture(true);await intercepted(s,async function(){assert.equal(this.views.p1.switchHint,null);});
});

test('rejected or cancelled step cannot retain an entry hint',async()=>{
  const s=fixture();await assert.rejects(intercepted(s,async()=>{throw new Error('synthetic rejection');}),/synthetic rejection/);
  assert.equal(s.views.p1.switchHint,null);
});

test('stale or malformed submission is rejected before hint mutation',async()=>{
  const s=fixture();try {
    await assert.rejects(s.step({actions:{p1:9},request_ids:{p1:6}}),/Stale/);
    await assert.rejects(s.step({actions:{p1:9,p2:1},request_ids:{p1:7,p2:0}}),/exactly/);
    assert.equal(s.views.p1.switchHint,null);
  }finally{s.close();}
});
