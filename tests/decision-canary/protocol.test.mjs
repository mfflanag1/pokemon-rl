import test from 'node:test';
import assert from 'node:assert/strict';
import {DecisionSession, hpFraction} from '../../packages/decision-canary/session.mjs';
import {scenarioPlan} from '../../packages/decision-canary/scenarios.mjs';
const spec = {context: 0, panel: 'train', episode: 0, run_seed: 912};
test('only constructed train panels and exact coordinates are admitted', async () => {
  for (const bad of [{...spec,panel:'final'},{...spec,context:16},{...spec,run_seed:-1},{...spec,episode:1.5},
    {...spec,private_plan:{}},{...spec,context:'0'}]) {
    const s = new DecisionSession(); try {await assert.rejects(s.begin(bad));} finally{s.close();}
  }
  for (const panel of ['train','variant']) for (let context=0;context<16;context++) {
    const plan=scenarioPlan({...spec,panel,context}); assert.ok(plan.teams.every(t=>t.partition==='train'));
  }
});
test('the learner starts before choosing, with a revealed attack and unused Tera', async () => {
  for (let context=0;context<16;context++) {
    const s=new DecisionSession();try {
      const f=await s.begin({...spec,context});const p=f.players[s.learnerSide];
      assert.equal(p.phase,'move');assert.ok(p.mask[4]);assert.ok(p.own.every(m=>m.tera===null));
      assert.ok(p.opponent.find(m=>m?.active).moves.includes(context%8<4?'thunderbolt':'surf'));
      assert.equal(p.own.find(m=>m.active).species,context<8?'Blissey':'Gyarados');
      assert.equal(f.submissions,context<8?2:4);
    }finally{s.close();}
  }
});
test('switch target changes with threat and slot order; Tera has beneficial and harmful variants', async () => {
  for (const context of [0,2,4,6,8,12]) {
    const scores=[];
    for (const action of context<8?[9,10]:[0,4]) {
      const s=new DecisionSession();try {await s.begin({...spec,context});scores.push((await s.decide(action)).reward);}finally{s.close();}
    }
    assert.ok((context===0||context===6||context===12)?scores[0]>scores[1]:scores[1]>scores[0]);
  }
});
test('reward is independently recoverable from the exposed own member, not remaining team HP', async () => {
  const s=new DecisionSession();try {
    const f=await s.begin({...spec,context:12});const side=s.learnerSide;const r=await s.decide(9);
    const p=r.frame.players[side].own.find(m=>m.active), old=f.players[side].own.find(m=>m.ident===p.ident);
    assert.equal(r.reward,Math.min(1,hpFraction(p)/hpFraction(old)));
    assert.equal(r.diagnostic_ended,true);assert.equal(r.natural_terminal,false);
    await assert.rejects(s.decide(0),/already decided/);
  }finally{s.close();}
});
test('local diagnostic completion leaves a playable full battle', async () => {
  const s=new DecisionSession();try {
    await s.begin({...spec,context:8});const r=await s.decide(4);assert.equal(r.frame.outcome,null);
    const tail=await s.continueBattle();assert.ok(['terminal','administrative_cutoff'].includes(tail.frame.outcome.kind));
    assert.ok(tail.decisions>0);
  }finally{s.close();}
});
