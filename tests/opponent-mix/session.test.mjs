import test from 'node:test';
import assert from 'node:assert/strict';
import {OpponentMixSession} from '../../packages/opponent-mix/session.mjs';
import {TerminalMatchupSession} from '../../packages/terminal-matchup/session.mjs';

for (const split of ['train','development']) {
  test(`advisor preserves own-player projections on ${split}`, async () => {
    const a=new OpponentMixSession(), b=new TerminalMatchupSession();
    try {
      let x=await a.start({episodeIndex:3,runSeed:1025262699,split,tacticalSide:'p2'});
      let y=await b.start({episodeIndex:3,runSeed:1025262699,split,...(split==='development'?{tacticalSide:'p2'}:{})});
      for (let i=0;i<8 && !x.outcome;i++) {
        assert.deepEqual(x.players,y.players); assert.deepEqual(x.provenance,y.provenance);
        if(split==='development')assert.deepEqual(x.tactical_actions,y.tactical_actions);
        assert.equal(x.schema_version,'opponent-mix-frame-v1');
        const actions={},request_ids={};
        for(const side of ['p1','p2']) {
          const p=x.players[side];if(!p.mask.some(Boolean))continue;
          const action=side==='p2'?x.tactical_actions[side]:p.mask.indexOf(true);
          assert.ok(p.mask[action]);actions[side]=action;request_ids[side]=p.request_id;
          assert.equal('tactical_actions' in p,false);
        }
        x=await a.step({actions,request_ids});y=await b.step({actions,request_ids});
      }
    }finally{a.close();b.close();}
  });
}
test('new advisor refuses final, injected plans, invalid seat and coordinates', async()=>{
  for(const options of [{split:'final'},{tacticalSide:'p3'},{plan:{}},{runSeed:-1},{swap:'false'}]){
    const s=new OpponentMixSession();try{
      await assert.rejects(s.start({episodeIndex:0,runSeed:1,...options}));assert.equal(s.started,false);
    }finally{s.close();}
  }
});
test('sealed predecessor continues to refuse train tactical advice',async()=>{
  const s=new TerminalMatchupSession();try{await assert.rejects(s.start({episodeIndex:0,runSeed:1,split:'train',tacticalSide:'p2'}));}finally{s.close();}
});
