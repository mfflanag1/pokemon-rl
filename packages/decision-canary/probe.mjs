import {writeFileSync} from 'node:fs';
import {DecisionSession} from './session.mjs';
import {CONTEXTS} from './scenarios.mjs';
const records=[];
for (const panel of ['train','variant']) for (let episode=0;episode<16;episode++) for (let context=0;context<16;context++) {
  const spec={panel,context,episode,run_seed:8622027};
  const s=new DecisionSession();let before;
  try {before=await s.begin(spec);}finally{s.close();}
  const side=CONTEXTS[context].side, player=before.players[side];
  const target=context<8?8+player.own.findIndex(m=>m.species===(context%8<4?'Dugtrio':'Vaporeon')):context%8<4?4:0;
  const scores={};
  for (const [action,legal] of player.mask.entries()) if(legal) {
    const branch=new DecisionSession();try {
      const fresh=await branch.begin(spec);
      if(JSON.stringify(fresh.players[side])!==JSON.stringify(player))throw new Error('Prefix replay changed');
      scores[action]=(await branch.decide(action)).reward;
    }finally{branch.close();}
  }
  const gap=scores[target]-Math.max(...Object.entries(scores).filter(([a])=>Number(a)!==target).map(([,r])=>r));
  if (!(gap>0)) throw new Error(`Expected unique tactical optimum failed: ${JSON.stringify({spec,scores,target})}`);
  records.push({spec,player,scores,target,gap,team_keys:before.provenance.teams.map(t=>t.team_key)});
}
writeFileSync(process.argv[2],JSON.stringify({status:'passed',contexts:records.length,branches:records.reduce((s,r)=>s+Object.keys(r.scores).length,0),
  minimum_gap:Math.min(...records.map(r=>r.gap)),records})+'\n',{flag:'wx'});
console.log(JSON.stringify({status:'passed',contexts:records.length,minimum_gap:Math.min(...records.map(r=>r.gap))}));
