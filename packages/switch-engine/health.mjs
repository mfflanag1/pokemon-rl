// Exhaustive getHealth probe for the admitted species; no battle evidence is reused.
import {Session} from '../switch-reference/session.mjs';
import {ROSTER} from '../switch-reference/profile.mjs';
import {seedFor} from '../reference-node/engine.mjs';
const s=new Session(),rows=[];
try {
  await s.start({teams:[ROSTER[0].id,ROSTER[16].id],seed:seedFor('m3-health-exhaustive')});
  const seen=new Set();
  for(const p of s.battle.sides.flatMap(s=>s.pokemon)){
    if(seen.has(p.maxhp))continue;seen.add(p.maxhp);
    for(let hp=0;hp<=p.maxhp;hp++){
      p.hp=hp;rows.push([hp,p.maxhp,parseInt(p.getHealth().shared)]);
    }
  }
  console.log(JSON.stringify(rows));
}finally{s.close();}
