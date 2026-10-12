// Conditional reference oracle: private HP and crit/roll are injected only here.
// This is not a natural episode or a sampler-frequency test.
import assert from 'node:assert/strict';
import {seedFor} from '../reference-node/engine.mjs';
import {PROFILE,ROSTER} from './profile.mjs';
import {Session} from './session.mjs';
export async function damageGrid(){
  const rows=[];
  for(const a of PROFILE.species)for(const b of PROFILE.species)for(const seat of [0,1]){
    const teamFor=(p,v)=>ROSTER.find(r=>r.species[0]===p.id&&r.variants[0]===v).id;
    const teams=[teamFor(a,'plain'),teamFor(b,'swift')];if(seat===1)teams.reverse();
    const s=new Session();try{
      await s.start({teams,seed:seedFor(`switch-damage-grid-v1/${a.id}/${b.id}/${seat}`)});
      const battle=s.battle,source=battle.sides[seat].active[0],target=battle.sides[1-seat].active[0];
      const hp=[Math.floor(source.maxhp/3)-1,Math.floor(source.maxhp/3),Math.floor(source.maxhp/3)+1];
      const values=[];let calls=0;
      for(const health of hp){source.hp=health;const moves=[];
        for(const name of a.moves){const crits=[];
          for(const critical of [false,true]){const rolls=[];
            for(let roll=0;roll<16;roll++){
              const move=battle.dex.getActiveMove(name);move.willCrit=critical;
              // Preserve Showdown's randomizer and integer ordering; intercept only
              // its bounded 0..15 draw. Any additional random call fails closed.
              const before=calls;battle.random=(m,n)=>{assert.equal(m,16);assert.equal(n,undefined);calls++;return roll;};
              const result=battle.actions.getDamage(source,target,move,true);
              assert.equal(calls,before+1);assert.ok(Number.isInteger(result)&&result>0);rolls.push(result);
            }crits.push(rolls);
          }moves.push(crits);
        }values.push(moves);
      }
      for(let crit=0;crit<2;crit++)for(let roll=0;roll<16;roll++){
        assert.equal(values[0][0][crit][roll],values[1][0][crit][roll]);
        assert.ok(values[1][0][crit][roll]>values[2][0][crit][roll],'Ability threshold');
        assert.equal(values[0][1][crit][roll],values[2][1][crit][roll],'Normal move is not boosted');
      }
      rows.push({attacker:a.id,defender:b.id,seat,teams,maxhp:source.maxhp,hp,values,calls});
    }finally{s.close();}
  }
  assert.equal(rows.length,18);return rows;
}
