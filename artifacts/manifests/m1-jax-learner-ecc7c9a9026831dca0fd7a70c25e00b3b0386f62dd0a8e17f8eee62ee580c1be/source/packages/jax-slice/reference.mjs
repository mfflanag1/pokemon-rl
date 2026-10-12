import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import path from 'node:path';
import {performance} from 'node:perf_hooks';
import {BattleStream,seedFor,describe} from '../reference-node/engine.mjs';
import {ROOT,fileHash} from '../reference-node/provenance.mjs';
import {hashJSON,canonical} from '../battle-contract/canonical.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {PROFILE,team,admit} from '../slice/reference.mjs';

class Capture extends BattleStream {
  constructor(){super({noCatch:true,keepAlive:true});this.projection=new PlayerProjection();}
  push(message){const n=message.indexOf('\n');this.projection.ingest(message.slice(0,n),message.slice(n+1));}
  views(){return ['p1','p2'].map(p=>this.projection.view(p));}
}
export async function battleCase(number){
  const seed=seedFor(`m1-g2-player-reference-v1/${number}`),s=new Capture(),teams=[team(),team()];admit(teams);
  await s.write(`>start ${JSON.stringify({formatid:PROFILE.reference_format,seed})}`);
  for(let i=0;i<2;i++)await s.write(`>player p${i+1} ${JSON.stringify({name:`P${i+1}`,team:teams[i]})}`);
  const b=s.battle;assert.equal(b.reportExactHP,true);assert.equal(b.turn,1);
  const initial=s.views(),turns=[];let current=null,events=[];
  const getDamage=b.actions.getDamage.bind(b.actions),random=b.random.bind(b);
  b.random=function(...args){const value=random(...args);if(current&&args.length===1&&args[0]===16){assert.equal(current.roll,undefined);current.roll=100-value;}return value;};
  b.actions.getDamage=function(source,target,move,...args){assert.equal(current,null);const event={actor:source.side.n,move:move.id};current=event;
    const result=getDamage(source,target,move,...args);event.crit=target.getMoveHitData(move).crit;event.damage=result;
    assert.equal(typeof event.crit,'boolean');assert.ok(event.roll>=85&&event.roll<=100);events.push(event);current=null;return result;};
  try{
    while(!b.ended){
      assert.ok(turns.length<4);const turn=b.turn,hex=hashJSON(['m1-g2-actions-1',number,turn]);
      const actions=[parseInt(hex[0],16)%2,parseInt(hex[1],16)%2],submitted_first=(number+turn)%2;
      const before=s.views();events=[];
      await s.write(`>p${submitted_first+1} move ${actions[submitted_first]+1}`);
      const pending=s.views();assert.deepEqual(pending,before,'Pending commitment visible');assert.equal(events.length,0);
      await s.write(`>p${2-submitted_first} move ${actions[1-submitted_first]+1}`);
      assert.equal(s.projection.errors.length,0);
      turns.push({turn,actions,submitted_first,pending,events,tie_first:actions[0]===actions[1]?events[0].actor:null,views:s.views()});
    }
    return {id:number,seed,initial,turns};
  }finally{s.destroy();}
}
export async function capture(out,count=1000){
  assert.ok(Number.isSafeInteger(count)&&count>0);mkdirSync(out);
  const sources=Object.fromEntries(['packages/jax-slice/reference.mjs','packages/slice/reference.mjs','contracts/slice-v1/profile.json','contracts/jax-slice-v1/view.md','packages/reference-node/observations.mjs'].map(p=>[p,fileHash(path.join(ROOT,p))]));
  const reference=describe();assert.equal(reference.commit,PROFILE.showdown_commit);assert.equal(reference.source_dirty,false);
  const plan={version:'m1-g2-player-campaign-1',profile:PROFILE,count,sources,reference,exact_opponent_hp:true,scope:'Fixed profile, both player channels including hidden pending submissions. No official-format certification.'};
  writeFileSync(path.join(out,'plan.json'),canonical(plan)+'\n',{flag:'wx'});
  const started=performance.now(),cases=[];
  for(let i=0;i<count;i++){cases.push(await battleCase(i));if((i+1)%100===0)console.log(JSON.stringify({completed:i+1,requested:count}));}
  assert.deepEqual(describe(),reference);for(const[p,h]of Object.entries(sources))assert.equal(fileHash(path.join(ROOT,p)),h);
  const body={version:'m1-g2-player-reference-1',plan_hash:hashJSON(plan),cases,elapsed_ms:Math.round(performance.now()-started)};
  const result={...body,content_hash:hashJSON(body)};
  writeFileSync(path.join(out,'reference.json'),canonical(result)+'\n',{flag:'wx'});return result;
}
if(process.argv[1]===new URL(import.meta.url).pathname)await capture(path.resolve(process.argv[2]),Number(process.argv[3]??1000));
