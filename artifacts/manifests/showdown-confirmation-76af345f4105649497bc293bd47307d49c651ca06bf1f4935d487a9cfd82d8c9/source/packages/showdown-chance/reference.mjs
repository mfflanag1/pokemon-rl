import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import path from 'node:path';
import {performance} from 'node:perf_hooks';
import {BattleStream,seedFor,describe} from '../reference-node/engine.mjs';
import {ROOT,fileHash} from '../reference-node/provenance.mjs';
import {hashJSON,canonical} from '../battle-contract/canonical.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {PROFILE,team,admit} from '../slice/reference.mjs';
import {CONTRACT} from './word-oracle.mjs';

class Capture extends BattleStream {
  constructor(){super({noCatch:true,keepAlive:true});this.projection=new PlayerProjection();}
  push(message){const n=message.indexOf('\n');this.projection.ingest(message.slice(0,n),message.slice(n+1));}
  views(){return ['p1','p2'].map(p=>this.projection.view(p));}
}
export async function battleCase(number,{instrument=true}={}){
  assert.ok(Number.isSafeInteger(number)&&number>=0);
  const seed=seedFor(`${CONTRACT.fresh_capture.namespace}/${number}`),s=new Capture(),teams=[team(),team()];admit(teams);
  try{
    await s.write(`>start ${JSON.stringify({formatid:PROFILE.reference_format,seed})}`);
    for(let i=0;i<2;i++)await s.write(`>player p${i+1} ${JSON.stringify({name:`P${i+1}`,team:teams[i]})}`);
    const b=s.battle;assert.equal(b.reportExactHP,true);assert.equal(b.turn,1);
    const initial=s.views(),turns=[];let current=null,events=[],chance=[],scope=null,drawIndex=0;
    const getDamage=b.actions.getDamage.bind(b.actions),random=b.random.bind(b),randomChance=b.randomChance.bind(b);
    const next=b.prng.rng.next.bind(b.prng.rng),shuffle=b.prng.shuffle.bind(b.prng);
    if(instrument)b.prng.rng.next=function(){const word=next();assert.ok(Number.isInteger(word)&&word>=0&&word<2**32);
      if(scope)scope.push({word,index:drawIndex});drawIndex++;return word;};
    function traced(role,metadata,fn){
      if(!instrument)return fn();
      assert.equal(scope,null);scope=[];
      try{const result=fn();assert.equal(scope.length,1,`Unexpected ${role} consumption`);
        chance.push({role,...metadata,...scope[0],result});return result;
      }finally{scope=null;}
    }
    b.randomChance=function(n,d){return current&&n===1&&d===24?traced('crit',{actor:current.actor},()=>randomChance(n,d)):randomChance(n,d);};
    b.random=function(...args){
      const value=current&&args.length===1&&args[0]===16?traced('roll',{actor:current.actor},()=>random(...args)):random(...args);
      if(current&&args.length===1&&args[0]===16){assert.equal(current.roll,undefined);current.roll=100-value;}return value;
    };
    b.prng.shuffle=function(items,start=0,end=items.length){
      const segment=items.slice(start,end);
      const moves=segment.length===2&&segment.every(x=>x.choice==='move'&&x.pokemon);
      if(!moves)return shuffle(items,start,end);
      const order=segment.map(x=>x.pokemon.side.n);assert.deepEqual([...order].sort(),[0,1]);
      traced('tie',{order},()=>{shuffle(items,start,end);return items[start].pokemon.side.n;});
    };
    b.actions.getDamage=function(source,target,move,...args){
      assert.equal(current,null);const event={actor:source.side.n,move:move.id};current=event;
      try{const result=getDamage(source,target,move,...args);event.crit=target.getMoveHitData(move).crit;event.damage=result;
        assert.equal(typeof event.crit,'boolean');assert.ok(event.roll>=85&&event.roll<=100);events.push(event);return result;
      }finally{current=null;}
    };
    while(!b.ended){
      assert.ok(turns.length<4);const turn=b.turn,hex=hashJSON([CONTRACT.fresh_capture.actions_namespace,number,turn]);
      const actions=[parseInt(hex[0],16)%2,parseInt(hex[1],16)%2],submitted_first=(number+turn)%2;
      const before=s.views();events=[];chance=[];
      await s.write(`>p${submitted_first+1} move ${actions[submitted_first]+1}`);
      const pending=s.views();assert.deepEqual(pending,before,'Pending commitment visible');assert.equal(events.length,0);assert.equal(chance.length,0);
      await s.write(`>p${2-submitted_first} move ${actions[1-submitted_first]+1}`);
      assert.equal(s.projection.errors.length,0);
      if(instrument){
        assert.equal(chance.filter(x=>x.role==='crit').length,events.length);
        assert.equal(chance.filter(x=>x.role==='roll').length,events.length);
        const ties=chance.filter(x=>x.role==='tie');
        assert.ok(actions[0]===actions[1]?ties.length>0:ties.length===0);
        if(ties.length){assert.equal(ties.at(-1).result,events[0].actor);
          assert.ok(chance.findIndex(x=>x.role!=='tie')>chance.findLastIndex(x=>x.role==='tie'),'Tie after first hit');}
      }
      turns.push({turn,actions,submitted_first,pending,events,tie_first:actions[0]===actions[1]?events[0].actor:null,chance,views:s.views()});
    }
    return {id:number,seed,initial,turns};
  }finally{s.destroy();}
}
export function sources(){
  return Object.fromEntries(['packages/showdown-chance/reference.mjs','packages/showdown-chance/word-oracle.mjs',
    'contracts/showdown-chance-v1/contract.json','packages/jax-slice/reference.mjs','packages/slice/reference.mjs',
    'contracts/slice-v1/profile.json','contracts/jax-slice-v1/view.md',
    ...['engine','provenance','observations'].map(n=>`packages/reference-node/${n}.mjs`),
    ...['prng','battle','battle-actions','battle-queue','pokemon','side'].map(n=>`vendor/pokemon-showdown/sim/${n}.ts`)
  ].map(p=>[p,fileHash(path.join(ROOT,p))]));
}
export async function capture(out){
  mkdirSync(out);const reference=describe();assert.equal(reference.commit,CONTRACT.showdown_commit);
  const plan={version:'showdown-player-capture-1',contract:CONTRACT,profile:PROFILE,reference,sources:sources()};
  writeFileSync(path.join(out,'plan.json'),canonical(plan)+'\n',{flag:'wx'});
  const start=performance.now(),shards=[];let rows=[];
  for(let i=0;i<CONTRACT.fresh_capture.count;i++){
    rows.push(await battleCase(i));
    if(rows.length===CONTRACT.fresh_capture.shard_size||i+1===CONTRACT.fresh_capture.count){
      const name=`cases-${String(shards.length).padStart(4,'0')}.jsonl.gz`,first=rows[0].id;
      writeFileSync(path.join(out,name),gzipSync(rows.map(c=>canonical(c)+'\n').join('')),{flag:'wx'});
      shards.push({file:name,sha256:fileHash(path.join(out,name)),first,count:rows.length});rows=[];
      console.log(JSON.stringify({completed:i+1,total:CONTRACT.fresh_capture.count}));
    }
  }
  assert.deepEqual(describe(),reference);assert.deepEqual(sources(),plan.sources);
  const body={version:'showdown-player-index-1',status:'complete',plan_hash:hashJSON(plan),count:CONTRACT.fresh_capture.count,
    shards,elapsed_ms:Math.round(performance.now()-start)};
  writeFileSync(path.join(out,'index.json'),canonical({...body,content_hash:hashJSON(body)})+'\n',{flag:'wx'});
}
if(process.argv[1]===new URL(import.meta.url).pathname)await capture(path.resolve(process.argv[2]));
