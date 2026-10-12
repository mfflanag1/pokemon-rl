import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import path from 'node:path';
import {performance} from 'node:perf_hooks';
import {BattleStream,Dex,seedFor,describe} from '../reference-node/engine.mjs';
import {ROOT,fileHash} from '../reference-node/provenance.mjs';
import {hashJSON,canonical} from '../battle-contract/canonical.mjs';

export const PROFILE=JSON.parse(readFileSync(path.join(ROOT,'contracts/slice-v1/profile.json')));
export function team(){return [{species:PROFILE.species,name:PROFILE.species,ability:PROFILE.ability,item:PROFILE.item,
  moves:PROFILE.moves,level:PROFILE.level,nature:PROFILE.nature,gender:PROFILE.gender,teraType:PROFILE.tera_type,
  evs:PROFILE.evs,ivs:PROFILE.ivs}];}
export function admit(teams){assert.deepEqual(teams,[team(),team()],'Unsupported fixed-team profile');}
class Capture extends BattleStream{constructor(){super({noCatch:true,keepAlive:true});this.messages=[];}push(m){this.messages.push(m);}}
function snapshot(b){return {turn:b.turn,phase:b.ended?'terminal':b.requestState,winner:b.ended?b.sides.findIndex(s=>s.name===b.winner):null,
  hp:b.sides.map(s=>s.pokemon[0].hp),pp:b.sides.map(s=>s.pokemon[0].moveSlots.map(m=>m.pp)),
  requests:b.ended?[]:b.sides.map(s=>{const r=s.activeRequest;assert.equal(r.active.length,1);assert.equal(r.active[0].canTerastallize,undefined);
    return {own_hp:s.pokemon[0].hp,moves:r.active[0].moves.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp,disabled:m.disabled}))};})};}
async function start(seed){const s=new Capture();const teams=[team(),team()];admit(teams);
  await s.write(`>start ${JSON.stringify({formatid:PROFILE.reference_format,seed})}`);
  for(let i=0;i<2;i++)await s.write(`>player p${i+1} ${JSON.stringify({name:`P${i+1}`,team:teams[i]})}`);
  assert.equal(s.battle.turn,1);assert.equal(s.battle.requestState,'move');
  for(const side of s.battle.sides){const p=side.pokemon[0];assert.equal(p.maxhp,105);assert.deepEqual([p.storedStats.atk,p.storedStats.def,p.storedStats.spe],[76,55,92]);}
  return s;}
export async function battleCase(number){
  const seed=seedFor(`m1-slice-reference-v1/${number}`),s=await start(seed),b=s.battle;
  const frames=[snapshot(b)],turns=[];let current=null,events=[];
  const getDamage=b.actions.getDamage.bind(b.actions),random=b.random.bind(b);
  b.random=function(...args){const x=random(...args);if(current&&args.length===1&&args[0]===16){assert.equal(current.roll,undefined);current.roll=100-x;}return x;};
  b.actions.getDamage=function(source,target,move,...args){assert.equal(current,null);const event={actor:source.side.n,move:move.id};current=event;
    const result=getDamage(source,target,move,...args);event.crit=target.getMoveHitData(move).crit;event.damage=result;
    assert.equal(typeof event.crit,'boolean');assert.ok(event.roll>=85&&event.roll<=100);events.push(event);current=null;return result;};
  try{
    while(!b.ended){assert.ok(turns.length<PROFILE.max_turns,'Turn bound violated');const turn=b.turn;
      // Action tape is independent of the battle's mechanics RNG.
      const hex=hashJSON(['m1-slice-actions-1',number,turn]);const actions=[parseInt(hex[0],16)%2,parseInt(hex[1],16)%2];
      events=[];await s.write(`>p1 move ${actions[0]+1}\n>p2 move ${actions[1]+1}`);
      assert.ok(events.length>=1&&events.length<=2);turns.push({turn,actions,tie_first:actions[0]===actions[1]?events[0].actor:null,events});frames.push(snapshot(b));
    }
    assert.ok(!s.messages.some(m=>m.includes('|error|')));
    return {id:number,seed,frames,turns};
  }finally{s.destroy();}
}
export async function damageTable(){const s=await start(seedFor('m1-slice-damage-enumeration'));const b=s.battle;
  try{const rows=[];for(const crit of [false,true])for(let roll=85;roll<=100;roll++){
    const move=Dex.getActiveMove('tackle');move.willCrit=crit;b.randomizer=base=>Math.trunc(base*roll/100);
    rows.push({crit,roll,damage:b.actions.getDamage(b.p1.active[0],b.p2.active[0],move,true)});
  }return rows;}finally{s.destroy();}}
export async function capture(out,count=10000){assert.ok(Number.isSafeInteger(count)&&count>0);mkdirSync(out);
  const sources=Object.fromEntries(['packages/slice/reference.mjs','contracts/slice-v1/profile.json'].map(p=>[p,fileHash(path.join(ROOT,p))]));
  const upstreamFiles=['data/moves.ts','data/abilities.ts','data/pokedex.ts','data/rulesets.ts','sim/battle-actions.ts','sim/battle.ts','sim/pokemon.ts','sim/side.ts','sim/prng.ts','sim/dex-moves.ts','sim/dex-formats.ts'];
  const upstream=Object.fromEntries(upstreamFiles.map(p=>[p,fileHash(path.join(ROOT,'vendor/pokemon-showdown',p))]));
  const reference=describe();assert.equal(reference.commit,PROFILE.showdown_commit);
  const plan={version:'m1-slice-campaign-1',profile:PROFILE,count,sources,upstream,reference,scope:'Conditional CPU transition parity using semantic outcomes recorded from Showdown; not JAX or unconditional RNG parity.'};
  writeFileSync(path.join(out,'plan.json'),canonical(plan)+'\n',{flag:'wx'});
  const started=performance.now(),cases=[];
  for(let i=0;i<count;i++){cases.push(await battleCase(i));if((i+1)%1000===0)console.log(JSON.stringify({completed:i+1,requested:count}));}
  const damage=await damageTable();assert.deepEqual(describe(),reference);for(const[p,h]of Object.entries(sources))assert.equal(fileHash(path.join(ROOT,p)),h);
  for(const[p,h]of Object.entries(upstream))assert.equal(fileHash(path.join(ROOT,'vendor/pokemon-showdown',p)),h);
  const body={version:'m1-slice-reference-1',plan_hash:hashJSON(plan),cases,damage_table:damage,elapsed_ms:Math.round(performance.now()-started)};
  const result={...body,content_hash:hashJSON(body)};writeFileSync(path.join(out,'reference.json'),canonical(result)+'\n',{flag:'wx'});return result;
}
if(process.argv[1]===new URL(import.meta.url).pathname)await capture(path.resolve(process.argv[2]),Number(process.argv[3]??10000));
