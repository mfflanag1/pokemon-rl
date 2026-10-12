// Passive instrumentation of the sealed oracle; never changes its random outcomes.
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {gzipSync,gunzipSync} from 'node:zlib';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import {seedFor} from '../reference-node/engine.mjs';
import {fileHash} from '../reference-node/provenance.mjs';
import {canonical,hashJSON} from '../battle-contract/canonical.mjs';
import {ROSTER,PROFILE,reference} from '../switch-reference/profile.mjs';
import {Session,choices,audit} from '../switch-reference/session.mjs';
import {mask} from '../switch-reference/actions.mjs';

const species=PROFILE.species.map(x=>x.id),member=p=>PROFILE.names.indexOf(p.name);
export const PLAN=JSON.parse(readFileSync(new URL('../../contracts/switch-engine-v1/plan.json',import.meta.url)));

export function snapshot(session,history){
  const frame=session.frame(),b=session.battle,ordered=b.sides.map(s=>PROFILE.names.map(n=>s.pokemon.find(p=>p.name===n)));
  return {species:ordered.map(s=>s.map(p=>species.indexOf(p.species.id))),swift:ordered.map(s=>s.map(p=>Number(p.set.evs.spe===252))),
    hp:ordered.map(s=>s.map(p=>p.hp)),pp:ordered.map(s=>s.map(p=>p.moveSlots.map(m=>m.pp))),active:b.sides.map(s=>member(s.active[0])),
    phase:{joint_move:0,replacement:1,terminal:2}[frame.phase],turn:b.turn,group:session.group,
    winner:frame.winner===null?-1:Number(frame.winner[1])-1,
    seen:[0,1].map(side=>[0,1].map(m=>Number(m===0||history.some(e=>e[1]===side&&e[2]===0&&e[3]===m)))),history:structuredClone(history)};
}

export async function battleCase(id,{instrument=true,smoke=false,spec:given,script,expected}={}){
  const teamIDs=[id%24,Math.floor(id/24)%24];
  const spec=given??{teams:teamIDs.map(i=>ROSTER[i].id),seed:seedFor(`${PLAN.namespace}/${smoke?'smoke':'fresh'}/${id}`)};
  const s=new Session();let current=null,words=null,events=[],draws=[],history=[];
  try{
    await s.start(spec);const b=s.battle,initial={state:snapshot(s,history),views:s.views(),audit:audit(s)},steps=[];
    const raw=b.prng.rng.next.bind(b.prng.rng),shuffle=b.prng.shuffle.bind(b.prng);
    const random=b.random.bind(b),crit=b.randomChance.bind(b),getDamage=b.actions.getDamage.bind(b.actions),switchIn=b.actions.switchIn.bind(b.actions);
    if(instrument)b.prng.rng.next=()=>{const word=raw();if(words)words.push(word);return word;};
    const traced=(role,detail,fn)=>{
      if(!instrument)return fn();assert.equal(words,null);words=[];
      try{const result=fn();assert.equal(words.length,1,`Unexpected ${role} draws`);draws.push({role,...detail,word:words[0],result});return result;}
      finally{words=null;}
    };
    b.randomChance=(n,d)=>current&&n===1&&d===24?traced('crit',{side:current.side},()=>crit(n,d)):crit(n,d);
    b.random=(...args)=>{
      const value=current&&args.length===1&&args[0]===16?traced('roll',{side:current.side},()=>random(...args)):random(...args);
      if(current&&args.length===1&&args[0]===16)current.roll=value;return value;
    };
    b.prng.shuffle=(items,start=0,end=items.length)=>{
      const segment=items.slice(start,end),a=segment[0],z=segment[1];
      if(segment.length!==2||!a.pokemon||!z.pokemon||a.choice!==z.choice||!['move','switch','instaswitch'].includes(a.choice))return shuffle(items,start,end);
      const order=segment.map(x=>x.pokemon.side.n);assert.deepEqual([...order].sort(),[0,1]);
      return traced('tie',{order,events_before:events.length},()=>{shuffle(items,start,end);return items[start].pokemon.side.n;});
    };
    b.actions.getDamage=(source,target,move,...args)=>{
      assert.equal(current,null);current={side:source.side.n};
      try{
        const amount=getDamage(source,target,move,...args),critical=target.getMoveHitData(move).crit;
        assert(Number.isInteger(amount)&&amount>0&&typeof critical==='boolean'&&Number.isInteger(current.roll));
        const slot=source.moveSlots.findIndex(m=>m.id===move.id);assert(slot>=0&&slot<2);
        const row=[b.turn,source.side.n,1,member(source),species.indexOf(source.species.id),slot,member(target),Math.max(0,target.hp-amount),Number(critical)];
        events.push({row,roll:current.roll,amount});history.push(row);return amount;
      }finally{current=null;}
    };
    b.actions.switchIn=(pokemon,...args)=>{
      const result=switchIn(pokemon,...args);assert(pokemon.isActive&&pokemon.hp>0);
      const row=[b.turn,pokemon.side.n,0,member(pokemon),species.indexOf(pokemon.species.id),-1,member(pokemon),pokemon.hp,0];
      events.push({row});history.push(row);return result;
    };
    if(expected)assert.deepEqual({frame:s.frame(),audit:audit(s)},expected[0]);
    while(!b.ended&&!s.truncated()){
      assert(steps.length<PLAN.max_groups);const frame=s.frame();events=[];draws=[];
      const actions=frame.views.map((v,side)=>{
        if(!frame.requested.includes(side))return -1;
        if(script)return script[frame.group][side];
        const legal=mask(v.request).flatMap((ok,i)=>ok?[i]:[]),style=id%5;
        if(style===0)return legal.includes(0)?0:legal[0];
        if(style===1)return legal.includes(1)?1:legal[0];
        if(style===2)return legal.find(a=>a>=2)??legal[0];
        return legal[parseInt(hashJSON([PLAN.namespace,id,frame.group,side]).slice(0,8),16)%legal.length];
      });
      await s.step(choices(frame,actions),(id+frame.group)%2);
      assert(events.length>0&&events.length<=2);
      const chance={crit:[false,false],roll:[0,0],tie:events[0].row[1]};
      for(const e of events)if(e.row[2]===1){chance.crit[e.row[1]]=Boolean(e.row[8]);chance.roll[e.row[1]]=e.roll;}
      if(instrument)for(const d of draws){
        if(d.role==='crit')assert.equal(d.result,Math.floor(d.word*24/2**32)===0);
        if(d.role==='roll')assert.equal(d.result,d.word>>>28);
        if(d.role==='tie')assert.equal(d.result,d.order[d.word>>>31]);
      }
      if(expected)assert.deepEqual({frame:s.frame(),audit:audit(s)},expected[steps.length+1]);
      steps.push({actions,chance,draws,events:events.map(e=>e.row),state:snapshot(s,history),views:s.views(),audit:audit(s)});
    }
    const result={id,spec,teams:spec.teams.map(t=>ROSTER.findIndex(r=>r.id===t)),initial,steps,
      outcome:s.battle.ended?'terminal':'truncated'};
    return result;
  }finally{s.close();}
}

export async function checkUnpatched(row){
  const s=new Session();
  try{
    await s.start(row.spec);
    assert.deepEqual(s.views(),row.initial.views);assert.deepEqual(audit(s),row.initial.audit);
    for(const step of row.steps){
      const frame=s.frame();await s.step(choices(frame,step.actions),(row.id+frame.group)%2);
      assert.deepEqual(s.views(),step.views);assert.deepEqual(audit(s),step.audit);
    }
    assert.equal(s.battle.ended?'terminal':'truncated',row.outcome);
  }finally{s.close();}
}

export async function capture(directory,{smoke=false}={}){
  mkdirSync(directory);const manifest=reference(),count=smoke?24:PLAN.fresh_games,started=performance.now();
  const plan={plan:PLAN,smoke,count,reference:manifest};writeFileSync(`${directory}/plan.json`,JSON.stringify(plan)+'\n',{flag:'wx'});
  const old=JSON.parse(gunzipSync(readFileSync('artifacts/runs/m3-switch-reference-v1/original/fixtures.json.gz'))),legacy=[];
  for(const [id,c] of old.cases.entries()){
    const script=c.snapshots.slice(0,-1).map((s,g)=>[0,1].map(side=>c.tape.find(d=>d.group===g&&d.side===`p${side+1}`)?.action??-1));
    legacy.push(await battleCase(id,{spec:c.spec,script,expected:c.snapshots}));
  }
  writeFileSync(`${directory}/legacy.json.gz`,gzipSync(canonical(legacy)),{flag:'wx'});
  let rows=[],shards=[];
  for(let id=0;id<count;id++){
    const row=await battleCase(id,{smoke});
    if(id<PLAN.passivity_games){
      await checkUnpatched(row);
    }
    rows.push(row);
    if(rows.length===PLAN.shard_size||id+1===count){
      const file=`cases-${String(shards.length).padStart(4,'0')}.json.gz`;writeFileSync(`${directory}/${file}`,gzipSync(canonical(rows)),{flag:'wx'});
      shards.push({file,sha256:fileHash(`${directory}/${file}`),count:rows.length,first:id+1-rows.length});rows=[];
      writeFileSync(`${directory}/progress.json`,JSON.stringify({completed:id+1,total:count}));
    }
  }
  assert.deepEqual(reference(),manifest);
  const report={status:'complete',plan_sha256:fileHash(`${directory}/plan.json`),count,shards,legacy:72,legacy_sha256:fileHash(`${directory}/legacy.json.gz`),passivity:Math.min(count,PLAN.passivity_games),elapsed_ms:performance.now()-started};
  writeFileSync(`${directory}/report.json`,JSON.stringify(report)+'\n',{flag:'wx'});console.log(JSON.stringify(report));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)await capture(process.argv[2],{smoke:process.argv.includes('--smoke')});
