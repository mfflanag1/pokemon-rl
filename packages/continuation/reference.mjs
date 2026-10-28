// Passive instrumentation of the sealed oracle; never changes its random outcomes.
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {gzipSync,gunzipSync} from 'node:zlib';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import {seedFor} from '../reference-node/engine.mjs';
import {fileHash} from '../reference-node/provenance.mjs';
import {canonical,hashJSON} from '../battle-contract/canonical.mjs';
import {ROSTER,PROFILE,reference} from './profile.mjs';
import {Session,choices,audit} from './session.mjs';
import {mask} from '../switch-reference/actions.mjs';

const species=PROFILE.species.map(x=>x.id),member=p=>PROFILE.names.indexOf(p.name);
export const PLAN=JSON.parse(readFileSync(new URL('../../contracts/continuation-v1/plan.json',import.meta.url)));

export function snapshot(session,history){
  const frame=session.frame(),b=session.battle,ordered=b.sides.map(s=>PROFILE.names.map(n=>s.pokemon.find(p=>p.name===n)));
  return {species:ordered.map(s=>s.map(p=>species.indexOf(p.species.id))),swift:ordered.map(s=>s.map(p=>Number(p.set.evs.spe===252))),
    hp:ordered.map(s=>s.map(p=>p.hp)),pp:ordered.map(s=>s.map(p=>p.moveSlots.map(m=>m.pp))),active:b.sides.map(s=>member(s.active[0])),
    phase:{joint_move:0,replacement:1,terminal:2}[frame.phase],turn:b.turn,group:session.group,
    winner:frame.winner===null?-1:Number(frame.winner[1])-1,
    seen:[0,1].map(side=>[0,1].map(m=>Number(m===0||history.some(e=>e[1]===side&&e[2]===0&&e[3]===m)))),history:structuredClone(history)};
}

export async function battleCase(id,{instrument=true,smoke=false,spec:given,script,expected}={}){
  const teamIDs=[Math.floor(id/6),(Math.floor(id/6)*7+9)%24];
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
        const legal=mask(v.request).flatMap((ok,i)=>ok?[i]:[]),style=id%6;
        if(style===0)return legal.includes(0)?0:legal[0];
        if(style===1)return legal.includes(1)?1:legal[0];
        if(style===2||(style===3&&frame.turn<=24)||(style===4&&frame.turn%3===1))return legal.find(a=>a>=2)??legal[0];
        if(style===3||style===4)return legal.includes(0)?0:legal[0];
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

export async function capture(directory){
  mkdirSync(directory);const manifest=reference(),started=performance.now();
  const plan={plan:PLAN,reference:manifest};writeFileSync(`${directory}/plan.json`,canonical(plan)+'\n',{flag:'wx'});
  const rows=[];
  for(let id=0;id<PLAN.fresh_cases;id++){
    const row=await battleCase(id);await checkUnpatched(row);rows.push(row);
  }
  const coverage={terminal:0,truncated:0,late_terminal:0,replacement:0,late_replacement:0,window_replacement:0,history_over32:0,
    boundary13:0,boundary25:0,boundary37:0,boundary49:0,boundary61:0,max_events:0,post12_moves:0};
  for(const row of rows){
    coverage[row.outcome]++;
    if(row.outcome==='terminal'&&row.steps.at(-1).state.turn>12)coverage.late_terminal++;
    for(const step of row.steps){
      const s=step.state;coverage.max_events=Math.max(coverage.max_events,s.history.length);
      if(s.history.length>32)coverage.history_over32++;
      if(s.phase===1){coverage.replacement++;if(s.turn>12)coverage.late_replacement++;if([12,24,36,48].includes(s.turn))coverage.window_replacement++;}
      if(s.phase===0&&[13,25,37,49,61].includes(s.turn))coverage[`boundary${s.turn}`]++;
      coverage.post12_moves+=step.events.filter(e=>e[0]>12&&e[2]===1).length;
    }
  }
  for(const [key,value] of Object.entries(coverage))assert.ok(value>0,`Missing fixture coverage: ${key}`);
  assert(coverage.max_events<=PLAN.max_events);
  const file='cases.json.gz';writeFileSync(`${directory}/${file}`,gzipSync(canonical(rows)),{flag:'wx'});
  assert.deepEqual(reference(),manifest);
  const report={format:'continuation-reference-1',status:'complete',plan_sha256:fileHash(`${directory}/plan.json`),
    file,sha256:fileHash(`${directory}/${file}`),games:rows.length,unpatched_replays:rows.length,coverage,elapsed_ms:performance.now()-started};
  writeFileSync(`${directory}/report.json`,JSON.stringify(report)+'\n',{flag:'wx'});console.log(JSON.stringify(report));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)await capture(process.argv[2]);
