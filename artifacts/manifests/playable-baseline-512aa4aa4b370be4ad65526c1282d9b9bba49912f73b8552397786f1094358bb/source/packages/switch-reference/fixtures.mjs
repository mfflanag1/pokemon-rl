import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import {seedFor} from '../reference-node/engine.mjs';
import {hashJSON,canonical} from '../battle-contract/canonical.mjs';
import {PROFILE,ROSTER,SIDES,reference} from './profile.mjs';
import {Session,choices,audit} from './session.mjs';
import {mask,command} from './actions.mjs';
import {damageGrid} from './damage.mjs';

const packView=v=>({view:v,mask:mask(v.request),commands:mask(v.request).map((legal,i)=>legal?command(v.request,i):null)});
const capture=s=>({frame:s.frame(),audit:audit(s)});
function select(frame,style){return frame.views.map(v=>{
  const m=mask(v.request),legal=m.flatMap((ok,i)=>ok?[i]:[]);
  if(!legal.length)return null;
  if(v.request.forceSwitch||style==='switch'||(style==='mixed'&&frame.turn%3===1))return legal.find(i=>i>=2)??legal[0];
  return m[style==='normal'?1:0]?(style==='normal'?1:0):legal[0];
});}
export async function fixture(spec,style,first){
  const s=new Session();try{
    await s.start(spec);const snapshots=[capture(s)];
    while(!s.battle.ended&&!s.truncated()){
      const f=s.frame();await s.step(choices(f,select(f,style)),first);snapshots.push(capture(s));
    }
    const final=s.frame();return {spec,style,first,snapshots,tape:s.tape,
      outcome:{status:s.battle.ended?'terminal':'truncated',winner:final.winner,turn:final.turn}};
  }finally{s.close();}
}
export async function information(){
  const initial=new Map(),after=new Map(),probes=[],comparisons=[];let initial_equal=0,post_equal=0,starts=0;
  for(const observer of [0,1])for(const own of ROSTER)for(const opponent of ROSTER){
    const initialKey=[observer,own.id,opponent.species[0]].join('/');
    const afterKey=[initialKey,opponent.variants[0]].join('/');
    const teams=observer===0?[own.id,opponent.id]:[opponent.id,own.id];
    // Reserve variant does not change the random namespace; no hidden state enters action selection.
    const spec={teams,seed:seedFor(`switch-information-v1/${afterKey}`)};
    const s=new Session();try{
      const f=await s.start(spec);starts++;
      const v=f.views[observer],h=hashJSON(v);
      if(initial.has(initialKey)){assert.equal(h,initial.get(initialKey));initial_equal++;}else initial.set(initialKey,h);
      const g=await s.step(choices(f,[1,1]),observer),v2=g.views[observer],h2=hashJSON(v2);
      if(after.has(afterKey)){assert.equal(h2,after.get(afterKey));post_equal++;}else after.set(afterKey,h2);
      comparisons.push({observer,own:own.id,opponent:opponent.id,initial_key:initialKey,after_key:afterKey,initial_hash:h,after_hash:h2,opponent_own_hash:hashJSON(f.views[1-observer]),initial_view:v,after_view:v2,opponent_own_view:f.views[1-observer]});
      // Keep complete positive-reveal examples in both seats for each lead/reserve/variant.
      if(own.id===ROSTER[0].id){
        assert.equal(g.phase,'joint_move');const actions=[1,1];actions[1-observer]=3;
        const reveal=await s.step(choices(g,actions),observer);
        assert.notEqual(hashJSON(reveal.views[observer]),h2);
        assert.ok(reveal.views[observer].protocol.some(l=>l.startsWith(`|switch|${SIDES[1-observer]}a: B|`)));
        probes.push({observer,spec,initial:packView(v),after:packView(v2),reveal:packView(reveal.views[observer])});
      }
    }finally{s.close();}
  }
  assert.equal(starts,1152);assert.equal(initial_equal,1008);assert.equal(post_equal,864);assert.equal(probes.length,48);
  return {starts,initial_equal,post_equal,comparisons,probes};
}
export async function generate(){
  const manifest=reference(),cases=[];
  for(const [i,row] of ROSTER.entries())for(const style of ['attack','mixed','switch']){
    const spec={teams:[row.id,ROSTER[(i*7+9)%24].id],seed:seedFor(`switch-fixture-v1/${i}/${style}`)};
    const a=await fixture(spec,style,i%2),b=await fixture(spec,style,1-i%2);
    // Submission order must not select a different mechanic outcome or observation history.
    assert.deepEqual(a.snapshots,b.snapshots);assert.deepEqual(a.outcome,b.outcome);
    cases.push(a);
  }
  const info=await information(),coverage={terminal:0,truncated:0,replacement:0,wait:0,voluntary_switch:0,
    critical_hits:0,super_effective:0,resisted:0,low_hp_ability_attacks:0,speed_tie_turns:0,masked_hp:0};
  const golden=[];
  for(const c of cases){coverage[c.outcome.status]++;
    for(const [i,s] of c.snapshots.entries()){
      if(s.frame.phase==='replacement')coverage.replacement++;
      if(s.frame.phase!=='terminal')for(const v of s.frame.views){golden.push(packView(v));if(v.request.wait)coverage.wait++;}
      for(const p of s.audit.sides)for(const m of p.pokemon){assert.equal(m.item,'');assert.ok(m.status===''||(m.status==='fnt'&&m.fainted&&m.hp===0));assert.deepEqual(m.volatiles,[]);assert.ok(Object.values(m.boosts).every(n=>n===0));}
      if(i+1<c.snapshots.length){
        const active=s.audit.sides.map(side=>side.pokemon.find(m=>m.active));
        if(s.frame.phase==='joint_move'&&active[0].stats.spe===active[1].stats.spe)coverage.speed_tie_turns++;
        for(let seat=0;seat<2;seat++){
          const mon=active[seat];if(!mon)continue;const next=c.snapshots[i+1].frame.views[seat].protocol.slice(s.frame.views[seat].protocol.length);
          const signature=PROFILE.species.find(p=>p.id===mon.species).moves[0];
          if(mon.hp>0&&3*mon.hp<=mon.maxhp&&next.some(l=>l.startsWith(`|move|${SIDES[seat]}a: ${mon.name}|${signature}|`)))coverage.low_hp_ability_attacks++;
        }
      }
    }
    const protocol=c.snapshots.at(-1).frame.views[0].protocol;
    for(const line of protocol){
      if(line.startsWith('|-crit|'))coverage.critical_hits++;
      if(line.startsWith('|-supereffective|'))coverage.super_effective++;
      if(line.startsWith('|-resisted|'))coverage.resisted++;
      if(/^\|(?:switch|\-damage)\|p2a:/.test(line)&&line.includes('/100'))coverage.masked_hp++;
    }
    for(const d of c.tape)if(d.action>=2&&c.snapshots[d.group].frame.phase==='joint_move')coverage.voluntary_switch++;
  }
  for(const [key,value] of Object.entries(coverage))assert.ok(value>0,`Missing coverage ${key}`);
  assert.deepEqual(reference(),manifest);
  const damage=await damageGrid();
  return {damage,format:'switch-reference-fixtures-1',profile:PROFILE,manifest,cases,information:info,golden,coverage};
}
if(process.argv[1]===new URL(import.meta.url).pathname){
  const out=process.argv[2];assert.ok(out,'Output directory required');mkdirSync(out,{recursive:false});
  const data=await generate(),bytes=Buffer.from(canonical(data)+'\n');
  writeFileSync(`${out}/fixtures.json.gz`,gzipSync(bytes,{level:9}),{flag:'wx'});
  const summary={format:data.format,profile:PROFILE.id,fixture_hash:hashJSON(data),games:data.cases.length,
    codec_views:data.golden.length,conditional_damage_values:data.damage.reduce((n,r)=>n+r.calls,0),information:{starts:data.information.starts,initial_equal:data.information.initial_equal,
      post_equal:data.information.post_equal,reveal_probes:data.information.probes.length},coverage:data.coverage};
  writeFileSync(`${out}/summary.json`,canonical(summary)+'\n',{flag:'wx'});console.log(JSON.stringify(summary));
}
