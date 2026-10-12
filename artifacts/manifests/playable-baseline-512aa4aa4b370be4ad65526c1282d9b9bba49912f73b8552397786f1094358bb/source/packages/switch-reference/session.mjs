import assert from 'node:assert/strict';
import {BattleStream,checkSeed} from '../reference-node/engine.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {hashJSON} from '../battle-contract/canonical.mjs';
import {PROFILE,SIDES,team} from './profile.mjs';
import {mask,command} from './actions.mjs';

// Profile-only oracle. Full simulator state is available solely to fixture audits.
export class Session extends BattleStream {
  constructor(){super({noCatch:true,keepAlive:true});this.projection=new PlayerProjection();this.group=0;this.tape=[];this.closed=false;}
  push(message){const n=message.indexOf('\n');this.projection.ingest(message.slice(0,n),message.slice(n+1));}
  views(){assert.equal(this.projection.errors.length,0);return SIDES.map(s=>this.projection.view(s));}
  async start(spec){
    assert.ok(!this.battle&&!this.closed);assert.deepEqual(Object.keys(spec).sort(),['seed','teams']);checkSeed(spec.seed);
    assert.ok(Array.isArray(spec.teams)&&spec.teams.length===2);const teams=spec.teams.map(team);this.spec=structuredClone(spec);
    await this.write(`>start ${JSON.stringify({formatid:PROFILE.reference_format,seed:spec.seed})}`);
    for(let i=0;i<2;i++)await this.write(`>player p${i+1} ${JSON.stringify({name:`P${i+1}`,team:teams[i]})}`);
    assert.equal(this.battle.reportExactHP,false);assert.equal(this.battle.debugMode,false);assert.equal(this.battle.turn,1);
    this.initial=this.frame();return this.initial;
  }
  frame(){
    assert.ok(this.battle&&!this.closed);const views=this.views();
    const ended=this.battle.ended,winner=ended?({'P1':'p1','P2':'p2','':null})[this.battle.winner]:null;
    assert.ok(!ended||winner!==undefined);
    const requested=ended?[]:views.flatMap((v,i)=>!v.request.wait?[i]:[]);
    const phase=ended?'terminal':requested.every(i=>views[i].request.forceSwitch)?'replacement':'joint_move';
    if(!ended){assert.ok(requested.length);assert.ok(phase==='replacement'||requested.length===2);for(const v of views)mask(v.request);}
    return {group:this.group,phase,turn:views[0].turn,requested,winner,views};
  }
  truncated(){const f=this.frame();return f.phase==='joint_move'&&f.turn>PROFILE.max_resolved_turns;}
  async step(decisions,first=0){
    assert.ok(!this.busy&&!this.failed,'Concurrent or failed session');this.busy=true;
    const writes=this.tape.length;
    try{return await this.stepOnce(decisions,first);}
    catch(error){if(this.tape.length!==writes)this.failed=true;throw error;}
    finally{this.busy=false;}
  }
  async stepOnce(decisions,first=0){
    const before=this.frame();assert.ok(!this.truncated()&&before.phase!=='terminal');assert.ok(first===0||first===1);
    assert.ok(Array.isArray(decisions));assert.equal(decisions.length,before.requested.length);
    assert.ok(this.tape.length+decisions.length<=PROFILE.max_decisions);
    const bySide=new Map();
    for(const d of decisions){
      assert.deepEqual(Object.keys(d).sort(),['action','group','request_id','side','view_hash']);assert.ok(SIDES.includes(d.side)&&!bySide.has(d.side));
      const i=SIDES.indexOf(d.side),v=before.views[i];assert.ok(before.requested.includes(i));
      assert.equal(d.group,before.group);assert.equal(d.request_id,v.request_id);assert.equal(d.view_hash,hashJSON(v));
      bySide.set(d.side,{...d,command:command(v.request,d.action)});
    }
    const order=[first,1-first].filter(i=>before.requested.includes(i));
    for(let j=0;j<order.length;j++){
      const i=order[j],d=bySide.get(SIDES[i]);assert.ok(d);await this.write(`>${d.side} ${d.command}`);this.tape.push(d);
      if(j===0&&order.length===2)assert.deepEqual(this.views(),before.views,'Pending choice leaked');
    }
    this.group++;return this.frame();
  }
  close(){this.closed=true;this.destroy();}
}
export function choices(frame,actions){return frame.requested.map(i=>({side:SIDES[i],group:frame.group,
  request_id:frame.views[i].request_id,view_hash:hashJSON(frame.views[i]),action:actions[i]}));}
export function audit(session){return {turn:session.battle.turn,seed:session.battle.prng.getSeed(),sides:session.battle.sides.map(s=>({
  id:s.id,pokemon:s.pokemon.map(m=>({name:m.name,species:m.species.id,hp:m.hp,maxhp:m.maxhp,active:m.isActive,
    fainted:m.fainted,ability:m.ability,item:m.item,status:m.status,stats:m.storedStats,
    pp:m.moveSlots.map(m=>({id:m.id,pp:m.pp})),boosts:{...m.boosts},volatiles:Object.keys(m.volatiles)}))}))};}
