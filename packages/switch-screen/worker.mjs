// Actors see only the unchanged receiving-player projection; no privileged audit.
import assert from 'node:assert/strict';
import readline from 'node:readline';
import {Session,choices} from '../switch-reference/session.mjs';
import {reference} from '../switch-reference/profile.mjs';

export class Worker {
  constructor(){this.games=[];this.reference=reference();}
  done(s){return s.frame().phase==='terminal'||s.truncated();}
  frame(){return {type:'batch',games:this.games.map(({id,s})=>({id,...s.frame(),truncated:s.truncated()}))};}
  async handle(c){
    assert.ok(c&&typeof c==='object');
    if(c.op==='start'){
      assert.ok(this.games.every(({s})=>this.done(s)),'Discarding unfinished game');
      assert.ok(Array.isArray(c.games)&&c.games.length>0&&c.games.length<=32);
      assert.equal(new Set(c.games.map(g=>g.id)).size,c.games.length);
      assert.ok(c.games.every(g=>Number.isSafeInteger(g.id)&&g.id>=0));
      for(const {s} of this.games)s.close();this.games=[];
      for(const g of c.games){const s=new Session();this.games.push({id:g.id,s});await s.start({teams:g.teams,seed:g.seed});}
      return this.frame();
    }
    if(c.op==='step'){
      const live=this.games.filter(({s})=>!this.done(s));
      assert.ok(live.length);assert.equal(c.decisions.length,live.length);
      for(let i=0;i<live.length;i++){
        const {id,s}=live[i],d=c.decisions[i],f=s.frame();
        assert.equal(d.id,id);assert.equal(d.group,f.group);assert.equal(d.turn,f.turn);
        assert.deepEqual(d.requests,f.views.map(v=>v.request_id));
        assert.ok(Array.isArray(d.actions)&&d.actions.length===2);
        assert.ok(d.actions.every((a,side)=>Number.isInteger(a)&&(f.requested.includes(side)?a>=0&&a<4:a===-1)));
      }
      for(let i=0;i<live.length;i++){
        const {id,s}=live[i],d=c.decisions[i];await s.step(choices(s.frame(),d.actions),(id+d.group)%2);
      }
      return this.frame();
    }
    if(c.op==='finish'){
      assert.ok(this.games.every(({s})=>this.done(s)));assert.deepEqual(reference(),this.reference);
      for(const {s} of this.games)s.close();this.games=[];return {type:'finished',reference:this.reference};
    }
    throw Error('Unknown screen operation');
  }
}
if(process.argv[1]===new URL(import.meta.url).pathname){
  const w=new Worker();console.log(JSON.stringify({type:'ready',reference:w.reference}));
  try{for await(const line of readline.createInterface({input:process.stdin})){
    const r=await w.handle(JSON.parse(line));console.log(JSON.stringify(r));if(r.type==='finished')break;
  }}catch(e){console.error(e.stack);process.exitCode=1;}
  finally{for(const {s} of w.games)s.close();}
}
