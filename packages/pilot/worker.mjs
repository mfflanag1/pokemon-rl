// Local JSONL adapter. Actors receive only R1 player channels; no battle state is exported.
import assert from 'node:assert/strict';
import readline from 'node:readline';
import {BattleStream,checkSeed,describe} from '../reference-node/engine.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {PROFILE,team,admit} from '../slice/reference.mjs';

class Match extends BattleStream {
  constructor(id) {super({noCatch:true,keepAlive:true});this.id=id;this.projection=new PlayerProjection();}
  push(message) {const n=message.indexOf('\n');this.projection.ingest(message.slice(0,n),message.slice(n+1));}
  views() {return ['p1','p2'].map(side=>this.projection.view(side));}
  frame() {
    assert.equal(this.projection.errors.length,0);
    const ended=this.battle.ended;
    const winner=ended ? ['P1','P2'].indexOf(this.battle.winner) : -1;
    assert.ok(!ended || winner>=0,'Unsupported draw/termination');
    return {id:this.id,ended,winner,views:this.views()};
  }
}
export class Worker {
  constructor() {this.games=[];this.reference=describe();assert.equal(this.reference.commit,PROFILE.showdown_commit);}
  frame() {return {type:'batch',games:this.games.map(s=>s.frame())};}
  async handle(command) {
    assert.ok(command && typeof command==='object');
    if(command.op==='start') {
      assert.ok(this.games.every(s=>s.battle.ended),'Cannot discard live games');
      assert.ok(Array.isArray(command.games)&&command.games.length>0&&command.games.length<=40);
      const ids=new Set();
      for(const g of command.games) {assert.ok(Number.isSafeInteger(g.id)&&g.id>=0&&!ids.has(g.id));ids.add(g.id);checkSeed(g.seed);}
      for(const s of this.games)s.destroy();this.games=[];
      for(const g of command.games) {
        const s=new Match(g.id);this.games.push(s);const teams=[team(),team()];admit(teams);
        await s.write(`>start ${JSON.stringify({formatid:PROFILE.reference_format,seed:g.seed})}`);
        for(let i=0;i<2;i++)await s.write(`>player p${i+1} ${JSON.stringify({name:`P${i+1}`,team:teams[i]})}`);
        assert.equal(s.battle.reportExactHP,true);assert.equal(s.battle.turn,1);
      }
      return this.frame();
    }
    if(command.op==='step') {
      const live=this.games.filter(s=>!s.battle.ended);
      assert.ok(live.length>0);assert.ok(Array.isArray(command.decisions));assert.equal(command.decisions.length,live.length);
      // Validate the entire batch before committing any action.
      for(let i=0;i<live.length;i++) {
        const s=live[i],d=command.decisions[i],views=s.views();
        assert.equal(d.id,s.id);assert.equal(d.turn,s.battle.turn);
        assert.deepEqual(d.requests,views.map(v=>v.request_id));
        assert.ok(Array.isArray(d.actions)&&d.actions.length===2&&d.actions.every(a=>Number.isInteger(a)&&a>=0&&a<2));
        assert.ok(s.battle.turn<=PROFILE.max_turns);
      }
      for(let i=0;i<live.length;i++) {
        const s=live[i],d=command.decisions[i],before=s.views(),first=(d.id+d.turn)%2;
        await s.write(`>p${first+1} move ${d.actions[first]+1}`);
        assert.deepEqual(s.views(),before,'Pending commitment changed player information');
        await s.write(`>p${2-first} move ${d.actions[1-first]+1}`);
      }
      return this.frame();
    }
    if(command.op==='finish') {
      assert.ok(this.games.every(s=>s.battle.ended));assert.deepEqual(describe(),this.reference);
      for(const s of this.games)s.destroy();this.games=[];
      return {type:'finished',reference:this.reference};
    }
    throw new Error('Unknown pilot operation');
  }
}
if(process.argv[1]===new URL(import.meta.url).pathname) {
  const worker=new Worker();console.log(JSON.stringify({type:'ready',reference:worker.reference}));
  try {
    for await(const line of readline.createInterface({input:process.stdin})) {
      const response=await worker.handle(JSON.parse(line));console.log(JSON.stringify(response));
      if(response.type==='finished')break;
    }
  } catch(error) {console.error(error.stack);process.exitCode=1;}
  finally {for(const s of worker.games)s.destroy();}
}
