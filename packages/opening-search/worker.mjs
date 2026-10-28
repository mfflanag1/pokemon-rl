import readline from 'node:readline';
import {OpponentMixSession} from '../opponent-mix/session.mjs';
import {describe} from '../reference-node/engine.mjs';
import {publicOpening, sampleWorlds, openWorld} from './world.mjs';

let session = null, roots = [], branches = [];
const clear = () => {
  for (const s of [...roots,...branches]) s.session.close();
  roots=[];branches=[];
};
for await (const line of readline.createInterface({input:process.stdin,crlfDelay:Infinity})) {
  let command;
  try {
    if (line.length > 1_000_000) throw new Error('Oversized command');
    command = JSON.parse(line); let data;
    if (command.op === 'hello') data = {schema_version:'opponent-mix-worker-v1',
      extension:'f33-opening-search-v1',reference_commit:describe().commit};
    else if (command.op === 'start') {
      session?.close(); session = new OpponentMixSession();
      if (command.split !== 'development') throw new Error('Development only');
      data = await session.start({episodeIndex:command.episode_index,runSeed:command.run_seed,
        split:command.split,swap:command.swap,tacticalSide:command.tactical_side});
    } else if (command.op === 'opening' && session) data = publicOpening(session,command.side);
    else if (command.op === 'hypotheses') {
      // This operation can run in a separate worker which has never seen the
      // real episode. Its function receives only the validated player input.
      data = await sampleWorlds(command.input,command.search_seed,command.budget);
    } else if (command.op === 'roots') {
      clear();
      if (!Array.isArray(command.worlds) || command.worlds.length > 8) throw new Error('World count');
      for (const world of command.worlds) roots.push({world,session:await openWorld(world)});
      data = roots.map(r => r.session.snapshot());
    } else if (command.op === 'branches') {
      if (branches.length || command.opponent_actions.length !== roots.length) throw new Error('Branch lifecycle');
      for (let w=0;w<roots.length;w++) {
        const {world,session:root}=roots[w], frame=root.snapshot();
        for (let action=0;action<14;action++) {
          if (!frame.players[world.side].mask[action]) continue;
          const child=await openWorld(world);
          branches.push({session:child,world:w,action});
          await child.step({actions:{[world.side]:action,[world.foe]:command.opponent_actions[w]},
            request_ids:Object.fromEntries(['p1','p2'].map(s => [s,frame.players[s].request_id]))});
        }
      }
      data=branches.map((r,index)=>({index,world:r.world,action:r.action,frame:r.session.snapshot()}));
    } else if (command.op === 'branch_steps') {
      if (!Array.isArray(command.steps) || new Set(command.steps.map(s=>s.index)).size !== command.steps.length)
        throw new Error('Unique branch steps required');
      data=[];
      for (const step of command.steps) {
        const branch=branches[step.index];
        if (!branch) throw new Error('Unknown branch');
        data.push({index:step.index,world:branch.world,action:branch.action,
          frame:await branch.session.step(step)});
      }
    } else if (command.op === 'world') {
      session?.close(); session = await openWorld(command.world); data = session.snapshot();
    } else if (command.op === 'step' && session) data = await session.step(command);
    else if (command.op === 'cutoff' && session) data = session.cutoff();
    else if (command.op === 'close') {session?.close();session=null;clear();data={closed:true};}
    else throw new Error('Unknown operation or missing session');
    process.stdout.write(JSON.stringify({id:command.id,ok:true,data})+'\n');
  } catch (error) {
    session?.close(); session = null; clear();
    process.stdout.write(JSON.stringify({id:command?.id??null,ok:false,error:error.message})+'\n');
  }
}
session?.close();clear();
