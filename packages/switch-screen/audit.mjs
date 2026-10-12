// Re-run every stored game with unpatched Showdown, including original view hashes.
import assert from 'node:assert/strict';
import {createReadStream,readFileSync} from 'node:fs';
import {createGunzip} from 'node:zlib';
import {createInterface} from 'node:readline';
import {resolve} from 'node:path';
import {Session,choices} from '../switch-reference/session.mjs';
import {reference} from '../switch-reference/profile.mjs';
import {hashJSON} from '../battle-contract/canonical.mjs';

export async function replayRecord(r){
  const s=new Session();try{
    await s.start(r.spec);
    for(const d of r.decisions){
      const f=s.frame();assert.equal(f.group,d.group);assert.equal(f.turn,d.turn);
      assert.deepEqual(f.views.map(v=>v.request_id),d.requests);
      assert.deepEqual(f.views.map(v=>v.protocol.length),d.view_lengths);
      assert.deepEqual(f.views.map(hashJSON),d.view_hashes,'Original receiving-player views');
      for(const side of [0,1])if(!f.requested.includes(side))assert.equal(d.actions[side],-1);
      await s.step(choices(f,d.actions),(2*r.pair+r.seat+d.group)%2);
    }
    const f=s.frame();assert.equal(f.group,r.groups);assert.equal(f.winner,r.winner);
    assert.equal(s.truncated(),r.truncated);assert.ok(f.phase==='terminal'||s.truncated());
    assert.deepEqual(f.views,r.final_views,'Complete original public/private player protocols');
  }finally{s.close();}
}

export async function replayRun(directory){
  const root=resolve(directory),ref=reference();
  assert.deepEqual(ref,JSON.parse(readFileSync(resolve(root,'reference.json'))));
  const index=JSON.parse(readFileSync(resolve(root,'matches-index.json')));let games=0;
  for(const item of index){
    assert.ok(/^matches-[0-9]+-[0-9]+-(random|damage-greedy|frozen-initial)\.jsonl\.gz$/.test(item.file));
    let n=0;const stream=createReadStream(resolve(root,item.file)).pipe(createGunzip());
    for await(const line of createInterface({input:stream})){await replayRecord(JSON.parse(line));n++;}
    assert.equal(n,item.games);games+=n;
  }
  assert.deepEqual(reference(),ref);return {status:'passed',games,unpatched_reference:true};
}
if(process.argv[1]===new URL(import.meta.url).pathname)console.log(JSON.stringify(await replayRun(process.argv[2])));
