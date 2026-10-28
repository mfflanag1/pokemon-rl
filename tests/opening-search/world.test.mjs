import test from 'node:test';
import assert from 'node:assert/strict';
import {OpponentMixSession} from '../../packages/opponent-mix/session.mjs';
import {publicOpening,ownTeam,validateInput,sampleWorlds,openWorld,admittedHypothesis} from '../../packages/opening-search/world.mjs';
import {seedFor,generateTeam} from '../../packages/reference-node/engine.mjs';

async function fixture() {
  const session=new OpponentMixSession();
  await session.start({episodeIndex:0,runSeed:1027263300,split:'development',swap:false,tacticalSide:null});
  return session;
}

test('actor boundary rejects coordinator metadata and foreign private requests',async()=>{
  const s=await fixture();
  try {
    const input=publicOpening(s,'p1');
    for (const field of ['provenance','seed','teams','opponent_player','outcome','swap','split'])
      assert.throws(()=>validateInput({...input,[field]:s.provenance}));
    assert.throws(()=>validateInput({...input,request:s.projection.players.p2.request}));
    assert.throws(()=>validateInput({...input,public_lines:[...input.public_lines,'|request|{}']}));
    assert.equal(input.request.side.id,'p1');
    assert.equal(Object.hasOwn(input,'provenance'),false);
  } finally{s.close();}
});

test('own stats and complete public opening survive reconstructed fresh start',async()=>{
  const s=await fixture();let reconstructed;
  try {
    const input=publicOpening(s,'p1');
    const teams=[ownTeam(input),s.stream.battle.p2.team];
    reconstructed=await openWorld({teams,mechanics_seed:seedFor('f33/test-reconstruction')});
    assert.deepEqual(publicOpening(reconstructed,'p1'),input);
  } finally{s.close();reconstructed?.close();}
});

test('unrevealed bench/item changes cannot affect allowed input or hypotheses',async()=>{
  const real=await fixture();let changed;
  try {
    const input=publicOpening(real,'p1');
    const other=structuredClone(real.stream.battle.p2.team);
    other[2].item=other[2].item==='Leftovers'?'Life Orb':'Leftovers';
    other[2].teraType=other[2].teraType==='Fire'?'Water':'Fire';
    other[3].moves=['protect','rest','sleeptalk','tackle'];
    changed=await openWorld({teams:[ownTeam(input),other],mechanics_seed:seedFor('different-secret-mechanics')});
    assert.deepEqual(publicOpening(changed,'p1'),input);
    const a=await sampleWorlds(input,1027263300,{worlds:2,attempts:512});
    const b=await sampleWorlds(publicOpening(changed,'p1'),1027263300,{worlds:2,attempts:512});
    assert.equal(a.worlds.length,2);assert.deepEqual(a,b);
    for (const world of a.worlds) {
      assert.equal(admittedHypothesis(world.teams[1],world.mechanics_seed),true);
      const h=await openWorld(world);
      try {assert.deepEqual(publicOpening(h,'p1'),input);}finally{h.close();}
    }
  }finally{real.close();changed?.close();}
});

test('ordered-team partition is checked after lead rotation',async()=>{
  // Exact prelaunch defect: original team admitted; fourth rotation reserved.
  const entry=generateTeam(seedFor('f33/hypotheses/1327821940/162'));
  assert.equal(admittedHypothesis(entry.team,entry.seed),true);
  const rotated=[...entry.team.slice(4),...entry.team.slice(0,4)];
  assert.equal(admittedHypothesis(rotated,entry.seed),false);
  const s=await fixture();
  try {
    const result=await sampleWorlds(publicOpening(s,'p1'),1327821940,{worlds:2,attempts:512});
    assert.equal(result.worlds.length,2);
    for(const world of result.worlds)assert.equal(admittedHypothesis(world.teams[1],world.mechanics_seed),true);
  }finally{s.close();}
});

test('full public event mismatch is rejected, never silently approximated',async()=>{
  const s=await fixture();
  try {
    const input=publicOpening(s,'p1');input.public_lines.splice(-1,0,'|-ability|p2a: Mewtwo|Levitate');
    const result=await sampleWorlds(input,1027263300,{worlds:2,attempts:512});
    assert.equal(result.worlds.length,0);assert.equal(result.reason,'Hypothesis budget exhausted');
  }finally{s.close();}
});

test('non-opening requests and invalid budgets are refused',async()=>{
  const s=await fixture();
  try {
    const input=publicOpening(s,'p1');
    await assert.rejects(()=>sampleWorlds(input,-1,{worlds:2,attempts:2}));
    await assert.rejects(()=>sampleWorlds(input,1,{worlds:0,attempts:2}));
    await assert.rejects(()=>sampleWorlds(input,1,{worlds:2,attempts:513}));
    const frame=s.snapshot();
    await s.step({actions:{p1:0,p2:0},request_ids:{p1:frame.players.p1.request_id,p2:frame.players.p2.request_id}});
    assert.throws(()=>publicOpening(s,'p1'),/Opening only/);
  }finally{s.close();}
});

test('unsupported consumed own item fails exact reconstruction rather than restoring a secret item',async()=>{
  const s=await fixture();
  try {
    const input=publicOpening(s,'p1');
    input.public_lines.splice(-1,0,'|-enditem|p1a: Falinks|Booster Energy');
    input.request.side.pokemon[0].item='';input.player.own[0].item='';
    const result=await sampleWorlds(input,1027263300,{worlds:2,attempts:128});
    assert.equal(result.worlds.length,0);
  }finally{s.close();}
});

test('simultaneous joint submission refuses missing or stale requests',async()=>{
  const s=await fixture();
  try {
    const f=s.snapshot();
    await assert.rejects(()=>s.step({actions:{p1:0},request_ids:{p1:f.players.p1.request_id}}));
    await assert.rejects(()=>s.step({actions:{p1:0,p2:0},request_ids:{p1:0,p2:0}}));
    assert.equal(s.submissions,0);
  }finally{s.close();}
});

test('forced pivot remains an intermediate request, never a fake completed turn',async()=>{
  const teams=[generateTeam(seedFor('f33/pivot/a')).team,generateTeam(seedFor('f33/pivot/b')).team];
  teams[0][0]={name:'Mew',species:'Mew',ability:'Synchronize',level:100,
    moves:['uturn','protect','psychic','recover'],item:'',teraType:'Psychic'};
  teams[1][0]={name:'Blissey',species:'Blissey',ability:'Natural Cure',level:100,
    moves:['splash','softboiled','thunderwave','protect'],item:'',teraType:'Normal'};
  const s=await openWorld({teams,mechanics_seed:seedFor('f33/pivot/chance')});
  try {
    const f=s.snapshot();const after=await s.step({actions:{p1:0,p2:0},
      request_ids:{p1:f.players.p1.request_id,p2:f.players.p2.request_id}});
    assert.equal(after.players.p1.phase,'replace');assert.equal(after.players.p1.turn,1);
    assert.equal(after.players.p2.phase,'wait');
    const a=after.players.p1.mask.findIndex(Boolean);
    const next=await s.step({actions:{p1:a},request_ids:{p1:after.players.p1.request_id}});
    assert.equal(next.players.p1.turn,2);assert.equal(next.players.p2.phase,'move');
  }finally{s.close();}
});
