import test from 'node:test';
import assert from 'node:assert/strict';
import {BattleStream, Dex, seedFor} from '../../packages/reference-node/engine.mjs';
import {PlayerProjection} from '../../packages/reference-node/observations.mjs';
import {DynamicRosterView} from '../../packages/dynamic-matchup/view.mjs';
import {DynamicMatchupSession} from '../../packages/dynamic-matchup/session.mjs';

const mon = (species, moves, ability) => ({species, name: species, moves, ability,
  level: 100, item: '', teraType: 'Water'});
const bench = () => ['Snorlax','Vaporeon','Umbreon','Espeon','Raichu'].map(s => mon(s, ['splash'], Dex.species.get(s).abilities[0]));

async function fixture(a, b, reserves = bench()) {
  const projection = new PlayerProjection();
  const views = Object.fromEntries(['p1','p2'].map(side => [side,new DynamicRosterView(side)]));
  const cursors = {p1:0,p2:0};
  const stream = new class extends BattleStream {
    constructor() { super({noCatch:true}); }
    pushMessage(type, data) { projection.ingest(type, data); }
  }();
  const frames = () => Object.fromEntries(['p1','p2'].map(side => {
    const p = projection.players[side];
    while (cursors[side] < p.protocol.length) {
      const line = p.protocol[cursors[side]++];
      if (!line.startsWith('|request|')) views[side].ingest(line);
    }
    return [side,views[side].withOwnTypes(views[side].observe(p.request,p.request_id))];
  }));
  await stream.write(`>start ${JSON.stringify({formatid:'gen9anythinggoes@@@!Team Preview',seed:seedFor('f24-types-fixture')})}\n`+
    `>player p1 ${JSON.stringify({name:'A',team:[a,...reserves]})}\n`+
    `>player p2 ${JSON.stringify({name:'B',team:[b,...bench()]})}`);
  frames();
  return {stream, projection, frames, async step(a,b) {
    await stream.write(`>p1 ${a}\n>p2 ${b}`); assert.deepEqual(projection.errors, []); return frames();
  }};
}

function currentTypes(player, side) {
  const m = (side === player.side ? player.own : player.opponent).find(m => m?.active);
  return m.tera && m.tera !== 'Stellar' ? [m.tera] : m.types ?? Dex.species.get(m.species).types;
}

test('pinned Double Shock, both player views, bench reset and reentry', async () => {
  const f = await fixture(mon('Pawmot',['doubleshock','splash'],'Natural Cure'),mon('Regirock',['splash'],'Clear Body'));
  try {
    let p = await f.step('move 1','move 1');
    assert.deepEqual(f.stream.battle.p1.active[0].getTypes(),['???','Fighting']);
    for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),['???','Fighting']);
    p = await f.step('switch 2','move 1');
    assert.equal(p.p1.own.find(m=>m.ident==='Pawmot').types,null);
    assert.equal(p.p2.opponent.find(m=>m?.ident==='Pawmot').types,null);
    p = await f.step('switch 2','move 1');
    for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),f.stream.battle.p1.active[0].getTypes());
  } finally { f.stream.battle.destroy(); }
});

test('pinned Burn Up creates known monotype typeless state', async () => {
  const f = await fixture(mon('Arcanine',['burnup','splash'],'Intimidate'),mon('Chansey',['splash'],'Natural Cure'));
  try {
    const p = await f.step('move 1','move 1');
    assert.deepEqual(f.stream.battle.p1.active[0].getTypes(),['???']);
    for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),['???']);
  } finally { f.stream.battle.destroy(); }
});

test('pinned added third type, replacement by second typeadd, then Soak reset', async () => {
  const f = await fixture(mon('Mew',['forestscurse','trickortreat','soak'],'Synchronize'),mon('Dragonite',['splash'],'Inner Focus'));
  try {
    for (const [move, expected] of [[1,['Dragon','Flying','Grass']],[2,['Dragon','Flying','Ghost']],[3,['Water']]]) {
      const p = await f.step(`move ${move}`,'move 1');
      assert.deepEqual(f.stream.battle.p2.active[0].getTypes(),expected);
      for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p2'),expected);
    }
  } finally { f.stream.battle.destroy(); }
});

test('Tera defense survives switching while ordinary temporary type resets', async () => {
  const f = await fixture(mon('Pawmot',['doubleshock','splash'],'Natural Cure'),mon('Regirock',['splash'],'Clear Body'));
  try {
    await f.step('move 1','move 1');
    let p = await f.step('move 2 terastallize','move 1');
    for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),['Water']);
    await f.step('switch 2','move 1');
    p = await f.step('switch 2','move 1');
    assert.deepEqual(f.stream.battle.p1.active[0].getTypes(),['Water']);
    assert.equal(p.p1.own.find(m=>m.ident==='Pawmot').types,null);
    for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),['Water']);
  } finally { f.stream.battle.destroy(); }
});

test('production session v2 is player-only and keeps final split forbidden', async () => {
  const s = new DynamicMatchupSession();
  try {
    await assert.rejects(s.start({episodeIndex:0,runSeed:73,split:'final'}),/never admitted/);
    const f = await s.start({episodeIndex:0,runSeed:1025262497,split:'train'});
    assert.equal(f.schema_version,'dynamic-matchup-frame-v1');
    for (const side of ['p1','p2']) {
      assert.equal(f.players[side].schema_version,'full-showdown-player-v2');
      assert(f.players[side].own.every(m=>Object.hasOwn(m,'types')));
      assert(f.players[side].opponent.filter(Boolean).every(m=>!Object.hasOwn(m,'stats')));
      assert(!Object.hasOwn(f.players[side],'provenance'));
      assert.equal(f.players[side].mask.length,14);
    }
  } finally { s.close(); }
});

test('pinned Reflect Type copies Tera defense; Transform copies pre-Tera ordinary types', async () => {
  for (const move of ['reflecttype','transform']) {
    const f = await fixture(mon('Mew',[move],'Synchronize'),mon('Dragonite',['splash'],'Inner Focus'));
    try {
      const p = await f.step('move 1','move 1 terastallize');
      const expected = move === 'reflecttype' ? ['Water'] : ['Dragon','Flying'];
      assert.deepEqual(f.stream.battle.p1.active[0].getTypes(),expected);
      for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),expected);
    } finally { f.stream.battle.destroy(); }
  }
});

test('own Illusion uses known own species without contaminating bench or public foe', () => {
  const own = new DynamicRosterView('p1'), foe = new DynamicRosterView('p2');
  for (const view of [own,foe]) {
    view.ingest('|switch|p1a: Pikachu|Pikachu, L100|100/100');
    view.ingest("|-start|p1a: Pikachu|typeadd|Grass|[from] move: Forest's Curse");
  }
  const actual = own.withOwnTypes({own:[{ident:'Zoroark',species:'Zoroark',active:true},
    {ident:'Pikachu',species:'Pikachu',active:false}]}).own;
  assert.deepEqual(actual[0].types,['Dark','Grass']);
  assert.equal(actual[1].types,null);
  assert.deepEqual(foe.publicTypes.typesFor('p1a: Pikachu','Pikachu'),['Electric','Grass']);
});

test('pinned temporary Forecast forme is exposed even when own request keeps base species', async () => {
  const f = await fixture(mon('Castform',['sunnyday','splash'],'Forecast'),mon('Regirock',['splash'],'Clear Body'));
  try {
    const p = await f.step('move 1','move 1');
    assert.equal(p.p1.own.find(m=>m.active).species,'Castform');
    assert.deepEqual(f.stream.battle.p1.active[0].getTypes(),['Fire']);
    for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),['Fire']);
  } finally { f.stream.battle.destroy(); }
});

test('pinned Mimicry terrain-end clears both changed and added types', async () => {
  const f = await fixture(mon('Stunfisk-Galar',['electricterrain','splash'],'Mimicry'),
    mon('Mew',['forestscurse','splash'],'Synchronize'));
  try {
    await f.step('move 1','move 2');
    let p = await f.step('move 2','move 1');
    assert.deepEqual(f.stream.battle.p1.active[0].getTypes(),['Electric','Grass']);
    for (let i=0;i<3;i++) p=await f.step('move 2','move 2');
    assert.deepEqual(f.stream.battle.p1.active[0].getTypes(),['Ground','Steel']);
    for (const side of ['p1','p2']) assert.deepEqual(currentTypes(p[side],'p1'),['Ground','Steel']);
  } finally { f.stream.battle.destroy(); }
});

test('pinned Illusion entrant does not inherit its namesake Tera from a prior appearance', async () => {
  const reserves=[...bench().slice(0,4),mon('Pikachu',['splash'],'Static')];
  const f=await fixture(mon('Zoroark',['splash'],'Illusion'),mon('Regirock',['splash'],'Clear Body'),reserves);
  try {
    await f.step('switch 6','move 1');
    let p=await f.step('move 1 terastallize','move 1');
    assert.equal(p.p2.opponent.find(m=>m?.active).tera,'Water');
    p=await f.step('switch 6','move 1');
    assert.equal(f.stream.battle.p1.active[0].species.name,'Zoroark');
    assert.equal(Boolean(f.stream.battle.p1.active[0].terastallized),false);
    assert.equal(p.p2.opponent.find(m=>m?.active).species,'Pikachu');
    assert.equal(p.p2.opponent.find(m=>m?.active).tera,null);
    assert.equal(p.p1.own.find(m=>m.active).tera,null);
    assert.equal(p.p1.own.find(m=>m.ident==='Pikachu').tera,'Water');
  } finally { f.stream.battle.destroy(); }
});
