import test from 'node:test';
import assert from 'node:assert/strict';
import {describeDecision, classifyMove} from '../../packages/battle-audit/describe.mjs';
import {TacticalView} from '../../packages/showdown-panel/view.mjs';
import {requestActions} from '../../packages/full-showdown/actions.mjs';

function fixture(moves=['thunderbolt','surf','stealthrock','calmmind']) {
  const view=new TacticalView();
  view.ingest('|switch|p1a: Raichu|Raichu, L100|200/200');
  view.ingest('|switch|p2a: Golem|Golem, L100|100/100');
  view.ingest('|turn|1');
  const own={ident:'p1: Raichu',details:'Raichu, L100',condition:'200/200',active:true,
    stats:{atk:200,def:150,spa:300,spd:150,spe:300},moves,baseAbility:'static',ability:'static',item:'',teraType:'Electric'};
  const bench={...own,ident:'p1: Gyarados',details:'Gyarados, L100',active:false,moves:['waterfall']};
  const request={side:{id:'p1',pokemon:[own,bench]},active:[{moves:moves.map(id=>({id,move:id,pp:16,maxpp:16,disabled:false})),canTerastallize:'Electric'}]};
  const player={schema_version:'full-showdown-player-v1',side:'p1',phase:'move',mask:requestActions(request).mask,side_conditions:{own:{},opponent:{}}};
  return {view,request,player};
}

test('immune attack gap, overlapping public tags and input immutability',()=>{
  const {request,view,player}=fixture(); const before=JSON.stringify({request,view,player});
  const d=describeDecision(request,view,player,0);
  assert.equal(d.selected_move,'thunderbolt'); assert.equal(d.selected_damage_proxy,0);
  assert.ok(d.best_legal_damage_proxy>.2); assert.ok(d.tags.includes('damage_choice'));
  assert.ok(d.tags.includes('hidden_information')); assert.ok(d.disagreement);
  assert.equal(JSON.stringify({request,view,player}),before);
  view.ingest('|move|p2a: Golem|Earthquake|p1a: Raichu');
  assert.ok(!describeDecision(request,view,player,0).tags.includes('hidden_information'));
});

test('hazards, setup, voluntary switch and tera timing describe selected decisions',()=>{
  const {request,view,player}=fixture();
  assert.equal(describeDecision(request,view,player,2).selected_family,'hazard');
  assert.ok(describeDecision(request,view,player,2).tags.includes('hazards'));
  assert.ok(describeDecision(request,view,player,3).tags.includes('status_setup_recovery'));
  view.ingest('|-sidestart|p1: Player|move: Stealth Rock');
  const sw=describeDecision(request,view,player,9);
  assert.ok(sw.tags.includes('voluntary_switch')); assert.ok(sw.tags.includes('hazards'));
  const tera=describeDecision(request,view,player,4);
  assert.equal(tera.selected_tera,true); assert.ok(tera.tags.includes('tera_timing'));
});

test('forced switch, Recharge and Struggle are handled without fictitious move slots',()=>{
  const {request,view,player}=fixture();
  delete request.active; request.forceSwitch=[true]; request.side.pokemon[0].condition='0 fnt';
  player.phase='replace'; player.mask=requestActions(request).mask;
  assert.ok(describeDecision(request,view,player,9).tags.includes('forced_replacement'));
  for (const id of ['recharge','struggle']) {
    const f=fixture([id]);
    const d=describeDecision(f.request,f.view,f.player,0);
    assert.equal(d.selected_family,'control'); assert.equal(d.selected_damage_proxy,null);
  }
});

test('wrong seat, masked action and explicit private opponent fields are rejected',()=>{
  const f=fixture();
  assert.throws(()=>describeDecision(f.request,f.view,{...f.player,side:'p2'},0),/seat/);
  assert.throws(()=>describeDecision(f.request,f.view,f.player,8),/mask/);
  f.view.active.p2.stats={atk:999};
  assert.throws(()=>describeDecision(f.request,f.view,f.player,0),/Private opponent/);
});

test('semantic classification includes clearing, pivoting, protection and recovery',()=>{
  for (const [id,family] of [['rapidspin','hazard_clear'],['uturn','pivot'],['protect','protect'],
    ['recover','recovery'],['toxic','status'],['swordsdance','setup']]) assert.equal(classifyMove(id),family);
});
