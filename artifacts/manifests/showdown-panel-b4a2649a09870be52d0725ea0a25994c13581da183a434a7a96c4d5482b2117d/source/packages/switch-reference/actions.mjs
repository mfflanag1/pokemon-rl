import assert from 'node:assert/strict';
import {PROFILE} from './profile.mjs';
const id=s=>s.toLowerCase().replace(/[^a-z0-9]/g,'');
function object(value,keys){assert.ok(value&&typeof value==='object'&&!Array.isArray(value));assert.ok(Object.keys(value).every(k=>keys.includes(k)),'Unknown request fields');}
const monKeys=['ident','details','condition','active','stats','moves','baseAbility','item','pokeball','ability','commanding','reviving','teraType','terastallized'];
export function mask(request) {
  object(request,['wait','side','active','forceSwitch','rqid','noCancel']);
  if('wait' in request){assert.equal(request.wait,true);assert.ok(!('active' in request)&&!('forceSwitch' in request));if(!('side' in request))return [false,false,false,false];}
  const side=request.side;object(side,['id','name','pokemon']);assert.ok(['p1','p2'].includes(side.id));
  assert.ok(Array.isArray(side.pokemon)&&side.pokemon.length===2);
  const names=side.pokemon.map(m=>m.ident);assert.deepEqual([...names].sort(),PROFILE.names.map(n=>`${side.id}: ${n}`));
  assert.equal(side.pokemon.filter(m=>m.active===true).length,1);
  assert.equal(new Set(side.pokemon.map(m=>m.details)).size,2);
  for(const m of side.pokemon){
    object(m,monKeys);assert.equal(typeof m.active,'boolean');assert.ok(!m.reviving&&!m.commanding&&!m.terastallized);
    const p=PROFILE.species.find(p=>m.details===`${p.species}, L50, M`);assert.ok(p,'Unsupported member');
    assert.deepEqual(m.moves,p.moves.map(id));assert.equal(m.item,'');assert.equal(m.ability,id(p.ability));assert.equal(m.baseAbility,id(p.ability));
    assert.equal(typeof m.condition,'string');const h=/^([0-9]+)\/([0-9]+)$/.exec(m.condition);
    assert.ok(m.condition==='0 fnt'||h&&Number(h[1])>0&&Number(h[1])<=Number(h[2]));
    object(m.stats,['atk','def','spa','spd','spe']);assert.equal(Object.keys(m.stats).length,5);assert.ok(Object.values(m.stats).every(n=>Number.isSafeInteger(n)&&n>0));
  }
  if(request.wait)return [false,false,false,false];
  const result=[false,false,false,false];
  for(let i=0;i<2;i++){const mon=side.pokemon.find(m=>m.ident===`${side.id}: ${PROFILE.names[i]}`);
    result[2+i]=!mon.active&&mon.condition!=='0 fnt';}
  if('forceSwitch' in request){assert.deepEqual(request.forceSwitch,[true]);assert.ok(!('active' in request)&&result.some(Boolean));return result;}
  assert.ok(Array.isArray(request.active)&&request.active.length===1);
  const active=request.active[0];object(active,['moves']);assert.ok(Array.isArray(active.moves)&&active.moves.length===2);
  const mon=side.pokemon.find(m=>m.active);assert.notEqual(mon.condition,'0 fnt');
  assert.deepEqual(active.moves.map(m=>m.id),mon.moves);
  for(let i=0;i<2;i++){
    const m=active.moves[i];object(m,['move','id','pp','maxpp','target','disabled']);assert.equal(typeof m.move,'string');assert.equal(id(m.move),m.id);
    assert.ok(Number.isSafeInteger(m.pp)&&Number.isSafeInteger(m.maxpp)&&m.pp>=0&&m.pp<=m.maxpp&&m.maxpp>0);
    assert.equal(typeof m.disabled,'boolean');result[i]=!m.disabled&&m.pp>0;
  }
  assert.ok(result.slice(0,2).some(Boolean),'PP exhaustion is out of scope');return result;
}
export function command(request,index) {
  assert.ok(Number.isInteger(index)&&index>=0&&index<4,'Action index');assert.ok(mask(request)[index],'Illegal action');
  return index<2?`move ${index+1}`:`switch ${request.side.pokemon.findIndex(m=>m.ident===`${request.side.id}: ${PROFILE.names[index-2]}`)+1}`;
}
