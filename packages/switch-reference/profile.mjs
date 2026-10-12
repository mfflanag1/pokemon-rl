import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {Dex,describe} from '../reference-node/engine.mjs';
import {hashJSON} from '../battle-contract/canonical.mjs';
export const PROFILE=JSON.parse(readFileSync(new URL('../../contracts/switch-v1/profile.json',import.meta.url)));
export const SIDES=['p1','p2'];
const ids=PROFILE.species.map(s=>s.id),variants=Object.keys(PROFILE.variants);
export const ROSTER=[];
for(const lead of ids)for(const reserve of ids)if(lead!==reserve)for(const a of variants)for(const b of variants)
  ROSTER.push({id:`${lead}-${a}_${reserve}-${b}`,species:[lead,reserve],variants:[a,b]});
export function team(id) {
  const row=ROSTER.find(r=>r.id===id);assert.ok(row,'Unknown profile team');
  return row.species.map((s,i)=>{const p=PROFILE.species.find(p=>p.id===s);return {
    name:PROFILE.names[i],species:p.species,ability:p.ability,moves:[...p.moves],teraType:p.teraType,
    gender:PROFILE.gender,item:PROFILE.item,level:PROFILE.level,nature:PROFILE.nature,
    evs:{...PROFILE.variants[row.variants[i]]},ivs:{...PROFILE.ivs}};});
}
export function reference() {
  const manifest=describe();assert.equal(manifest.commit,PROFILE.showdown_commit);
  const format=Dex.formats.get(PROFILE.reference_format),rules=Dex.formats.getRuleTable(format);
  assert.ok(!format.debug);assert.ok(!rules.has('teampreview'));assert.ok(rules.has('terastalclause'));
  return {reference:manifest,profile_hash:hashJSON(PROFILE),format:{id:format.id,mod:format.mod,
    rules:[...rules.entries()].sort(),values:[...rules.valueRules.entries()].sort()},roster:ROSTER};
}
