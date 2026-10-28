import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {PLAN,checkUnpatched} from './reference.mjs';
import {reference,ROSTER} from './profile.mjs';
import {fileHash} from '../reference-node/provenance.mjs';
import {seedFor} from '../reference-node/engine.mjs';
const directory=process.argv[2],read=name=>JSON.parse(readFileSync(`${directory}/${name}`));
const plan=read('plan.json'),report=read('report.json');
assert.deepEqual(plan.plan,PLAN);assert.deepEqual(plan.reference,reference());
assert.equal(report.status,'complete');assert.equal(report.games,PLAN.fresh_cases);
assert.equal(report.plan_sha256,fileHash(`${directory}/plan.json`));
assert.equal(report.file,'cases.json.gz');assert.equal(report.sha256,fileHash(`${directory}/${report.file}`));
const rows=JSON.parse(gunzipSync(readFileSync(`${directory}/${report.file}`)));assert.equal(rows.length,PLAN.fresh_cases);
for(const [id,row] of rows.entries()){
  assert.equal(row.id,id);const teams=[Math.floor(id/6),(Math.floor(id/6)*7+9)%24];
  assert.deepEqual(row.teams,teams);
  assert.deepEqual(row.spec,{teams:teams.map(i=>ROSTER[i].id),seed:seedFor(`${PLAN.namespace}/fresh/${id}`)});
  await checkUnpatched(row);
}
assert.deepEqual(plan.reference,reference());
console.log(JSON.stringify({status:'passed',unpatched_reference_games:rows.length}));
