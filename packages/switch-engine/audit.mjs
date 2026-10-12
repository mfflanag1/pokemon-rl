// Read-only fresh replay of passivity controls and current pinned oracle admission.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {checkUnpatched,PLAN} from './reference.mjs';
import {reference} from '../switch-reference/profile.mjs';
import {fileHash} from '../reference-node/provenance.mjs';
const directory=process.argv[2],read=name=>JSON.parse(readFileSync(`${directory}/${name}`));
const plan=read('plan.json'),report=read('report.json');
assert.deepEqual(plan.plan,PLAN);assert.deepEqual(plan.reference,reference());
assert.equal(report.plan_sha256,fileHash(`${directory}/plan.json`));
const first=report.shards[0];assert.equal(first.file,'cases-0000.json.gz');
assert.equal(fileHash(`${directory}/${first.file}`),first.sha256);
const rows=JSON.parse(gunzipSync(readFileSync(`${directory}/${first.file}`)));
const count=Math.min(plan.count,PLAN.passivity_games);assert(rows.length>=count);
for(let i=0;i<count;i++){assert.equal(rows[i].id,i);await checkUnpatched(rows[i]);}
assert.deepEqual(plan.reference,reference());
console.log(JSON.stringify({status:'pass',unpatched_games:count,plan_sha256:fileHash(`${directory}/plan.json`),report_sha256:fileHash(`${directory}/report.json`)}));
