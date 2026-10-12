import assert from 'node:assert/strict';
import {writeFileSync,readFileSync} from 'node:fs';
import path from 'node:path';
import {PRNG,seedFor,describe} from '../reference-node/engine.mjs';
import {ROOT,fileHash} from '../reference-node/provenance.mjs';
import {canonical,hashJSON} from '../battle-contract/canonical.mjs';

export const CONTRACT=JSON.parse(readFileSync(path.join(ROOT,'contracts/showdown-chance-v1/contract.json')));
export function boundaryWords(){
  const words=new Set([0,1,0xffffffff]);
  for(const n of [24,16,2])for(let k=1;k<n;k++){
    const edge=Number((BigInt(k)*2n**32n+BigInt(n)-1n)/BigInt(n));
    for(const w of [edge-1,edge,edge+1])words.add(w);
  }
  return [...words].sort((a,b)=>a-b);
}
export function wordCase(word){
  assert.ok(Number.isInteger(word)&&word>=0&&word<2**32);
  const prng=new PRNG(seedFor('m1-showdown-word-oracle'));
  let calls=0;prng.rng.next=()=>{calls++;return word;};
  const bucket24=prng.random(24),crit=prng.randomChance(1,24),roll=100-prng.random(16),ties=[];
  for(const order of [[0,1],[1,0]]){prng.shuffle(order);ties.push(order[0]);}
  assert.equal(calls,5);
  return {word,bucket24,crit,roll,ties};
}
export function audit(out,randomWords=CONTRACT.word_audit.random_words){
  assert.ok(Number.isInteger(randomWords)&&randomWords>=0);
  const reference=describe(),rng=new PRNG(seedFor(`m1-showdown-word-probes/${CONTRACT.word_audit.seed}`));
  const words=boundaryWords();for(let i=0;i<randomWords;i++)words.push(rng.rng.next());
  const rows=words.map(wordCase);
  const sources=Object.fromEntries(['packages/showdown-chance/word-oracle.mjs','contracts/showdown-chance-v1/contract.json'].map(p=>[p,fileHash(path.join(ROOT,p))]));
  assert.deepEqual(describe(),reference);
  const body={version:'showdown-word-oracle-1',reference,sources,random_words:randomWords,boundary_words:boundaryWords().length,rows};
  writeFileSync(out,canonical({...body,content_hash:hashJSON(body)})+'\n',{flag:'wx'});
}
if(process.argv[1]===new URL(import.meta.url).pathname)audit(process.argv[2],process.argv[3]===undefined?undefined:Number(process.argv[3]));
