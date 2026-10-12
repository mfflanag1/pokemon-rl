import {readFileSync, writeFileSync} from 'node:fs';
import {Dex, describe} from '../packages/reference-node/engine.mjs';
import {canonical, hashJSON, parseJSON} from '../packages/battle-contract/canonical.mjs';
import {fileHash} from '../packages/reference-node/provenance.mjs';
const vocabulary = parseJSON(readFileSync('contracts/battle-v1/vocabulary.json'));
const names = {};
for (const [domain, ids] of Object.entries(vocabulary.domains)) {
  names[domain] = {};
  for (const id of ids.slice(2)) {
    const entry = Dex[domain].get(id);
    if (!entry.exists || entry.id !== id) throw new Error(`Unresolved ${domain}:${id}`);
    for (const key of [id, entry.name.toLowerCase()]) {
      if (names[domain][key] && names[domain][key] !== id) throw new Error(`Ambiguous ${domain}:${key}`);
      names[domain][key] = id;
    }
  }
}
// Engine-advertised control move; not a Dex move and not an extra policy action.
const domains = structuredClone(vocabulary.domains);
domains.moves.push('recharge'); names.moves.recharge = 'recharge';
// Public announcement shared by both As One variants; never guess the hidden variant.
domains.abilities.push('asone'); names.abilities['as one'] = 'asone'; names.abilities.asone = 'asone';
const body = {request_only_moves: ['recharge'], public_ability_labels: ['asone'], version: 'r5-name-catalog-1', showdown_commit: describe().commit,
  vocabulary_hash: hashJSON(vocabulary), domains, names,
  producer_sha256: fileHash(new URL(import.meta.url))};
writeFileSync('contracts/player-features-v1/catalog.json', canonical({...body, content_hash: hashJSON(body)}) + '\n');
console.log(JSON.stringify({content_hash: hashJSON(body), domains: Object.fromEntries(Object.entries(names).map(([k,v])=>[k,Object.keys(v).length]))}));
