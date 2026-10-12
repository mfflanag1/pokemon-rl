import {readFileSync} from 'node:fs';
import {ROOT, fileHash} from '../reference-node/provenance.mjs';
import {contractManifest} from '../reference-node/contract-v1.mjs';
import {hashJSON, parseJSON} from '../battle-contract/canonical.mjs';

const files = ['phases.mjs', 'session.mjs', 'fixtures.mjs', 'manifest.mjs', 'contract.mjs'].map(p => `packages/episodes/${p}`);
files.push('contracts/episode-v1/schema.json');
const loaded = Object.fromEntries(files.map(p => [p, fileHash(`${ROOT}/${p}`)]));

export function episodeManifest(profile = 'official') {
  if (!['official', 'constructed'].includes(profile)) throw new Error('Unknown R4 profile');
  for (const [p, hash] of Object.entries(loaded)) if (fileHash(`${ROOT}/${p}`) !== hash) throw new Error('R4 source changed; restart worker');
  const payload = {version: 'r4-manifest-1', profile, phase_version: 'r4-phases-1',
    audit_version: 'r4-selected-mechanics-1', reference_contract: contractManifest(), source_hashes: loaded};
  return {...payload, manifest_id: hashJSON(payload)};
}

export const readEpisode = path => parseJSON(readFileSync(path));
