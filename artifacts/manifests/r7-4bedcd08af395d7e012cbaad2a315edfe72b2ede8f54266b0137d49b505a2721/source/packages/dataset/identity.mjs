import {readFileSync} from 'node:fs';
import {hashJSON, canonical} from '../battle-contract/canonical.mjs';
import {validateSchema} from '../battle-contract/schema.mjs';

export const POLICY = JSON.parse(readFileSync(new URL('../../contracts/dataset-v1/policy.json', import.meta.url)));
const expectedRanges = [{split: 'train', lower: 0, upper: 9000}, {split: 'development', lower: 9000, upper: 9500}, {split: 'final', lower: 9500, upper: 10000}];
if (POLICY.schema_version !== 'r7-split-policy-1' || POLICY.identity_version !== 'r7-ordered-team-1' ||
    POLICY.hash_version !== 'r7-team-partition-1' || POLICY.denominator !== 10000 ||
    POLICY.profile !== 'gen9randombattle' || POLICY.showdown_commit !== '942cda71d14bf0618bd57991708a2e2734b6daba' ||
    typeof POLICY.salt !== 'string' || !POLICY.salt || canonical(POLICY.ranges) !== canonical(expectedRanges)) throw new Error('Invalid frozen R7 policy');
function freeze(value) { if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); } }
freeze(POLICY);
export const POLICY_HASH = hashJSON(POLICY);
export function canonicalTeam(payload) {
  validateSchema(payload, 'team');
  return payload.pokemon.map((p, i) => ({species: payload.ids[i].species, ability: payload.ids[i].ability,
    item: payload.ids[i].item, moves: [...payload.ids[i].moves], tera_type: payload.ids[i].tera_type,
    level: p.level, gender: p.gender, evs: {...p.evs}, ivs: {...p.ivs}}));
}
export function teamKey(payload) { return hashJSON([POLICY.identity_version, canonicalTeam(payload)]); }
export function partition(payload) {
  return partitionDigest(hashJSON([POLICY.hash_version, POLICY.salt, canonicalTeam(payload)]));
}
export function partitionDigest(hex) {
  if (!/^[0-9a-f]{64}$/.test(hex)) throw new Error('Expected full partition digest');
  const value = BigInt(`0x${hex}`);
  for (const r of POLICY.ranges) if (value * 10000n >= BigInt(r.lower) * (1n << 256n) && value * 10000n < BigInt(r.upper) * (1n << 256n)) return r.split;
  throw new Error('Partition range incomplete');
}
export function pairPartition(a, b) {
  const one = partition(a), two = partition(b);
  if (one !== two) throw new Error('Cross-partition matchup');
  return one;
}
export function requirePolicy(policy) {
  if (canonical(policy) !== canonical(POLICY)) throw new Error('Frozen R7 split policy differs');
}
