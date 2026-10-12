import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync, renameSync} from 'node:fs';
import {canonical, hashJSON, parseJSON} from '../battle-contract/canonical.mjs';

export const SEED_VERSION = 'r3-team-seeds-v1';
export const digest = bytes => createHash('sha256').update(bytes).digest('hex');
export const readJSON = path => parseJSON(readFileSync(path));
export function seedAt(namespace, index) {
  if (typeof namespace !== 'string' || !namespace || !Number.isSafeInteger(index) || index < 0) throw new TypeError('Invalid seed namespace/index');
  return `sodium,${hashJSON([SEED_VERSION, namespace, index])}`;
}
export function seal(body) { return {...body, content_hash: hashJSON(body)}; }
export function verifySeal(document) {
  const {content_hash, ...body} = document;
  if (hashJSON(body) !== content_hash) throw new TypeError('Corpus metadata hash mismatch');
  return document;
}
export function atomicJSON(path, document) {
  const temp = `${path}.tmp-${process.pid}`;
  writeFileSync(temp, canonical(document) + '\n', {flag: 'wx'});
  renameSync(temp, path);
}
export const shardName = number => `teams-${String(number).padStart(6, '0')}.jsonl.gz`;

export function validateIndex(index, {allowIncomplete = false} = {}) {
  verifySeal(index);
  if (index.corpus_version !== 'r3-corpus-1' || (index.status !== 'complete' && !(allowIncomplete && index.status === 'generating'))) throw new TypeError('Not a recognized R3 corpus');
  const {config, chunks} = index;
  if (hashJSON(config) !== index.corpus_id || config.seed_version !== SEED_VERSION || config.split !== 'unassigned') throw new TypeError('Invalid corpus identity');
  if (!Number.isSafeInteger(config.count) || config.count < 1 || !Number.isSafeInteger(config.shard_size) || config.shard_size < 1) throw new TypeError('Invalid corpus size');
  let next = 0;
  for (const [i, chunk] of chunks.entries()) {
    if (chunk.file !== shardName(i) || chunk.start !== next || chunk.count !== Math.min(config.shard_size, config.count - next) || chunk.count < 1) {
      throw new TypeError('Missing, reordered or malformed shard');
    }
    for (const key of ['sha256', 'jsonl_sha256']) if (!/^[0-9a-f]{64}$(?![\s\S])/.test(chunk[key])) throw new TypeError('Invalid shard digest');
    next += chunk.count;
  }
  if (next !== index.completed || next > config.count || (!allowIncomplete && next !== config.count)) throw new TypeError('Incomplete corpus');
  return index;
}
