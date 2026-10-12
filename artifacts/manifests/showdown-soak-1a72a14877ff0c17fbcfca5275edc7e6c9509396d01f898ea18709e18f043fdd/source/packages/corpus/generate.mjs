import {readFileSync, writeFileSync, existsSync, mkdirSync, renameSync, unlinkSync, appendFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {gzipSync, gunzipSync} from 'node:zlib';
import {hostname} from 'node:os';
import {generateTeam} from '../reference-node/engine.mjs';
import {contractManifest, teamArtifact} from '../reference-node/contract-v1.mjs';
import {ROOT} from '../reference-node/provenance.mjs';
import {canonical, hashJSON} from '../battle-contract/canonical.mjs';
import {sourceCatalog} from './catalog.mjs';
import {SEED_VERSION, digest, seedAt, readJSON, seal, verifySeal, atomicJSON, shardName, validateIndex} from './common.mjs';

const producerFiles = ['packages/corpus/generate.mjs', 'packages/corpus/common.mjs', 'packages/corpus/catalog.mjs'];
const producerHashes = Object.fromEntries(producerFiles.map(p => [p, digest(readFileSync(`${ROOT}/${p}`))]));

function acquireLock(directory) {
  const lock = `${directory}/.writer-lock.json`;
  const ownerBytes = canonical({pid: process.pid, host: hostname()});
  try { writeFileSync(lock, ownerBytes, {flag: 'wx'}); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    // Serialize stale-lock recovery so competing restarts cannot unlink a new owner.
    // A crash during this short section leaves a conservative manual-recovery guard.
    const recovery = `${directory}/.writer-recovery.json`;
    writeFileSync(recovery, ownerBytes, {flag: 'wx'});
    try {
      if (existsSync(lock)) {
        const owner = readJSON(lock);
        if (owner.host !== hostname() || !Number.isSafeInteger(owner.pid) || owner.pid <= 0) throw new Error('Unrecognized or remote writer lock');
        try { process.kill(owner.pid, 0); throw new Error(`Corpus writer is active (pid ${owner.pid})`); }
        catch (error) { if (error.code !== 'ESRCH') throw error; }
        unlinkSync(lock); // Only a recorded local PID proven no longer alive.
      }
      writeFileSync(lock, ownerBytes, {flag: 'wx'});
    } finally { unlinkSync(recovery); }
  }
  return () => unlinkSync(lock);
}

export function checkChunk(directory, chunk) {
  const compressed = readFileSync(`${directory}/${chunk.file}`);
  if (compressed.length !== chunk.bytes || digest(compressed) !== chunk.sha256) throw new Error(`Shard integrity failure: ${chunk.file}`);
  const plain = gunzipSync(compressed);
  if (plain.length !== chunk.jsonl_bytes || digest(plain) !== chunk.jsonl_sha256) throw new Error(`JSONL integrity failure: ${chunk.file}`);
  return plain;
}

export function generateCorpus({out, namespace, count = 100000, shardSize = 1000, onProgress = () => {}}, generate = generateTeam) {
  if (!out || typeof namespace !== 'string' || !namespace || namespace.length > 200) throw new TypeError('Supply output directory and namespace');
  if (!Number.isSafeInteger(count) || count < 1 || count > 10000000 || !Number.isSafeInteger(shardSize) || shardSize < 1 || shardSize > 10000) throw new TypeError('Invalid count/shard size');
  const directory = resolve(out);
  const manifest = contractManifest(), catalog = sourceCatalog();
  const vocabularyBytes = readFileSync(`${ROOT}/contracts/battle-v1/vocabulary.json`);
  const config = {battle_manifest_id: manifest.manifest_id, namespace, count, shard_size: shardSize,
    seed_version: SEED_VERSION, split: 'unassigned', producer_hashes: producerHashes,
    catalog_hash: hashJSON(catalog), vocabulary_hash: manifest.payload.vocabulary_hash};
  const corpusId = hashJSON(config);
  mkdirSync(directory, {recursive: true});
  const unlock = acquireLock(directory);
  const started = Date.now();
  let progress, attemptedIndex = null, phase = 'initialize', recordingFailures = false;
  try {
    const statePath = `${directory}/progress.json`;
    if (existsSync(statePath)) {
      progress = verifySeal(readJSON(statePath));
      if (progress.corpus_id !== corpusId || canonical(progress.config) !== canonical(config)) throw new Error('Resume configuration/provenance differs');
      validateIndex(progress, {allowIncomplete: true});
    } else {
      if (existsSync(`${directory}/index.json`) || existsSync(`${directory}/manifest.json`)) throw new Error('Existing corpus without progress metadata');
      progress = seal({corpus_version: 'r3-corpus-1', corpus_id: corpusId, config, status: 'generating', completed: 0, chunks: []});
      writeFileSync(`${directory}/manifest.json`, canonical(manifest) + '\n', {flag: 'wx'});
      writeFileSync(`${directory}/catalog.json`, canonical(catalog) + '\n', {flag: 'wx'});
      writeFileSync(`${directory}/vocabulary.json`, vocabularyBytes, {flag: 'wx'});
      atomicJSON(statePath, progress);
    }
    if (hashJSON(readJSON(`${directory}/catalog.json`)) !== config.catalog_hash ||
        hashJSON(readJSON(`${directory}/vocabulary.json`)) !== config.vocabulary_hash ||
        canonical(readJSON(`${directory}/manifest.json`)) !== canonical(manifest)) throw new Error('Corpus metadata changed');
    for (const chunk of progress.chunks) checkChunk(directory, chunk);
    if (existsSync(`${directory}/index.json`)) {
      const complete = validateIndex(readJSON(`${directory}/index.json`));
      if (complete.corpus_id !== corpusId || canonical(complete.chunks) !== canonical(progress.chunks)) throw new Error('Completion index disagrees with progress');
      return complete;
    }
    recordingFailures = true;
    while (progress.completed < count) {
      // Each shard is a transaction: no partial rows enter the committed prefix.
      attemptedIndex = null;
      phase = 'verify_provenance';
      for (const [p, expected] of Object.entries(producerHashes)) if (digest(readFileSync(`${ROOT}/${p}`)) !== expected) throw new Error('Producer source changed; restart explicitly');
      if (contractManifest().manifest_id !== manifest.manifest_id) throw new Error('Reference provenance changed');
      const start = progress.completed, size = Math.min(shardSize, count - start);
      const lines = [];
      phase = 'generate';
      for (let offset = 0; offset < size; offset++) {
        attemptedIndex = start + offset;
        const seed = seedAt(namespace, attemptedIndex), entry = generate(seed);
        if (entry.seed !== seed) throw new Error('Generator returned a different seed');
        lines.push(canonical(teamArtifact(entry, manifest.manifest_id)) + '\n');
      }
      attemptedIndex = null;
      phase = 'write_shard';
      const plain = Buffer.from(lines.join('')), compressed = gzipSync(plain, {level: 6});
      const file = shardName(progress.chunks.length);
      if (existsSync(`${directory}/${file}`)) {
        // Crash after file rename but before checkpoint: preserve, then regenerate.
        mkdirSync(`${directory}/orphans`, {recursive: true});
        renameSync(`${directory}/${file}`, `${directory}/orphans/${file}-${Date.now()}`);
      }
      const temp = `${directory}/${file}.part-${process.pid}`;
      writeFileSync(temp, compressed, {flag: 'wx'});
      renameSync(temp, `${directory}/${file}`);
      const chunk = {file, start, count: size, bytes: compressed.length, sha256: digest(compressed),
        jsonl_bytes: plain.length, jsonl_sha256: digest(plain)};
      const nextProgress = seal({corpus_version: 'r3-corpus-1', corpus_id: corpusId, config, status: 'generating',
        completed: start + size, chunks: [...progress.chunks, chunk]});
      phase = 'checkpoint';
      atomicJSON(statePath, nextProgress);
      progress = nextProgress;
      phase = 'notify_progress';
      onProgress({completed: progress.completed, requested: count, elapsed_ms: Date.now() - started});
    }
    phase = 'publish_index';
    const failures = existsSync(`${directory}/failures.jsonl`) ? readFileSync(`${directory}/failures.jsonl`) : Buffer.alloc(0);
    const result = seal({corpus_version: 'r3-corpus-1', corpus_id: corpusId, config, status: 'complete',
      completed: count, chunks: progress.chunks, failure_events: failures.length ? failures.toString('utf8').trim().split('\n').length : 0,
      failures_sha256: digest(failures)});
    validateIndex(result);
    atomicJSON(`${directory}/index.json`, result);
    return result;
  } catch (error) {
    if (recordingFailures) appendFileSync(`${directory}/failures.jsonl`, canonical({time: new Date().toISOString(), corpus_id: corpusId,
      attempted_index: attemptedIndex, seed: attemptedIndex === null ? null : seedAt(namespace, attemptedIndex),
      committed_records: progress?.completed ?? 0, phase, message: error.message}) + '\n');
    throw error;
  } finally { unlock(); }
}
