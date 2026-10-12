import {fork} from 'node:child_process';
import {readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {gunzipSync} from 'node:zlib';
import {ROOT, fileHash} from '../reference-node/provenance.mjs';
import {canonical, hashJSON} from '../battle-contract/canonical.mjs';
import {atomicJSON, readJSON, seal, verifySeal, digest} from '../corpus/common.mjs';
import {episodeManifest} from './manifest.mjs';
import {writerLock} from './storage.mjs';

const files = ['campaign.mjs', 'worker.mjs', 'storage.mjs', 'cli.mjs'].map(p => `packages/episodes/${p}`);
files.push('packages/corpus/common.mjs');
const sourceHashes = Object.fromEntries(files.map(p => [p, fileHash(`${ROOT}/${p}`)]));

export function checkEpisodeFile(directory, record) {
  const bytes = readFileSync(`${directory}/${record.file}`);
  if (bytes.length !== record.bytes || digest(bytes) !== record.sha256) throw new Error(`Episode file corrupted: ${record.file}`);
  const plain = gunzipSync(bytes);
  if (plain.length !== record.json_bytes || digest(plain) !== record.json_sha256) throw new Error(`Episode JSON corrupted: ${record.file}`);
  return plain;
}

function checkRecords(index) {
  verifySeal(index);
  if (index.version !== 'r4-campaign-1' || hashJSON(index.config) !== index.campaign_id) throw new Error('Campaign identity');
  if (!['collecting', 'complete'].includes(index.status)) throw new Error('Campaign status');
  let previous = -1;
  for (const record of index.episodes) {
    if (!Number.isSafeInteger(record.index) || record.index <= previous || record.index >= index.config.count ||
        record.file !== `episode-${String(record.index).padStart(6, '0')}.json.gz` || !record.replay_verified) throw new Error('Campaign record index');
    previous = record.index;
  }
  if (index.completed !== index.episodes.length || (index.status === 'complete' && index.completed !== index.config.count)) throw new Error('Incomplete campaign accounting');
  return index;
}

function callWorker(child, job) {
  return new Promise((resolveJob, reject) => {
    const cleanup = () => { child.off('message', message); child.off('error', failed); child.off('exit', exited); };
    const message = result => { cleanup(); result.ok ? resolveJob(result.record) : reject(new Error(result.message)); };
    const failed = error => { cleanup(); reject(error); };
    const exited = code => failed(new Error(`Worker exited (${code})`));
    child.once('message', message); child.once('error', failed); child.once('exit', exited);
    child.send(job, error => { if (error) failed(error); });
  });
}

export async function campaign({out, namespace, count = 1000, workers = 4, onProgress = () => {}}) {
  if (typeof namespace !== 'string' || !namespace || namespace.length > 180 || !out) throw new Error('Need output and bounded namespace');
  if (!Number.isSafeInteger(count) || count < 1 || count > 100000 || !Number.isSafeInteger(workers) || workers < 1 || workers > 8) throw new Error('Invalid campaign size/workers');
  const directory = resolve(out), manifest = episodeManifest(), started = Date.now();
  const config = {manifest_id: manifest.manifest_id, namespace, count, source_hashes: sourceHashes,
    label_version: 'r4-campaign-label-1', order: 'alternating-submission-order', split: 'unassigned'};
  const campaignId = hashJSON(config);
  mkdirSync(directory, {recursive: true});
  const unlock = writerLock(directory), children = [];
  let state;
  try {
    const progressPath = `${directory}/progress.json`;
    if (existsSync(progressPath)) {
      state = checkRecords(readJSON(progressPath));
      if (state.campaign_id !== campaignId || canonical(state.config) !== canonical(config)) throw new Error('Campaign resume provenance/configuration differs');
      if (canonical(readJSON(`${directory}/manifest.json`)) !== canonical(manifest)) throw new Error('Campaign reference differs');
      for (const record of state.episodes) checkEpisodeFile(directory, record);
    } else {
      if (existsSync(`${directory}/manifest.json`) || existsSync(`${directory}/index.json`)) throw new Error('Existing campaign without checkpoint');
      writeFileSync(`${directory}/manifest.json`, canonical(manifest) + '\n', {flag: 'wx'});
      state = seal({version: 'r4-campaign-1', campaign_id: campaignId, config, status: 'collecting', completed: 0, episodes: []});
      atomicJSON(progressPath, state);
    }
    if (existsSync(`${directory}/index.json`)) {
      const complete = checkRecords(readJSON(`${directory}/index.json`));
      if (complete.campaign_id !== campaignId || canonical(complete.episodes) !== canonical(state.episodes)) throw new Error('Campaign index/checkpoint mismatch');
      const failures = existsSync(`${directory}/failures.jsonl`) ? readFileSync(`${directory}/failures.jsonl`) : Buffer.alloc(0);
      if (digest(failures) !== complete.failures_sha256) throw new Error('Campaign failure accounting changed');
      return complete;
    }
    const done = new Set(state.episodes.map(r => r.index));
    const remaining = Array.from({length: count}, (_, i) => i).filter(i => !done.has(i));
    let next = 0, failure = null;
    const loops = Array.from({length: Math.min(workers, remaining.length)}, async () => {
      const child = fork(new URL('./worker.mjs', import.meta.url), [], {cwd: ROOT, stdio: ['ignore', 'ignore', 'pipe', 'ipc']});
      children.push(child);
      let diagnostics = '';
      child.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-8000); });
      while (!failure && next < remaining.length) {
        const index = remaining[next++];
        try {
          for (const [p, expected] of Object.entries(sourceHashes)) if (fileHash(`${ROOT}/${p}`) !== expected) throw new Error('Campaign source changed');
          const record = await callWorker(child, {directory, namespace, index});
          if (record.index !== index) throw new Error('Worker returned wrong episode index');
          const episodes = [...state.episodes, record].sort((a, b) => a.index - b.index);
          const nextState = seal({version: 'r4-campaign-1', campaign_id: campaignId, config, status: 'collecting', completed: episodes.length, episodes});
          atomicJSON(progressPath, nextState); state = nextState;
          onProgress({completed: state.completed, requested: count, elapsed_ms: Date.now() - started});
        } catch (error) {
          failure ??= error;
          appendFileSync(`${directory}/failures.jsonl`, canonical({index, label: `${namespace}/${String(index).padStart(6, '0')}`,
            message: error.message, diagnostics, committed_episodes: state.completed, time: new Date().toISOString()}) + '\n');
        }
      }
    });
    const settled = await Promise.allSettled(loops);
    for (const result of settled) if (result.status === 'rejected') {
      failure ??= result.reason;
      appendFileSync(`${directory}/failures.jsonl`, canonical({index: null, label: null,
        message: String(result.reason), committed_episodes: state.completed, time: new Date().toISOString()}) + '\n');
    }
    if (failure) throw failure;
    if (canonical(episodeManifest()) !== canonical(manifest)) throw new Error('Campaign provenance changed before publication');
    const failures = existsSync(`${directory}/failures.jsonl`) ? readFileSync(`${directory}/failures.jsonl`) : Buffer.alloc(0);
    const complete = seal({version: 'r4-campaign-1', campaign_id: campaignId, config, status: 'complete', completed: state.completed,
      episodes: state.episodes, workers, elapsed_ms: Date.now() - started,
      failure_events: failures.length ? failures.toString().trim().split('\n').length : 0, failures_sha256: digest(failures)});
    checkRecords(complete); atomicJSON(`${directory}/index.json`, complete);
    return complete;
  } finally {
    for (const child of children) child.kill();
    unlock();
  }
}
