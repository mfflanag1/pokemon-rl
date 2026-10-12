import {fork} from 'node:child_process';
import {readFileSync, writeFileSync, mkdirSync, appendFileSync} from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {performance} from 'node:perf_hooks';
import {generateTeam, seedFor} from '../reference-node/engine.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {episodeManifest} from '../episodes/manifest.mjs';
import {ROOT, fileHash} from '../reference-node/provenance.mjs';
import {hashJSON, canonical, parseJSON} from '../battle-contract/canonical.mjs';
import {seal, verifySeal} from '../corpus/common.mjs';
import {POLICY, POLICY_HASH, teamKey, partition, pairPartition} from './identity.mjs';

export const SOURCE_FILES = ['packages/dataset/identity.mjs', 'packages/dataset/collect.mjs', 'packages/dataset/worker.mjs',
  'contracts/dataset-v1/policy.json'];
const sourceHashes = () => Object.fromEntries(SOURCE_FILES.map(file => [file, fileHash(path.join(ROOT, file))]));
const write = (dir, file, value, mode = 0o644) => writeFileSync(path.join(dir, file), canonical(value) + '\n', {flag: 'wx', mode});

export function buildJobs(teams, namespace) {
  const jobs = [];
  for (const split of ['train', 'development']) {
    const rows = teams[split];
    if (rows.length % 2) throw new Error('Pairing needs even team counts');
    for (let i = 0; i < rows.length; i += 2) {
      const pair = [rows[i], rows[i + 1]], pairNumber = i / 2;
      if (pairPartition(...pair.map(t => t.artifact.payload)) !== split) throw new Error('Pair split mismatch');
      const group = hashJSON(['r7-matchup-1', POLICY_HASH, pair.map(t => t.team_key).sort()]);
      const label = `${namespace}/${split}/pair/${pairNumber}`;
      const policySeeds = [seedFor(`${label}/policy/0`), seedFor(`${label}/policy/1`)];
      for (const mirror of [false, true]) {
        const ordered = mirror ? [...pair].reverse() : pair;
        jobs.push({id: `${split}-${String(pairNumber).padStart(4, '0')}-${Number(mirror)}`, split, group, mirror,
          team_keys: ordered.map(t => t.team_key),
          spec: {format_id: POLICY.profile, teams: ordered.map(t => t.entry), mechanics_seed: seedFor(`${label}/mechanics`),
            policy_seeds: mirror ? [...policySeeds].reverse() : policySeeds,
            policy_id: 'upstream-random-move-0.9-no-tera-r1', max_submissions: 20000}});
      }
    }
  }
  return jobs;
}

export function checkJobSeeds(jobs, teams, exposed) {
  const used = new Map(Object.values(teams).flat().map(t => [t.entry.seed, `${t.split}/${t.team_key}/team`]));
  for (const job of jobs) {
    for (const [seed, role] of [[job.spec.mechanics_seed, 'mechanics'], ...job.spec.policy_seeds.map(s => [s, 'policy'])]) {
      const identity = `${job.split}/${job.group}/${role}`;
      if (exposed.has(seed) || (used.has(seed) && used.get(seed) !== identity)) throw new Error('Battle/policy seed overlaps exposure or another role/group');
      used.set(seed, identity);
    }
  }
}

function send(child, job) {
  return new Promise((resolve, reject) => {
    const cleanup = () => { child.off('message', message); child.off('error', failed); child.off('exit', exited); };
    const message = value => { cleanup(); value.ok ? resolve(value.record) : reject(new Error(value.message)); };
    const failed = e => { cleanup(); reject(e); };
    const exited = code => failed(new Error(`Dataset worker exited ${code}`));
    child.once('message', message); child.once('error', failed); child.once('exit', exited);
    child.send(job, e => { if (e) failed(e); });
  });
}

export async function collect({out, exposures, namespace = 'r7-reference-2026-10-10-v1', workers = 4,
  counts = {train: 64, development: 32, final: 128}, onProgress = () => {}}) {
  if (!Number.isSafeInteger(workers) || workers < 1 || workers > 8) throw new Error('Invalid worker count');
  if (!namespace || namespace.length > 180) throw new Error('Invalid namespace');
  if (Object.keys(counts).sort().join(',') !== 'development,final,train' || Object.values(counts).some(n => !Number.isSafeInteger(n) || n < 2 || n > 10000 || n % 2)) throw new Error('Invalid reservation counts');
  const directory = path.resolve(out), registry = verifySeal(parseJSON(readFileSync(exposures)));
  if (registry.version !== 'r7-exposures-1' || registry.policy_hash !== POLICY_HASH) throw new Error('Exposure policy mismatch');
  for (const [field, pattern] of [['team_keys', /^[0-9a-f]{64}$/], ['episode_keys', /^[0-9a-f]{64}$/], ['seeds', /^sodium,[0-9a-f]{64}$/]]) {
    if (!Array.isArray(registry[field]) || !registry[field].length || new Set(registry[field]).size !== registry[field].length || registry[field].some(x => !pattern.test(x))) throw new Error('Incomplete exposure registry');
  }
  mkdirSync(directory); mkdirSync(`${directory}/teams`); mkdirSync(`${directory}/raw`);
  const manifest = episodeManifest(), sources = sourceHashes();
  const plan = seal({version: 'r7-collection-plan-1', namespace, policy: POLICY, policy_hash: POLICY_HASH,
    team_counts: counts, episode_manifest_id: manifest.manifest_id, source_hashes: sources,
    exposure_registry_hash: registry.content_hash, exposure_registry_sha256: fileHash(exposures),
    workers, pairing: 'adjacent-same-split-with-swapped-seat-replay-pairs',
    final_access: 'reservation-only', expected_episodes: counts.train + counts.development,
    frozen_at: new Date().toISOString()});
  // The plan is durable before generating candidates or inspecting outcomes.
  write(directory, 'plan.json', plan); write(directory, 'reference-manifest.json', manifest);
  const teams = {}, sampling = {}, seen = new Set(registry.team_keys), legacySeeds = new Set(registry.seeds);
  try {
    for (const split of ['train', 'development', 'final']) {
      const started = performance.now(), rows = [], rejected = {other_partition: 0, duplicate_or_exposed: 0, seed_overlap: 0};
      let attempt = 0;
      while (rows.length < counts[split]) {
        if (attempt >= counts[split] * 1000) throw new Error('Reservation rejection budget exceeded');
        const seed = seedFor(`${namespace}/team/${split}/${attempt++}`);
        if (legacySeeds.has(seed)) { rejected.seed_overlap++; continue; }
        const entry = generateTeam(seed), artifact = teamArtifact(entry, manifest.reference_contract.manifest_id);
        if (partition(artifact.payload) !== split) { rejected.other_partition++; continue; }
        const key = teamKey(artifact.payload);
        if (seen.has(key)) { rejected.duplicate_or_exposed++; continue; }
        rows.push({split, team_key: key, artifact, entry}); seen.add(key); legacySeeds.add(seed);
      }
      teams[split] = rows;
      write(directory, `teams/${split}.json`, seal({version: 'r7-team-reservation-1', split, plan_hash: plan.content_hash, rows}), split === 'final' ? 0o600 : 0o644);
      sampling[split] = {attempts: attempt, accepted: rows.length, rejected, elapsed_ms: Math.round(performance.now() - started)};
      onProgress({stage: 'reserved', split, ...sampling[split]});
    }
    const jobs = buildJobs(teams, namespace);
    checkJobSeeds(jobs, teams, new Set(registry.seeds));
    write(directory, 'jobs.json', seal({version: 'r7-job-plan-1', plan_hash: plan.content_hash, jobs}));
    // No final teams or jobs are sent to workers.
    const records = [], children = [], started = performance.now(); let cursor = 0, failure;
    try {
      const settled = await Promise.allSettled(Array.from({length: Math.min(workers, jobs.length)}, async () => {
        const child = fork(new URL('./worker.mjs', import.meta.url), [], {cwd: ROOT, execArgv: [], stdio: ['ignore', 'ignore', 'pipe', 'ipc']});
        children.push(child); let diagnostics = '';
        child.stderr.on('data', data => { diagnostics = (diagnostics + data).slice(-4000); });
        while (!failure && cursor < jobs.length) {
          const job = jobs[cursor++];
          try {
            const record = await send(child, {directory, job, manifest});
            if (record.job_id !== job.id) throw new Error('Wrong worker result');
            records.push(record);
            appendFileSync(`${directory}/progress.jsonl`, canonical(record) + '\n');
            if (records.length % 16 === 0) onProgress({stage: 'episodes', completed: records.length, requested: jobs.length});
          } catch (e) { failure ??= e; appendFileSync(`${directory}/failures.jsonl`, canonical({job_id: job.id, error: e.message, diagnostics}) + '\n'); }
        }
      }));
      for (const r of settled) if (r.status === 'rejected') failure ??= r.reason;
      if (failure) throw failure;
      if (records.length !== jobs.length || canonical(sourceHashes()) !== canonical(sources)) throw new Error('Incomplete or changed producer');
      if (canonical(episodeManifest()) !== canonical(manifest)) throw new Error('Reference runtime changed before publication');
      const index = seal({version: 'r7-collection-1', plan_hash: plan.content_hash, status: 'complete', sampling,
        records: records.sort((a, b) => a.job_id.localeCompare(b.job_id, 'en')), expected: jobs.length,
        collection_wall_ms: Math.round(performance.now() - started), workers, failure_events: 0,
        hardware: {cpu: os.cpus()[0].model, logical_cpus: os.cpus().length, memory_bytes: os.totalmem(), platform: os.platform(), release: os.release(), arch: os.arch(), node: process.version},
        files: Object.fromEntries(['plan.json', 'reference-manifest.json', 'jobs.json', ...Object.keys(teams).map(s => `teams/${s}.json`)].map(f => [f, fileHash(path.join(directory, f))]))});
      write(directory, 'index.json', index); return index;
    } finally { for (const child of children) child.kill(); }
  } catch (e) { write(directory, 'failed.json', {status: 'failed', message: e.message}); throw e; }
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const [out, exposures] = process.argv.slice(2);
  const index = await collect({out, exposures, onProgress: p => console.log(JSON.stringify(p))});
  console.log(JSON.stringify({status: index.status, episodes: index.records.length, elapsed_ms: index.collection_wall_ms}));
}
