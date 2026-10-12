import {writeFileSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import {performance} from 'node:perf_hooks';
import {runPhaseEpisode, replayPhaseEpisode} from '../episodes/session.mjs';
import {canonical, parseJSON} from '../battle-contract/canonical.mjs';
import {digest} from '../corpus/common.mjs';

process.on('message', async ({directory, job, manifest}) => {
  try {
    if (!['train', 'development'].includes(job.split)) throw new Error('Final collection is sealed');
    const start = performance.now();
    const bundle = await runPhaseEpisode(job.spec, {manifest, order: job.mirror ? 'p2-first' : 'p1-first'});
    if (bundle.episode.payload.outcome.status !== 'terminal') throw new Error('Incomplete dataset episode');
    const generated = performance.now(), plain = Buffer.from(canonical(bundle) + '\n');
    await replayPhaseEpisode(parseJSON(plain), manifest);
    const replayed = performance.now(), compressed = gzipSync(plain, {level: 6});
    const file = `raw/${job.id}.json.gz`;
    writeFileSync(`${directory}/${file}`, compressed, {flag: 'wx'});
    const p = bundle.episode.payload;
    process.send({ok: true, record: {job_id: job.id, split: job.split, group: job.group, mirror: job.mirror,
      file, bytes: compressed.length, sha256: digest(compressed), json_bytes: plain.length, json_sha256: digest(plain),
      episode_key: p.episode_key, episode_hash: bundle.episode.content_hash, replay_verified: true,
      decisions: p.decisions.length, turns: p.outcome.turn, unavailable_rejections: p.decisions.filter(d => !d.accepted).length,
      generation_ms: Math.round(generated - start), replay_ms: Math.round(replayed - generated),
      serialization_storage_ms: Math.round(performance.now() - replayed)}});
  } catch (error) { process.send({ok: false, message: error.stack ?? error.message, job_id: job.id}); }
});
