import {readFileSync, writeFileSync, existsSync, mkdirSync, renameSync} from 'node:fs';
import {gzipSync} from 'node:zlib';
import {defaultSpec} from '../reference-node/runner.mjs';
import {canonical, hashJSON, parseJSON} from '../battle-contract/canonical.mjs';
import {digest} from '../corpus/common.mjs';
import {runPhaseEpisode, replayPhaseEpisode} from './session.mjs';

export function summarize(bundle) {
  const p = bundle.episode.payload, phase_groups = {}, decision_phases = {}, examples = {};
  const seen = new Set();
  for (const [i, frame] of p.frames.entries()) {
    if (frame.phase === 'terminal' || seen.has(frame.group)) continue;
    seen.add(frame.group); phase_groups[frame.phase] = (phase_groups[frame.phase] ?? 0) + 1;
    examples[frame.phase] ??= i;
  }
  let pending_rejections = 0;
  for (const d of p.decisions) {
    const phase = p.frames[d.frame_before].side_phases[d.side];
    decision_phases[phase] = (decision_phases[phase] ?? 0) + 1;
    if (!d.accepted && p.frames[d.frame_before].commitments[d.side === 'p1' ? 'p2' : 'p1'] !== null) pending_rejections++;
  }
  return {decisions: p.decisions.length, transitions: p.transitions.length, turns: p.outcome.turn,
    winner: p.outcome.winner, phase_groups, decision_phases, examples,
    unavailable_rejections: p.decisions.filter(d => !d.accepted).length, rejections_with_opponent_choice: pending_rejections};
}

process.on('message', async job => {
  try {
    const {directory, namespace, index} = job;
    const manifest = parseJSON(readFileSync(`${directory}/manifest.json`));
    const label = `${namespace}/${String(index).padStart(6, '0')}`;
    const generated = await runPhaseEpisode(defaultSpec(label), {manifest, order: index % 2 ? 'p2-first' : 'p1-first'});
    if (generated.episode.payload.outcome.status !== 'terminal') throw new Error('Campaign episode is incomplete');
    const plain = Buffer.from(canonical(generated) + '\n');
    // Replay the serialized representation, not just the in-memory producer object.
    const replay = await replayPhaseEpisode(parseJSON(plain), manifest);
    if (replay.episode.content_hash !== generated.episode.content_hash) throw new Error('Replay mismatch');
    const compressed = gzipSync(plain, {level: 6}), file = `episode-${String(index).padStart(6, '0')}.json.gz`;
    if (existsSync(`${directory}/${file}`)) {
      mkdirSync(`${directory}/orphans`, {recursive: true});
      renameSync(`${directory}/${file}`, `${directory}/orphans/${file}-${Date.now()}-${process.pid}`);
    }
    const temporary = `${directory}/${file}.part-${process.pid}`;
    writeFileSync(temporary, compressed, {flag: 'wx'});
    renameSync(temporary, `${directory}/${file}`);
    process.send({ok: true, record: {index, label, file, bytes: compressed.length, sha256: digest(compressed),
      json_bytes: plain.length, json_sha256: digest(plain), episode_hash: generated.episode.content_hash,
      normalized_output_hash: generated.episode.payload.normalized_output_hash, replay_verified: true,
      summary: summarize(generated)}});
  } catch (error) { process.send({ok: false, index: job.index, message: error.stack ?? error.message}); }
});
