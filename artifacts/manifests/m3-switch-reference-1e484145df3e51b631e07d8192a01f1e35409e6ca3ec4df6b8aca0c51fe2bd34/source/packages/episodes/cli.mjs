import {parseArgs} from 'node:util';
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {dirname} from 'node:path';
import {gunzipSync} from 'node:zlib';
import {canonical, parseJSON} from '../battle-contract/canonical.mjs';
import {defaultSpec} from '../reference-node/runner.mjs';
import {campaign} from './campaign.mjs';
import {runPhaseEpisode, replayPhaseEpisode} from './session.mjs';
import {episodeManifest} from './manifest.mjs';

const {positionals, values} = parseArgs({allowPositionals: true, options: {
  out: {type: 'string'}, namespace: {type: 'string'}, count: {type: 'string'}, workers: {type: 'string'},
  'truncate-after': {type: 'string'}, fixture: {type: 'string'},
}});
const [command, argument] = positionals;
if (command === 'campaign') {
  const index = await campaign({out: values.out, namespace: values.namespace, count: Number(values.count ?? 1000), workers: Number(values.workers ?? 4),
    onProgress: p => { if (p.completed % 10 === 0 || p.completed === p.requested) console.error(JSON.stringify(p)); }});
  console.log(JSON.stringify({campaign_id: index.campaign_id, episodes: index.completed, failure_events: index.failure_events}));
} else if (command === 'run') {
  if (!argument || !values.out) throw new Error('run LABEL --out FILE');
  const fixture = values.fixture ?? null;
  const bundle = await runPhaseEpisode(defaultSpec(argument), {fixture, truncateAfter: values['truncate-after'] === undefined ? null : Number(values['truncate-after'])});
  mkdirSync(dirname(values.out), {recursive: true});
  writeFileSync(values.out, canonical(bundle) + '\n', {flag: 'wx'});
  writeFileSync(`${values.out}.manifest.json`, canonical(episodeManifest(fixture ? 'constructed' : 'official')) + '\n', {flag: 'wx'});
  console.log(JSON.stringify({path: values.out, outcome: bundle.episode.payload.outcome, decisions: bundle.episode.payload.decisions.length}));
} else if (command === 'replay') {
  const data = readFileSync(argument), bundle = parseJSON(argument.endsWith('.gz') ? gunzipSync(data) : data);
  const result = await replayPhaseEpisode(bundle);
  console.log(JSON.stringify({verified: true, episode_hash: result.episode.content_hash}));
} else throw new Error('Use campaign --out DIR --namespace NAME, run LABEL --out FILE, or replay FILE');
