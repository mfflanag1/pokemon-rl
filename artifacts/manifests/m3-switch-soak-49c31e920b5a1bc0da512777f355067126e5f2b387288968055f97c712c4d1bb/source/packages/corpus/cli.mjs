import {parseArgs} from 'node:util';
import {generateCorpus} from './generate.mjs';
import {replayCorpus} from './replay.mjs';

try {
  const {values, positionals} = parseArgs({allowPositionals: true, options: {
    out: {type: 'string'}, namespace: {type: 'string'}, count: {type: 'string', default: '100000'},
    'shard-size': {type: 'string', default: '1000'}, samples: {type: 'string', default: '1000'},
  }});
  if (!values.out) throw new Error('Supply --out corpus-directory');
  if (positionals[0] === 'generate' && positionals.length === 1) {
    const result = generateCorpus({out: values.out, namespace: values.namespace, count: Number(values.count),
      shardSize: Number(values['shard-size']), onProgress: progress => console.error(JSON.stringify(progress))});
    console.log(JSON.stringify({corpus_id: result.corpus_id, teams: result.completed, shards: result.chunks.length, failure_events: result.failure_events}));
  } else if (positionals[0] === 'replay' && positionals.length === 1) {
    const result = replayCorpus({out: values.out, samples: Number(values.samples)});
    console.log(JSON.stringify({verified: result.verified, samples: result.sample_count, corpus_id: result.corpus_id}));
  } else throw new Error('Usage: generate --out DIR --namespace NAME [--count 100000 --shard-size 1000] | replay --out DIR [--samples 1000]');
} catch (error) {
  console.error(`${error.name}: ${error.message}`);
  process.exitCode = 1;
}
