import {readFileSync, writeFileSync} from 'node:fs';
import {createInterface} from 'node:readline';
import {describe, generateTeam, seedFor} from './engine.mjs';
import {defaultSpec, runEpisode, replayEpisode} from './runner.mjs';

async function dispatch(message) {
  switch (message.command) {
    case 'describe': return describe();
    case 'generate_teams': {
      if (!Array.isArray(message.seeds) || message.seeds.length > 1000) throw new Error('Supply at most 1000 explicit seeds');
      return message.seeds.map(generateTeam);
    }
    case 'run_episode': return runEpisode(message.spec ?? defaultSpec(message.label ?? 'r1-smoke'));
    case 'replay_episode': return replayEpisode(message.episode);
    default: throw new Error('Unknown command');
  }
}

async function main() {
  const [command = 'describe', argument, output] = process.argv.slice(2);
  if (command === 'worker') {
    // Requests run sequentially in one process; malformed input cannot restart the oracle.
    for await (const line of createInterface({input: process.stdin, crlfDelay: Infinity})) {
      if (!line.trim()) continue;
      let request;
      try {
        request = JSON.parse(line);
        if (!request || typeof request !== 'object' || !['string', 'number'].includes(typeof request.id)) throw new Error('Request needs an id');
        const result = await dispatch(request);
        process.stdout.write(`${JSON.stringify({id: request.id, ok: true, result})}\n`);
      } catch (error) {
        process.stdout.write(`${JSON.stringify({id: request?.id ?? null, ok: false, error: {type: error.name, message: error.message}})}\n`);
      }
    }
    return;
  }
  let result;
  if (command === 'describe') result = describe();
  else if (command === 'team') result = generateTeam(seedFor(argument ?? 'r1-team'));
  else if (command === 'run') result = await runEpisode(defaultSpec(argument ?? 'r1-smoke'));
  else if (command === 'replay' && argument) {
    const episode = JSON.parse(readFileSync(argument, 'utf8'));
    const replayed = await replayEpisode(episode);
    result = {verified: true, content_hash: replayed.content_hash, outcome: replayed.outcome};
  } else throw new Error('Usage: describe | team [label] | run [label] [output.json] | replay episode.json | worker');
  const text = `${JSON.stringify(result, null, 2)}\n`;
  if (output && command === 'run') {
    writeFileSync(output, text, {flag: 'wx'});
    console.log(JSON.stringify({saved: output, content_hash: result.content_hash, outcome: result.outcome}));
  } else process.stdout.write(text);
}

main().catch(error => { console.error(`${error.name}: ${error.message}`); process.exitCode = 1; });
