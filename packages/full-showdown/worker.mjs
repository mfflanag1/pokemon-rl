import readline from 'node:readline';
import {FullShowdownSession} from './session.mjs';
import {describe} from '../reference-node/engine.mjs';

let session = null;
const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  let command;
  try {
    if (line.length > 1_000_000) throw new Error('Oversized command');
    command = JSON.parse(line);
    let data;
    if (command.op === 'hello') {
      data = {schema_version: 'full-showdown-worker-v1', reference_commit: describe().commit};
    } else if (command.op === 'start') {
      session?.close(); session = new FullShowdownSession();
      // Optional coordinator-only curriculum provider; never loaded from actor input.
      let plan = null, provider = null;
      if (command.curriculum === true) {
        provider = await import('../training-curriculum/index.mjs');
        plan = provider.curriculumEpisode({episodeIndex: command.episode_index, runSeed: `f1:${command.run_seed}`});
      }
      data = await session.start({episodeIndex: command.episode_index, runSeed: command.run_seed, plan});
      if (provider) {
        session.provenance.prefix_evidence = provider.verifyPublicEvidence(session.projection.spectator, plan.evidence_requirements);
        data.provenance = session.provenance;
      }
    } else if (command.op === 'step' && session) {
      data = await session.step(command);
    } else if (command.op === 'cutoff' && session) {
      data = session.cutoff();
    } else if (command.op === 'close') {
      session?.close(); session = null; data = {closed: true};
    } else throw new Error('Unknown operation or missing session');
    process.stdout.write(`${JSON.stringify({id: command.id, ok: true, data})}\n`);
  } catch (error) {
    session?.close(); session = null;
    process.stdout.write(`${JSON.stringify({id: command?.id ?? null, ok: false, error: error.message})}\n`);
  }
}
session?.close();
