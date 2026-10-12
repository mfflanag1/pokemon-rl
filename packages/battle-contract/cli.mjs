import {readFileSync, mkdirSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {canonical, parseJSON} from './canonical.mjs';
import {validateArtifact} from './artifacts.mjs';

async function main() {
  const [command, argument, output] = process.argv.slice(2);
  if (command === 'validate' && argument) {
    const artifact = validateArtifact(parseJSON(readFileSync(argument)));
    console.log(JSON.stringify({verified: true, artifact_id: artifact.artifact_id}));
    return;
  }
  const {contractManifest, buildVocabulary, episodeArtifacts} = await import('../reference-node/contract-v1.mjs');
  if (command === 'vocabulary') {
    process.stdout.write(canonical(buildVocabulary()) + '\n');
  } else if (command === 'describe') {
    process.stdout.write(canonical(contractManifest()) + '\n');
  } else if (command === 'export' && argument && output) {
    const {runEpisode, defaultSpec} = await import('../reference-node/runner.mjs');
    const reference = await runEpisode(defaultSpec(argument));
    const bundle = episodeArtifacts(reference);
    const directory = resolve(output);
    mkdirSync(directory, {recursive: true});
    const artifacts = [bundle.manifest, ...bundle.teams, ...bundle.views, ...bundle.decisions, bundle.episode];
    // Exclusive creation avoids overwriting earlier evidence.
    writeFileSync(`${directory}/artifacts.jsonl`, artifacts.map(a => canonical(a)).join('\n') + '\n', {flag: 'wx'});
    writeFileSync(`${directory}/reference.json`, JSON.stringify(reference) + '\n', {flag: 'wx'});
    console.log(JSON.stringify({directory, manifest_id: bundle.manifest.manifest_id, artifacts: artifacts.length, decisions: bundle.decisions.length}));
  } else throw new Error('Usage: describe | vocabulary | validate file.json | export label new-output-directory');
}

main().catch(error => { console.error(`${error.name}: ${error.message}`); process.exitCode = 1; });
