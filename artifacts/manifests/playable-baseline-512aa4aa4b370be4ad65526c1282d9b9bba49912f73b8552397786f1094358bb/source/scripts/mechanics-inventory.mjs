import {mkdirSync, writeFileSync, existsSync, readFileSync} from 'node:fs';
import path from 'node:path';
import {buildManifest} from '../packages/mechanics/manifest.mjs';
import {captureFixtures} from '../packages/mechanics/fixtures.mjs';
import {fixtureCatalog} from '../packages/mechanics/catalog.mjs';
import {canonical, hashJSON, ROOT, fileHash} from '../packages/reference-node/provenance.mjs';

const destination = path.resolve(process.argv[2] ?? 'artifacts/runs/r6-mechanics-v4');
const manifest = buildManifest(), fixtures = await captureFixtures(), catalog = fixtureCatalog(manifest, fixtures);
mkdirSync(destination, {recursive: true});
function save(name, value) {
  const file = path.join(destination, name), bytes = canonical(value) + '\n';
  if (existsSync(file) && readFileSync(file, 'utf8') !== bytes) throw new Error(`Refusing to overwrite changed R6 artifact: ${file}`);
  writeFileSync(file, bytes);
}
save('manifest.json', manifest); save('fixtures.json', fixtures); save('catalog.json', catalog);
const body = {schema_version: 'r6-report-1', artifact_kind: 'mechanics-report', manifest_hash: manifest.content_hash,
  counts: {entries: manifest.entries.length, pathways: manifest.pathways.length, unclassified_pathways: manifest.unclassified_pathways.length,
    source_files: Object.keys(manifest.sources).length, fixture_families: catalog.families.length,
    reference_cases: fixtures.cases.length, replay_verified: fixtures.cases.filter(c => c.replay_verified).length,
    custom_engine_supported: manifest.entries.filter(e => e.support === 'implemented-tested').length,
    by_domain: Object.fromEntries(['moves', 'abilities', 'items', 'forms', 'conditions', 'core'].map(d => [d,
      {envelope: manifest.entries.filter(e => e.domain === d).length, observed_initial: manifest.entries.filter(e => e.domain === d && e.observed_count > 0).length}]))},
  files: Object.fromEntries(['manifest.json', 'fixtures.json', 'catalog.json'].map(name => [name, fileHash(path.join(destination, name))])),
  producer: Object.fromEntries(['packages/mechanics/source-index.mjs', 'packages/mechanics/policy.mjs', 'packages/mechanics/manifest.mjs',
    'packages/mechanics/fixtures.mjs', 'packages/mechanics/catalog.mjs', 'scripts/mechanics-inventory.mjs',
    'tools/mechanics-deps/package-lock.json'].map(file => [file, fileHash(path.join(ROOT, file))])),
  scope: 'Conservative mechanics inventory and reference fixture catalog only; no accelerated-engine fidelity or minimal reachability claim.'};
save('report.json', {...body, content_hash: hashJSON(body)});
console.log(JSON.stringify({destination, ...body.counts, manifest_hash: manifest.content_hash}));
