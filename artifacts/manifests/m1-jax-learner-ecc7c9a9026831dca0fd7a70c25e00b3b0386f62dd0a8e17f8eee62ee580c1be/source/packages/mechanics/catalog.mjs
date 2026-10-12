import {FAMILIES} from './policy.mjs';
import {hashJSON} from '../reference-node/provenance.mjs';

export function fixtureCatalog(manifest, fixtures) {
  const body = {
    schema_version: 'r6-catalog-1', artifact_kind: 'fixture-catalog', manifest_hash: manifest.content_hash,
    fixtures_hash: fixtures.content_hash,
    families: Object.entries(FAMILIES).map(([id, [ticket, obligations]]) => ({id, ticket, obligations,
      required_dimensions: ['positive and failure paths', 'both player seats', 'boundary and cleanup',
        'interactions with suppression/immunity/targeting', 'random branches and draw schedule where applicable'],
      reference_cases: fixtures.cases.filter(c => c.fixture_families.includes(id)).map(c => c.id),
      differential_status: 'planned'})),
    entries: manifest.entries.map(e => ({entry: e.key, ticket: e.ticket, fixture_families: e.fixture_families,
      reference_cases: fixtures.cases.filter(c => c.entries.includes(e.key)).map(c => c.id),
      required_checks: ['declarative data parity', ...e.callback_paths.map(c => `callback:${c.path}`),
        ...(e.embedded_condition ? ['embedded condition lifecycle'] : [])],
      differential_status: 'planned'})),
    inherited_evidence: [
      {input: 'r4_index', purpose: '13 request/phase fixtures: natural, truncation and constructed; retain their original evidence labels.'},
      {input: 'r5_probes', purpose: '40 observer probes; information-boundary evidence, not official reachability or custom-engine parity.'},
    ],
    coverage_policy: 'A passing reference case is not a differential test. Family membership creates obligations; it does not cover every member. Promotion requires per-entry and interaction evidence against a separately versioned engine.',
  };
  return {...body, content_hash: hashJSON(body)};
}
