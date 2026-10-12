import {readFileSync} from 'node:fs';
import {describe, Dex} from './engine.mjs';
import {fileHash, ROOT, hashJSON as r1Hash} from './provenance.mjs';
import {PlayerProjection, normalizeProtocol} from './observations.mjs';
import {makeArtifact, validateArtifact} from '../battle-contract/artifacts.mjs';
import {hashJSON, parseJSON, canonical} from '../battle-contract/canonical.mjs';
import {admissibleActions, actionMask, fromProtocol, requestPhase, toProtocol} from '../battle-contract/actions.mjs';
import {validateSchema} from '../battle-contract/schema.mjs';

const sourcePaths = [
  'contracts/battle-v1/schema.json', 'contracts/battle-v1/actions.json',
  'packages/battle-contract/canonical.mjs', 'packages/battle-contract/schema.mjs',
  'packages/battle-contract/actions.mjs', 'packages/battle-contract/artifacts.mjs',
  'packages/reference-node/contract-v1.mjs', 'package-lock.json',
];
const loadedHashes = Object.fromEntries(sourcePaths.map(name => [name, fileHash(`${ROOT}/${name}`)]));

export const seedObject = seed => {
  if (typeof seed !== 'string' || seed.length !== 71 || !/^sodium,[0-9a-f]{64}$/.test(seed)) throw new TypeError('Invalid reference seed');
  return {kind: 'showdown-sodium', hex: seed.slice(7)};
};

export function buildVocabulary() {
  // Dex.species.all() omits cosmetic entries resolved lazily by Dex.species.get().
  const species = Dex.species.all();
  const cosmeticSpecies = species.flatMap(s => (s.cosmeticFormes ?? []).map(name => Dex.species.get(name)));
  const domains = Object.fromEntries([
    ['species', [...species, ...cosmeticSpecies]], ['moves', Dex.moves.all()], ['items', Dex.items.all()],
    ['abilities', Dex.abilities.all()], ['types', Dex.types.all()],
  ].map(([domain, values]) => [domain, ['<unknown>', '<none>', ...[...new Set(values.map(x => x.id))].sort()]]));
  return {version: 'showdown-dex-ids-v2', showdown_commit: describe().commit, domains};
}

export function contractManifest() {
  for (const [name, expected] of Object.entries(loadedHashes)) {
    if (fileHash(`${ROOT}/${name}`) !== expected) throw new Error('Contract source changed; restart the worker');
  }
  const vocabulary = parseJSON(readFileSync(`${ROOT}/contracts/battle-v1/vocabulary.json`));
  if (canonical(vocabulary) !== canonical(buildVocabulary())) throw new Error('Vocabulary differs from pinned Dex');
  return makeArtifact('manifest', null, {
    reference_manifest: describe(), canonical_version: 'pokemon-json-integers-v1',
    action_version: 'gen9-singles-request-v1', observation_version: 'player-protocol-v1',
    producer: {node: process.version}, contract_hashes: loadedHashes, vocabulary_hash: hashJSON(vocabulary),
  });
}

export function teamArtifact(entry, manifestId) {
  return makeArtifact('team', manifestId, {
    generator_seed: seedObject(entry.seed), source_team_hash: entry.team_hash,
    pokemon: structuredClone(entry.team),
    ids: entry.team.map(mon => ({
      species: Dex.species.get(mon.species).id,
      base_species: Dex.species.get(Dex.species.get(mon.species).baseSpecies).id,
      ability: Dex.abilities.get(mon.ability).id, item: Dex.items.get(mon.item).id,
      moves: mon.moves.map(move => Dex.moves.get(move).id), tera_type: Dex.types.get(mon.teraType).id,
    })),
  });
}

export function playerViewArtifact(view, manifestId) {
  return makeArtifact('player_view', manifestId, {
    viewer: view.side, request_id: view.request_id, turn: view.turn,
    phase: requestPhase(view.request), request: structuredClone(view.request),
    visible_protocol: [...view.protocol], admissible_actions: admissibleActions(view.request),
    action_mask: actionMask(view.request), opponent_private_state: {state: 'unknown'},
  });
}

export async function submitAction(session, submission, manifestId) {
  validateSchema(submission, 'submission');
  const current = playerViewArtifact(session.projection.view(submission.side), manifestId);
  if (current.artifact_id !== submission.view_id || current.payload.request_id !== submission.request_id) {
    throw new TypeError('Stale or mismatched player view');
  }
  return session.submit(submission.side, submission.request_id, toProtocol(submission.action, current.payload.request));
}

// Convert evidence, never execute an imported input tape. A replay is a separate R1 check.
export function episodeArtifacts(source, manifest = contractManifest()) {
  validateArtifact(manifest, {kind: 'manifest'});
  if (manifest.payload.reference_manifest.manifest_id !== describe().manifest_id) throw new TypeError('Manifest differs from the loaded reference');
  const {raw_output, raw_output_hash, content_hash, ...body} = source;
  if (source.schema_version !== 'r1-episode-1' || source.manifest_id !== manifest.payload.reference_manifest.manifest_id ||
      r1Hash(body) !== content_hash || r1Hash(raw_output) !== raw_output_hash ||
      r1Hash(raw_output.map(({type, data}) => ({type, data: type === 'end' ? data : normalizeProtocol(data)}))) !== source.normalized_output_hash) {
    throw new TypeError('R1 evidence/manifest mismatch');
  }
  const projection = new PlayerProjection();
  for (const {type, data} of raw_output) projection.ingest(type, data);
  for (const side of ['p1', 'p2']) {
    if (r1Hash(projection.players[side].protocol) !== r1Hash(source.protocol[side])) throw new TypeError('R1 private projection mismatch');
  }
  const manifestId = manifest.manifest_id;
  const teams = source.spec.teams.map(entry => teamArtifact(entry, manifestId));
  const episodeKey = hashJSON(source.spec);
  const views = [], decisions = [];
  for (const [sequence, record] of source.decisions.entries()) {
    // The stored prefix boundary is before submission, including on re-requests.
    const lines = source.protocol[record.side].slice(0, record.protocol_length);
    let request = null, requestId = 0, turn = 0;
    for (const line of lines) {
      if (line.startsWith('|request|')) { request = parseJSON(line.slice(9)); requestId++; }
      if (line.startsWith('|turn|')) turn = Number(line.slice(6));
    }
    if (requestId !== record.request_id || turn !== record.turn || r1Hash(request) !== record.request_hash || record.phase !== requestPhase(request)) {
      throw new TypeError('R1 decision boundary mismatch');
    }
    const view = playerViewArtifact({side: record.side, request_id: requestId, turn, request, protocol: lines}, manifestId);
    views.push(view);
    decisions.push(makeArtifact('decision', manifestId, {
      episode_key: episodeKey, sequence, view_id: view.artifact_id, side: record.side,
      request_id: requestId, turn, phase: record.phase,
      submitted_action: fromProtocol(record.choice, request), accepted: record.accepted,
      feedback: structuredClone(record.errors),
    }));
  }
  const winner = source.outcome.winner === '' ? null : {'Player 1': 'p1', 'Player 2': 'p2'}[source.outcome.winner];
  if (winner === undefined) throw new TypeError('Unknown reference winner');
  const episode = makeArtifact('episode', manifestId, {
    episode_key: episodeKey, format_id: source.spec.format_id, team_ids: teams.map(t => t.artifact_id),
    mechanics_seed: seedObject(source.spec.mechanics_seed), policy_seeds: source.spec.policy_seeds.map(seedObject),
    policy_id: source.spec.policy_id, split: 'unassigned', decision_ids: decisions.map(d => d.artifact_id),
    input_tape: [...source.input_tape],
    outcome: {winner, turns: source.outcome.turns, remaining_pokemon: [...source.outcome.score], reason: winner === null ? 'tie' : 'win'},
    source_r1_content_hash: content_hash, source_raw_output_hash: raw_output_hash,
    source_normalized_output_hash: source.normalized_output_hash,
  });
  return {manifest, teams, views, decisions, episode};
}
