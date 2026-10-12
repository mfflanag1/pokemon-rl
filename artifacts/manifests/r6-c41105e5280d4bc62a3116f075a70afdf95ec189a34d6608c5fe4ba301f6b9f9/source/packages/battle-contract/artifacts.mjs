import {canonical, hashJSON, parseJSON} from './canonical.mjs';
import {validateSchema} from './schema.mjs';
import {admissibleActions, actionMask, requestPhase} from './actions.mjs';

export function validateArtifact(artifact, {kind, manifestId} = {}) {
  validateSchema(artifact);
  const {schema_version, artifact_kind, manifest_id, payload, content_hash, artifact_id} = artifact;
  if (kind && artifact_kind !== kind) throw new TypeError(`Expected ${kind} artifact`);
  if (manifestId && manifest_id !== manifestId) throw new TypeError('Manifest mismatch');
  if (hashJSON({schema_version, artifact_kind, manifest_id, payload}) !== content_hash) throw new TypeError('Content hash mismatch');
  if (artifact_id !== `${artifact_kind}:${content_hash}`) throw new TypeError('Artifact ID mismatch');
  if (artifact_kind === 'manifest' && manifest_id !== hashJSON(payload)) throw new TypeError('Manifest identity mismatch');
  if (artifact_kind === 'manifest') {
    const {manifest_id: sourceId, ...source} = payload.reference_manifest;
    if (hashJSON(source) !== sourceId || source.commit !== source.build.commit || source.lock_sha256 !== source.build.lock_sha256) {
      throw new TypeError('Reference manifest identity mismatch');
    }
  }
  if (artifact_kind === 'team' && payload.source_team_hash !== hashJSON(payload.pokemon)) throw new TypeError('Source team hash mismatch');
  if (artifact_kind === 'player_view') {
    if (payload.request.side.id !== payload.viewer || payload.request.side.pokemon.some(mon => !mon.ident.startsWith(`${payload.viewer}: `))) {
      throw new TypeError('Private request belongs to another viewer');
    }
    if (payload.phase !== requestPhase(payload.request) ||
        canonical(payload.admissible_actions) !== canonical(admissibleActions(payload.request)) ||
        canonical(payload.action_mask) !== canonical(actionMask(payload.request))) throw new TypeError('Request/action mask mismatch');
    let lastRequest = null, requestCount = 0, turn = 0;
    for (const line of payload.visible_protocol) {
      if (line.startsWith('|request|')) {
        lastRequest = parseJSON(line.slice(9)); requestCount++;
        if (lastRequest?.side?.id !== payload.viewer) throw new TypeError('Foreign private request in player history');
      }
      if (line.startsWith('|turn|')) {
        if (!/^\|turn\|[0-9]+$(?![\s\S])/.test(line)) throw new TypeError('Malformed turn record');
        turn = Number(line.slice(6));
      }
    }
    if (requestCount !== payload.request_id || turn !== payload.turn || canonical(lastRequest) !== canonical(payload.request)) {
      throw new TypeError('Player history/request boundary mismatch');
    }
  }
  if (artifact_kind === 'decision') {
    if (payload.accepted !== (payload.feedback.length === 0) || payload.feedback.some(e => e.side !== payload.side)) {
      throw new TypeError('Inconsistent decision feedback');
    }
  }
  if (artifact_kind === 'episode' && ((payload.outcome.winner === null) !== (payload.outcome.reason === 'tie'))) {
    throw new TypeError('Inconsistent terminal reason');
  }
  return artifact;
}

export function makeArtifact(kind, manifestId, payload) {
  const manifest_id = kind === 'manifest' ? hashJSON(payload) : manifestId;
  const body = {schema_version: '1.0.0', artifact_kind: kind, manifest_id, payload};
  const content_hash = hashJSON(body);
  return validateArtifact({...body, content_hash, artifact_id: `${kind}:${content_hash}`});
}

export const loadArtifact = (text, options) => validateArtifact(parseJSON(text), options);
export const loadPlayerView = (text, manifestId) => loadArtifact(text, {kind: 'player_view', manifestId}).payload;

export function vocabularyIndex(vocabulary, domain, knowledge) {
  validateSchema(knowledge, 'known_id');
  const entries = vocabulary.domains[domain];
  if (!entries || entries[0] !== '<unknown>' || entries[1] !== '<none>') throw new TypeError('Invalid vocabulary domain');
  if (knowledge.state === 'unknown') return 0;
  if (knowledge.value === '') return 1;
  const index = entries.indexOf(knowledge.value);
  if (index < 2) throw new TypeError(`Unknown known ID: ${knowledge.value}`);
  return index;
}
