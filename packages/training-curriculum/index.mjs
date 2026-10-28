import {readFileSync} from 'node:fs';
import {generateTeam, seedFor} from '../reference-node/engine.mjs';
import {teamArtifact} from '../reference-node/contract-v1.mjs';
import {hashJSON} from '../battle-contract/canonical.mjs';
import {partition, teamKey, POLICY_HASH} from '../dataset/identity.mjs';

function freeze(value) {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
export const PLAN = freeze(JSON.parse(readFileSync(new URL('../../contracts/training-curriculum-v1/plan.json', import.meta.url))));
export const CATALOG = freeze(JSON.parse(readFileSync(new URL('../../contracts/training-curriculum-v1/catalog.json', import.meta.url))));
if (PLAN.schema_version !== 'training-curriculum-plan-1' || CATALOG.schema_version !== 'training-curriculum-catalog-1' ||
    hashJSON(CATALOG) !== PLAN.catalog_sha256 || POLICY_HASH !== PLAN.r7_policy_sha256 ||
    CATALOG.showdown_commit !== PLAN.showdown_commit) throw new Error('Curriculum contract or frozen partition drift');
const roster = CATALOG.scenarios.map(s => s.id);
if (hashJSON(roster) !== hashJSON(PLAN.scenario_ids) || new Set(roster).size !== roster.length) throw new Error('Curriculum roster drift');
const byId = new Map(CATALOG.scenarios.map(s => [s.id, s]));

/** Accept only fresh train-partition records. Rejected candidate bytes are not exported. */
export function admitTrainingTeam(entry, source = 'constructed') {
  if (!['official', 'constructed'].includes(source)) throw new Error('Unknown curriculum team source');
  if (!Array.isArray(entry?.team) || entry.team.length !== 6) throw new Error('Curriculum requires exactly six members');
  if (entry.team_hash !== hashJSON(entry.team)) throw new Error('Team content hash mismatch');
  const payload = teamArtifact(entry, '0'.repeat(64)).payload;
  if (partition(payload) !== 'train') return null;
  return {...structuredClone(entry), source, partition: 'train', team_key: teamKey(payload)};
}

for (const scenario of CATALOG.scenarios) {
  if (scenario.teams.length !== 2 || !scenario.leakage_group.startsWith('training-curriculum-v1/family/')) throw new Error('Invalid scenario family');
  for (const team of scenario.teams) {
    const admitted = admitTrainingTeam(team);
    if (!admitted || admitted.team_key !== team.team_key || team.partition !== 'train') throw new Error('Frozen constructed team is not train-admitted');
  }
}

/** Scheduling is independent of RNG, outcomes, request contents, and learner state. */
export function curriculumSlot(episodeIndex) {
  if (!Number.isSafeInteger(episodeIndex) || episodeIndex < 0 || episodeIndex >= PLAN.episode_budget) throw new Error('Episode index outside frozen curriculum budget');
  const stage = PLAN.stages.find(s => episodeIndex >= s.start && episodeIndex < s.end);
  const local = episodeIndex - stage.start, slot = local % PLAN.block_size;
  const targeted = slot < stage.targeted_per_block;
  const ordinal = targeted ? stage.targeted_before + Math.floor(local / PLAN.block_size) * stage.targeted_per_block + slot : null;
  return freeze({episode_index: episodeIndex, stage: stage.id, source: targeted ? 'targeted' : 'official',
    scenario_id: targeted ? roster[ordinal % roster.length] : null, targeted_ordinal: ordinal});
}

function officialTeam(runKey, episodeIndex, side, excludedKey = null) {
  for (let attempt = 0; attempt < PLAN.official_team_candidate_budget; attempt++) {
    const entry = generateTeam(seedFor(`${PLAN.namespace}/official/${runKey}/${episodeIndex}/${side}/${attempt}`));
    const admitted = admitTrainingTeam(entry, 'official');
    if (admitted && admitted.team_key !== excludedKey) return {...admitted, admission_attempt: attempt};
  }
  throw new Error('Official train-team admission budget exhausted; no replacement episode');
}

/** Coordinator-only plan: never pass this object, IDs, or evidence targets to an actor. */
export function curriculumEpisode({episodeIndex, runSeed}) {
  if (typeof runSeed !== 'string' || !runSeed || runSeed.length > 256) throw new Error('runSeed must be a nonempty string of at most 256 characters');
  const slot = curriculumSlot(episodeIndex), runKey = hashJSON(['training-curriculum-run-1', runSeed]);
  const scenario = slot.scenario_id ? byId.get(slot.scenario_id) : null;
  let teams;
  if (scenario) teams = structuredClone(scenario.teams);
  else {
    const first = officialTeam(runKey, episodeIndex, 'p1');
    teams = [first, officialTeam(runKey, episodeIndex, 'p2', first.team_key)];
  }
  return freeze({teams, seed: seedFor(`${PLAN.namespace}/mechanics/${runKey}/${episodeIndex}`),
    prefix: structuredClone(scenario?.prefix ?? []), evidence_requirements: structuredClone(scenario?.evidence_requirements ?? []),
    provenance: {source: slot.source, curriculum_version: PLAN.version, curriculum_plan_sha256: hashJSON(PLAN),
      catalog_sha256: PLAN.catalog_sha256, r7_policy_sha256: POLICY_HASH, showdown_commit: PLAN.showdown_commit,
      split: 'train', episode_index: episodeIndex, stage: slot.stage, scenario_id: slot.scenario_id,
      family_id: scenario?.family ?? null, synthetic_leakage_group: scenario?.leakage_group ?? null,
      counterfactual_group: scenario?.counterfactual_group ?? null, run_seed_sha256: runKey,
      prefix_actions_trainable: false, source_distribution: scenario ? 'constructed-six-member-targeted' : 'official-whole-team-generator-conditioned-on-r7-train',
      complete_battle_continuation: true, outcome_selected: false}});
}

/** Coordinator audit only. Evidence is realized spectator protocol, never full-state assertions. */
export function verifyPublicEvidence(spectator, requirements) {
  if (!Array.isArray(spectator) || spectator.some(line => typeof line !== 'string')) throw new Error('Expected projected spectator lines');
  if (spectator.some(line => line.startsWith('|split|') || line.startsWith('|request|'))) throw new Error('Unprojected/private protocol is not public evidence');
  return requirements.map(requirement => {
    const reveals = spectator.filter(line => ['switch', 'drag', 'replace'].includes(line.split('|')[1]) && line.split('|')[2]?.startsWith(`${requirement.side}a:`));
    if (requirement.kind === 'revealed-count') {
      const count = new Set(reveals.map(line => line.split('|')[2])).size;
      if (count !== requirement.count) throw new Error(`Expected ${requirement.count} revealed members on ${requirement.side}, observed ${count}`);
      return {requirement, observed_count: count};
    }
    if (requirement.kind === 'unrevealed') {
      if (reveals.some(line => line.split('|')[3]?.split(',')[0] === requirement.species)) throw new Error('Hidden reserve already revealed');
      return {requirement, observed_reveals: reveals};
    }
    if (requirement.kind !== 'event') throw new Error('Unknown evidence requirement');
    const matches = spectator.filter(line => {
      const [, event, ident, value] = line.split('|');
      return event === requirement.event && ident?.startsWith(requirement.side) &&
        (!requirement.value || (event === 'switch' ? value?.split(',')[0] : value) === requirement.value) &&
        (!requirement.contains || line.includes(requirement.contains));
    });
    if (!matches.length) throw new Error(`Missing realized ${requirement.event} evidence for ${requirement.side}`);
    return {requirement, matches};
  });
}
