import {pathToFileURL} from 'node:url';
import {BattleStream} from '../reference-node/engine.mjs';
import {PlayerProjection} from '../reference-node/observations.mjs';
import {legalChoices} from '../showdown-bot/protocol.mjs';
import {PLAN, curriculumEpisode, verifyPublicEvidence} from './index.mjs';

/** Bounded ten-prefix evidence capture; no learner, outcome search, or file writes. */
export async function captureCurriculumEvidence() {
  const records = [];
  for (let episodeIndex = 0; episodeIndex < PLAN.scenario_ids.length; episodeIndex++) {
    const plan = curriculumEpisode({episodeIndex, runSeed: 'bounded-c1-fixtures-v1'});
    const projection = new PlayerProjection(); let ended = false;
    const stream = new class extends BattleStream {
      constructor() { super({noCatch: true}); }
      pushMessage(type, data) { projection.ingest(type, data); if (type === 'end') ended = true; }
    }();
    try {
      await stream.write(`>start ${JSON.stringify({formatid: 'gen9randombattle', seed: plan.seed})}\n` +
        plan.teams.map((t, i) => `>player p${i + 1} ${JSON.stringify({name: `Curriculum ${i + 1}`, team: t.team})}`).join('\n'));
      for (const row of plan.prefix) {
        const pending = ['p1', 'p2'].filter(side => projection.players[side].pending);
        if (ended || pending.join(',') !== Object.keys(row).sort().join(',')) throw new Error('Curriculum prefix request group mismatch');
        for (const side of pending) {
          if (!legalChoices(projection.players[side].request).includes(row[side])) throw new Error('Unavailable scripted choice');
          projection.players[side].pending = false;
        }
        await stream.write(pending.map(side => `>${side} ${row[side]}`).join('\n'));
        if (projection.errors.length) throw new Error('Scripted prefix produced rejected action');
      }
      if (ended) throw new Error('Prefix ended battle before learner handoff');
      records.push({scenario_id: plan.provenance.scenario_id, seed: plan.seed,
        team_keys: plan.teams.map(t => t.team_key), prefix: plan.prefix,
        evidence: verifyPublicEvidence(projection.spectator, plan.evidence_requirements),
        handoff_turns: ['p1', 'p2'].map(side => projection.players[side].turn),
        public_protocol: projection.spectator});
    } finally { stream.battle?.destroy(); }
  }
  return {schema_version: 'training-curriculum-prefix-evidence-1', catalog_sha256: PLAN.catalog_sha256,
    showdown_commit: PLAN.showdown_commit, run_seed: 'bounded-c1-fixtures-v1', records,
    training_performed: false, final_evaluation_used: false, outcomes_used_for_selection: false};
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.length !== 2) throw new Error('Evidence capture takes no arguments');
  console.log(JSON.stringify(await captureCurriculumEvidence(), null, 2));
}
