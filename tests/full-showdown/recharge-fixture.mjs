import {fileURLToPath} from 'node:url';
import {Dex, seedFor} from '../../packages/reference-node/engine.mjs';
import {teamArtifact} from '../../packages/reference-node/contract-v1.mjs';
import {hashJSON} from '../../packages/reference-node/provenance.mjs';
import {partition, teamKey} from '../../packages/dataset/identity.mjs';
import {FullShowdownSession} from '../../packages/full-showdown/session.mjs';

const namespace = 'full-showdown-f1/unit/recharge-v2';
const stats = value => Object.fromEntries(['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map(stat => [stat, value]));
const mon = (species, ability, moves = ['protect']) => ({name: species, species,
  speciesId: Dex.species.get(species).id, gender: '', shiny: false, level: 80,
  moves, ability, item: '', teraType: 'Water', role: 'Constructed protocol fixture',
  evs: stats(0), ivs: stats(31)});

function admittedTeam(lead, side) {
  const reserves = [mon('Raichu', 'Static'), mon('Snorlax', 'Immunity'),
    mon('Vaporeon', 'Water Absorb'), mon('Umbreon', 'Synchronize'), mon('Espeon', 'Synchronize')];
  // Deterministic reserve-order admission only, before running any battle.
  // Neither the mechanic setup nor any seed depends on simulated outcomes.
  for (let rotation = 0; rotation < reserves.length; rotation++) {
    const team = [lead, ...reserves.slice(rotation), ...reserves.slice(0, rotation)];
    const entry = {seed: seedFor(`${namespace}/${side}`), team, team_hash: hashJSON(team)};
    const payload = teamArtifact(entry, '0'.repeat(64)).payload;
    if (partition(payload) === 'train') return {...entry, partition: 'train', team_key: teamKey(payload)};
  }
  throw new Error('Fixed recharge fixture train admission exhausted');
}

/** No Guard makes Hyper Beam connect without selecting a favorable mechanics seed. */
export async function rechargeFrames() {
  const session = new FullShowdownSession();
  try {
    const plan = {teams: [admittedTeam(mon('Machamp', 'No Guard', ['hyperbeam']), 'p1'),
      admittedTeam(mon('Chansey', 'Natural Cure', ['defensecurl']), 'p2')],
    seed: seedFor(`${namespace}/mechanics`),
    provenance: {source: 'constructed-unit-fixture', partition: 'train', namespace}};
    const initial = await session.start({episodeIndex: 0, runSeed: 73, plan});
    const move = frame => ({actions: {p1: 0, p2: 0}, request_ids: {
      p1: frame.players.p1.request_id, p2: frame.players.p2.request_id}});
    const recharge = await session.step(move(initial));
    const rechargeRequest = structuredClone(session.projection.players.p1.request.active[0]);
    const resumed = await session.step(move(recharge));
    return {initial, recharge, rechargeRequest, resumed,
      publicEvidence: session.projection.spectator.filter(line => line.startsWith('|-mustrecharge|') || line.startsWith('|cant|'))};
  } finally { session.close(); }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  process.stdout.write(`${JSON.stringify(await rechargeFrames())}\n`);
}
