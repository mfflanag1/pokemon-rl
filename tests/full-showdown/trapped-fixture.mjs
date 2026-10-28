import {fileURLToPath} from 'node:url';
import {Dex, seedFor} from '../../packages/reference-node/engine.mjs';
import {teamArtifact} from '../../packages/reference-node/contract-v1.mjs';
import {hashJSON} from '../../packages/reference-node/provenance.mjs';
import {partition, teamKey} from '../../packages/dataset/identity.mjs';
import {FullShowdownSession} from '../../packages/full-showdown/session.mjs';

const namespace = 'full-showdown-f1/unit/hidden-arena-trap-v1';
const stats = value => Object.fromEntries(['hp', 'atk', 'def', 'spa', 'spd', 'spe'].map(stat => [stat, value]));
const mon = (species, ability) => ({name: species, species, speciesId: Dex.species.get(species).id,
  gender: '', shiny: false, level: 80, moves: ['protect'], ability, item: '', teraType: 'Water',
  role: 'Constructed protocol fixture', evs: stats(0), ivs: stats(31)});

function admittedTeam(lead, side) {
  const team = [lead, mon('Raichu', 'Static'), mon('Snorlax', 'Immunity'),
    mon('Vaporeon', 'Water Absorb'), mon('Umbreon', 'Synchronize'), mon('Espeon', 'Synchronize')];
  const entry = {seed: seedFor(`${namespace}/${side}`), team, team_hash: hashJSON(team)};
  const payload = teamArtifact(entry, '0'.repeat(64)).payload;
  // Fixed constructed sets: no outcome, mechanics seed, or candidate search.
  if (partition(payload) !== 'train') throw new Error('Fixed trapping fixture must be train admitted');
  return {...entry, partition: 'train', team_key: teamKey(payload)};
}

/** Actual reference replies, shared by Node assertions and the Python collector test. */
export async function trappedFrames() {
  const session = new FullShowdownSession();
  try {
    const plan = {teams: [admittedTeam(mon('Pikachu', 'Static'), 'p1'), admittedTeam(mon('Dugtrio', 'Arena Trap'), 'p2')],
      seed: seedFor(`${namespace}/mechanics`), provenance: {source: 'constructed-unit-fixture', partition: 'train', namespace}};
    const initial = await session.start({episodeIndex: 0, runSeed: 73, plan});
    const first = {actions: {p1: 9, p2: 0},
      request_ids: {p1: initial.players.p1.request_id, p2: initial.players.p2.request_id}};
    const rejected = await session.step(first);
    const retry = {actions: {p1: 0}, request_ids: {p1: rejected.players.p1.request_id}};
    const continued = await session.step(retry);
    const cutoff = session.cutoff();
    return {initial, rejected, continued, cutoff, commands: [first, retry]};
  } finally { session.close(); }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  process.stdout.write(`${JSON.stringify(await trappedFrames())}\n`);
}
