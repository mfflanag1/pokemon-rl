import fs from 'node:fs';
import {officialPlan} from '../full-training/session.mjs';

const jobs = JSON.parse(fs.readFileSync(0, 'utf8'));
if (!Array.isArray(jobs) || jobs.length > 5000) throw new Error('Invalid fixed schedule inventory');
for (const job of jobs) {
  if (job.swap !== false || !['train', 'development'].includes(job.split)) throw new Error('Invalid schedule admission');
  const plan = officialPlan(job.episode, job.seed, job.split, false);
  process.stdout.write(JSON.stringify({...job, mechanics_seed: plan.seed,
    teams: plan.teams.map(t => ({team_key: t.team_key, partition: t.partition, seed: t.seed}))}) + '\n');
}
