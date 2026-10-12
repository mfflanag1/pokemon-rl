import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, rmSync, readFileSync, writeFileSync, existsSync, readdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {campaign} from '../../packages/episodes/campaign.mjs';
import {writerLock} from '../../packages/episodes/storage.mjs';
import {readJSON} from '../../packages/corpus/common.mjs';

test('campaign checkpoints replayed episodes, resumes exact indices and retains failure/orphan evidence', async () => {
  const out = mkdtempSync(join(tmpdir(), 'r4-campaign-'));
  const options = {out, namespace: 'r4-recovery-test', count: 2, workers: 1};
  try {
    const unlock = writerLock(out);
    await assert.rejects(campaign(options), /active/);
    unlock();
    await assert.rejects(campaign({...options, onProgress: () => { throw new Error('injected controller interruption'); }}), /injected/);
    const checkpoint = readJSON(`${out}/progress.json`);
    assert.equal(checkpoint.completed, 1);
    assert.equal(checkpoint.episodes[0].replay_verified, true);
    assert.equal(existsSync(`${out}/index.json`), false);
    const first = readFileSync(`${out}/episode-000000.json.gz`);
    writeFileSync(`${out}/episode-000001.json.gz`, 'uncommitted orphan');
    const complete = await campaign(options);
    assert.equal(complete.completed, 2);
    assert.equal(complete.failure_events, 1);
    assert.deepEqual(complete.episodes.map(r => r.index), [0, 1]);
    assert.deepEqual(readFileSync(`${out}/episode-000000.json.gz`), first);
    assert.equal(readdirSync(`${out}/orphans`).length, 1);
    assert.deepEqual(await campaign({...options, workers: 2}), complete);
    await assert.rejects(campaign({...options, namespace: 'wrong'}), /configuration/);
    writeFileSync(`${out}/episode-000000.json.gz`, 'corrupt committed bytes');
    await assert.rejects(campaign(options), /corrupted/);
  } finally { rmSync(out, {recursive: true, force: true}); }
});
