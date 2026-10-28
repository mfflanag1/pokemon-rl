import test from 'node:test';
import assert from 'node:assert/strict';
import {AuditSession} from '../../packages/battle-audit/session.mjs';
import {FullTrainingSession} from '../../packages/full-training/session.mjs';
import {describeDecision} from '../../packages/battle-audit/describe.mjs';

test('audit keeps exact official projection and own-seat advice isolation', async () => {
  const audit = new AuditSession(), base = new FullTrainingSession();
  try {
    const options = {episodeIndex:0,runSeed:10222027,split:'development',swap:false};
    const a = await audit.start(options), b = await base.start(options);
    assert.deepEqual(a.players,b.players);
    assert.deepEqual(a.provenance,b.provenance);
    const own = audit.projection.players.p1;
    const first = describeDecision(own.request,audit.auditViews.p1,a.players.p1,a.tactical_actions.p1);
    // Real private opponent stats change, without any public protocol event.
    const foreign = audit.projection.players.p2.request.side.pokemon[0];
    const old = foreign.stats.atk;
    foreign.stats.atk = old + 10000;
    assert.notEqual(foreign.stats.atk,old);
    const second = describeDecision(own.request,audit.auditViews.p1,a.players.p1,a.tactical_actions.p1);
    assert.deepEqual(second,first);
    assert.equal(Object.hasOwn(a.players.p1,'tactical_actions'),false);
    assert.equal(Object.hasOwn(a.players.p1,'diagnostics'),false);
  } finally {audit.close();base.close();}
});

test('reserved split and supplied engine plan are rejected', async () => {
  for (const options of [{episodeIndex:0,runSeed:42,split:'final'},
    {episodeIndex:0,runSeed:42,split:'train',plan:{}}]) {
    const s=new AuditSession();
    try {await assert.rejects(s.start(options));} finally {s.close();}
  }
});
