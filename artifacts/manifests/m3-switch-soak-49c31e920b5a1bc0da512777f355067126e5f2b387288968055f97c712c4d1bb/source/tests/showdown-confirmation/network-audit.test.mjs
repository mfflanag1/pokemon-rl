import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {auditTrace} from '../../packages/showdown-confirmation/network-audit.mjs';

const fixture = () => JSON.parse(gunzipSync(readFileSync(new URL('./fixtures/network.json.gz', import.meta.url))));
test('actual tactical local-server trace reconstructs all six games and seven losses', () => {
  const {report, trace} = fixture(), result = auditTrace(report, trace);
  assert.equal(result.games, 6); assert.equal(result.faults, 7); assert.equal(result.actor_inputs_and_actions_checked, 174);
});
test('wire audit rejects altered tactical memory and reported choices', () => {
  for (const mutate of [e => { e.actor_input.state.weather = 'invented'; }, e => { e.choice = 'switch 6'; }]) {
    const {report, trace} = fixture(); mutate(trace.find(e => e.type === 'decision'));
    assert.throws(() => auditTrace(report, trace), assert.AssertionError);
  }
});
test('wire audit rejects resubmission of a server-confirmed accepted action', () => {
  const {report, trace} = fixture();
  const at = trace.findIndex(e => e.scenario === 'accepted-choice' && e.type === 'restored' && e.committed);
  const before = trace.find(e => e.scenario === 'accepted-choice' && e.type === 'send');
  trace.splice(at + 1, 0, {...before, generation: trace[at].generation});
  assert.throws(() => auditTrace(report, trace), /Duplicate or accepted/);
});
test('wire audit rejects changed counters, terminals and policy identity', () => {
  for (const mutate of [r => { r.records[0].decisions++; }, r => { r.records[0].results[0].winner = 'invented'; }, r => { r.policy = 'baseline-v1'; }]) {
    const {report, trace} = fixture(); mutate(report);
    assert.throws(() => auditTrace(report, trace), assert.AssertionError);
  }
});
