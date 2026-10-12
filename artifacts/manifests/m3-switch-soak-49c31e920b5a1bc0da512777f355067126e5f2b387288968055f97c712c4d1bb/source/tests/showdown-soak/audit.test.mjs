import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {WireAudit} from '../../packages/showdown-soak/audit.mjs';

const fixture = () => JSON.parse(gunzipSync(readFileSync(new URL('./fixtures/network.json.gz', import.meta.url))));
const run = data => { const a = new WireAudit(true, data.teams); for (const game of data.games) a.check(game); return a.finish(); };
const firstDecision = game => game.events.find(e => e.role === 'tactical' && e.type === 'decision');

test('actual two-session wire evidence checks both policies, seats, rematches and all fault classes', () => {
  const result = run(fixture()); assert.equal(result.status, 'pass'); assert.equal(result.coverage.games, 8);
  assert.equal(result.coverage.faults, 6); assert.equal(result.coverage.rematches, 6);
  assert(result.decisions.tactical > 50 && result.decisions.baseline > 50);
});
test('actor memory or own request corruption is rejected for either player', () => {
  for (const role of ['tactical', 'baseline']) for (const key of ['state', 'request']) {
    const data = fixture(), e = data.games[0].events.find(e => e.role === role && e.type === 'decision');
    e.actor_input[key].invented = true; assert.throws(() => run(data), /Actor input\/action mismatch/);
  }
});
test('accepted pending actions cannot be resent after recovery', () => {
  const data = fixture(), game = data.games.find(g => g.spec.scenario === 'accepted-choice');
  const sent = game.events.find(e => e.role === 'tactical' && e.type === 'send');
  const restored = game.events.findIndex(e => e.type === 'restored');
  const at = game.events.findIndex((e, i) => i > restored && e.role === 'tactical' && e.type === 'ready');
  game.events.splice(at + 1, 0, {...sent, generation: game.events[at].generation, at_ms: game.events[at].at_ms});
  assert.throws(() => run(data), /Duplicate or accepted choice replayed/);
});
test('changed team uploads and policy-seat claims are rejected', () => {
  const data = fixture(); data.games[0].events.find(e => e.type === 'team').packed = 'invented';
  assert.throws(() => run(data), /Uploaded team/);
  const other = fixture(); other.games[0].spec.tactical_seat = 'p2'; assert.throws(() => run(other), /Schedule mismatch/);
});
test('received simulator team receipts must agree with the admitted own team', () => {
  const data = fixture();
  const event = data.games[0].events.find(e => e.type === 'receive' && e.messages.some(m => m.line.startsWith('|soakteam|')));
  event.messages.find(m => m.line.startsWith('|soakteam|')).line += 'invented';
  assert.throws(() => run(data), /Received simulator team/);
});
test('held opponent action must be released after the accepted tactical action is restored', () => {
  const data = fixture(), game = data.games.find(g => g.events.some(e => e.type === 'deliver'));
  assert(game, 'Live fixture must exercise the deterministic hold');
  game.events = game.events.filter(e => e.type !== 'deliver'); assert.throws(() => run(data), assert.AssertionError);
});
test('accepted-choice control forbids direct baseline delivery before restoration', () => {
  const data = fixture(), game = data.games.find(g => g.events.some(e => e.type === 'deliver'));
  const send = game.events.find(e => e.role === 'baseline' && e.type === 'send' && e.held);
  send.held = false; send.delivered = true;
  assert.throws(() => run(data), /Baseline delivered before tactical restoration/);
});
test('a lost write cannot be relabeled delivered', () => {
  const data = fixture(), game = data.games.find(g => g.spec.scenario === 'lost-write');
  game.events.find(e => e.type === 'send' && !e.delivered).delivered = true; assert.throws(() => run(data), assert.AssertionError);
});
test('missing games and duplicated battles fail complete coverage and identity checks', () => {
  const data = fixture(); data.games.pop(); assert.throws(() => run(data), /Complete coverage/);
  const other = fixture(); other.games[1].results.forEach(r => { r.room = other.games[0].results[0].room; });
  assert.throws(() => run(other), /Duplicate battle/);
});
test('changed terminal counts and outcomes are rejected', () => {
  for (const key of ['decisions', 'retries', 'turns', 'winner']) {
    const data = fixture(); data.games[0].results[0][key] = key === 'winner' ? 'invented' : 99999;
    assert.throws(() => run(data), assert.AssertionError);
  }
});
test('incorrect connection and session ledger telemetry is rejected', () => {
  for (const key of ['rooms', 'finished', 'generations', 'reconnects']) {
    const data = fixture(); data.games[0].telemetry.roles.tactical[key]++; assert.throws(() => run(data), assert.AssertionError);
  }
});
test('forged request ID cannot validate an attempted action', () => {
  const data = fixture(), game = data.games[0], sent = game.events.find(e => e.type === 'send' && e.role === 'tactical');
  sent.data = sent.data.replace(/\|\d+$/, '|99999'); assert.throws(() => run(data), assert.AssertionError);
});
test('invalid event time and memory telemetry cannot pass', () => {
  const data = fixture(); firstDecision(data.games[0]).at_ms = -1; assert.throws(() => run(data), /Event clock order/);
  const other = fixture(); other.games[0].telemetry.rss = 769 * 2 ** 20; assert.throws(() => run(other), /RSS budget/);
});
test('the named login must precede battle traffic', () => {
  const data = fixture(); data.games[0].events = data.games[0].events.filter(e => e.type !== 'identity');
  assert.throws(() => run(data), /Login readiness|Battle before named login/);
});
test('an unexpected retry class cannot be counted as legitimate hidden legality', () => {
  const data = fixture(), event = data.games[0].events.find(e => e.type === 'receive');
  event.messages.push({room: data.games[0].results[0].room, line: '|error|[Invalid choice] forged'});
  assert.throws(() => run(data), /Unexpected choice error/);
});
