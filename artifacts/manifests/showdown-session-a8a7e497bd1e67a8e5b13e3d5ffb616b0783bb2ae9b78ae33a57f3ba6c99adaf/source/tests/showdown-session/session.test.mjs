import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as pause} from 'node:timers/promises';
import {RecoveringBot} from '../../packages/showdown-session/client.mjs';

const ID = 'battle-gen9randombattle-1';
const request = (rqid = 4) => ({rqid, side: {id: 'p1', name: 'RecoverBot', pokemon: [
  {ident: 'p1: Squirtle', details: 'Squirtle, L80', condition: '200/200', active: true,
    stats: {atk: 100, spa: 100}, moves: ['watergun']},
]}, active: [{moves: [{id: 'watergun', move: 'Water Gun', pp: 20}]}]});
const history = (turn = 1) => `>${ID}\n|init|battle\n|gametype|singles\n|tier|[Gen 9] Random Battle\n|switch|p2a: Foe|Fennekin, L80|100/100\n|turn|${turn}`;
const frame = (r, extra = '') => `>${ID}\n|request|${JSON.stringify(r)}\n${extra}`;
class Socket extends EventTarget {
  readyState = 1; sent = [];
  send(x) { this.sent.push(x); }
  close() { this.readyState = 3; }
  drop() { this.readyState = 3; this.dispatchEvent(new Event('close')); }
  message(data) { this.dispatchEvent(new MessageEvent('message', {data})); }
}
async function until(fn) {
  for (let i = 0; i < 100; i++) { if (fn()) return; await pause(5); }
  assert.fail('Timed out waiting for test condition');
}
function setup(t, options = {}) {
  const sockets = [], failures = [], restores = [], decisions = [], readyEvents = [];
  const bot = new RecoveringBot({name: 'RecoverBot', decisionDelayMs: 0, retryDelayMs: 0,
    socketFactory: () => { const s = new Socket(); sockets.push(s); return s; }, ...options});
  bot.on('failure', e => failures.push(e)); bot.on('restored', e => restores.push(e));
  bot.on('decision', e => decisions.push(e)); bot.on('ready', e => readyEvents.push(e));
  t.after(() => bot.close());
  const initial = bot.connect(); initial.catch(() => {});
  async function login(i = sockets.length - 1) {
    await until(() => sockets[i]); const s = sockets[i];
    s.message(`|challstr|4|challenge-${i}`); await until(() => s.sent.some(x => x.startsWith('|/trn ')));
    s.message('|updateuser| RecoverBot|1|1|{}'); await until(() => bot.connection.ready || bot.failed);
    return s;
  }
  return {bot, sockets, failures, restores, decisions, readyEvents, initial, login};
}

test('rebuilds history, submits an uncommitted current request once, cancels old timer', async t => {
  const c = setup(t, {decisionDelayMs: 30}); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame(request())); await until(() => c.bot.connection.rooms.size);
  s.drop(); const next = await c.login(1);
  assert(next.sent.includes(`|/join ${ID}`)); assert.equal(c.bot.ready, false);
  next.message(history()); next.message(frame(request()));
  await until(() => c.decisions.length === 1);
  assert.equal(s.sent.filter(x => x.includes('/choose')).length, 0);
  assert.equal(c.restores[0].committed, false); assert.equal(c.bot.ready, true);
  next.message(frame(request())); await pause(50); assert.equal(c.decisions.length, 1);
  assert.equal(c.failures.length, 0);
});

test('accepted pending choice is not replayed; next fresh request can act', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame(request())); await until(() => c.decisions.length === 1);
  s.drop(); const next = await c.login(1);
  next.message(history()); next.message(frame(request(), '|sentchoice|move 1'));
  await until(() => c.restores.length); await pause(20);
  assert.equal(c.restores[0].committed, true); assert.equal(c.decisions.length, 1);
  next.message(`>${ID}\n|turn|2\n|request|${JSON.stringify(request(5))}`);
  await until(() => c.decisions.length === 2);
  assert.equal(c.bot.decisions.get(ID), 2); assert.equal(c.failures.length, 0);
});

test('an unsent action is reconsidered from the server request, not a local sent marker', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame(request())); await until(() => c.decisions.length === 1);
  s.drop(); const next = await c.login(1);
  next.message(history() + '\n' + frame(request())); await until(() => c.decisions.length === 2);
  assert.equal(c.restores[0].committed, false); assert.equal(c.decisions[0].rqid, c.decisions[1].rqid);
});

test('wait and completed battles restore without submitting; terminal emits once', async t => {
  const c = setup(t); let ends = 0; c.bot.on('battleEnd', () => ends++);
  const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame({rqid: 4, wait: true})); await until(() => c.bot.connection.rooms.size);
  s.drop(); const next = await c.login(1);
  next.message(history() + '\n' + frame({rqid: 4, wait: true})); await until(() => c.restores.length);
  assert.equal(c.decisions.length, 0); assert(c.bot.ready);
  next.drop(); const last = await c.login(2);
  last.message(history(2) + '\n|win|RecoverBot'); await until(() => ends === 1);
  assert.equal(c.restores.at(-1).ended, true); assert.equal(c.decisions.length, 0);
  last.message(`>${ID}\n|win|RecoverBot`); await pause(10); assert.equal(ends, 1);
});

test('missing history is fatal and does not retry the protocol error', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  s.message(frame(request())); await until(() => c.failures.length);
  assert.match(c.failures[0].message, /history/); assert.equal(c.sockets.length, 1); assert.equal(c.decisions.length, 0);
});

test('request rollback, seat changes, and truncated public history fail before any action', async t => {
  for (const fault of ['rqid', 'seat', 'turn']) {
    const c = setup(t, {decisionDelayMs: 30}); const s = await c.login(); await c.initial;
    s.message(history(5) + '\n' + frame(request())); await until(() => c.bot.connection.rooms.size);
    s.drop(); const next = await c.login(1); const r = request(fault === 'rqid' ? 3 : 4);
    if (fault === 'seat') r.side.id = 'p2';
    next.message(history(fault === 'turn' ? 4 : 5) + '\n' + frame(r));
    await until(() => c.failures.length); await pause(35); assert.equal(c.decisions.length, 0);
    assert.equal(c.sockets.length, 2);
  }
});

test('missing rooms and silent rejoin responses fail on a bounded recovery deadline', async t => {
  for (const response of ['noinit', 'silent']) {
    const c = setup(t, {recoveryTimeoutMs: 30}); const s = await c.login(); await c.initial;
    s.message(history() + '\n' + frame({rqid: 4, wait: true})); await until(() => c.bot.connection.rooms.size);
    s.drop(); const next = await c.login(1);
    if (response === 'noinit') next.message(`>${ID}\n|noinit|nonexistent|The room is gone`);
    await until(() => c.failures.length); assert.match(c.failures[0].message, /recovery/);
    assert.equal(c.sockets.length, 2); assert.equal(c.decisions.length, 0);
  }
});

test('retry budget spans successful reconnects; stale socket messages are ignored', async t => {
  const c = setup(t, {maxReconnects: 1}); const s = await c.login(); await c.initial;
  s.drop(); const next = await c.login(1); await until(() => c.bot.ready);
  s.message(history() + '\n' + frame(request())); await pause(10); assert.equal(c.decisions.length, 0);
  next.drop(); await until(() => c.failures.length);
  assert.match(c.failures[0].message, /budget exhausted/); assert.equal(c.sockets.length, 2);
});

test('initial transport outage retries and shutdown cancels delayed recovery', async t => {
  const c = setup(t, {retryDelayMs: 20}); c.sockets[0].drop();
  await c.login(1); await c.initial; assert.equal(c.bot.reconnects, 1);
  c.sockets[1].drop(); c.bot.close(); await pause(60);
  assert.equal(c.sockets.length, 2); assert.equal(c.failures.length, 0);
});

test('each remote connection gets fresh authentication; permanent login errors do not retry', async t => {
  const calls = []; let reads = 0;
  const c = setup(t, {url: 'wss://example.com/showdown/websocket', passwordProvider: () => { reads++; return 'local-secret'; },
    authenticate: async (name, password, challenge) => { calls.push({name, password, challenge}); return `signed-${calls.length}`; }});
  const s = await c.login(); await c.initial; s.drop(); const next = await c.login(1);
  assert.equal(reads, 2); assert.deepEqual(calls.map(x => x.challenge), ['4|challenge-0', '4|challenge-1']);
  assert(next.sent.includes('|/trn RecoverBot,0,signed-2'));
  assert(!JSON.stringify([...c.readyEvents, ...c.restores, ...c.decisions]).includes('local-secret'));
  next.message('|nametaken|RecoverBot|Rejected'); await until(() => c.failures.length);
  assert.equal(c.sockets.length, 2); assert.match(c.failures[0].message, /unavailable/);
});

test('pending restoration prevents accepting or initiating another challenge', async t => {
  const c = setup(t, {accept: ['Friend']}); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame({rqid: 4, wait: true})); await until(() => c.bot.connection.rooms.size);
  s.drop(); const next = await c.login(1);
  next.message('|pm| Friend| RecoverBot|/challenge gen9randombattle||||'); await pause(10);
  assert(!next.sent.some(x => x.includes('/accept'))); assert.throws(() => c.bot.challenge('Friend'), /not ready/);
});

test('stateful policies and invalid or unbounded recovery configuration are rejected', () => {
  assert.throws(() => new RecoveringBot({policy: 'random'}), /stateless/);
  for (const options of [{maxReconnects: Infinity}, {maxReconnects: 11}, {retryDelayMs: -1}, {recoveryTimeoutMs: 0}]) {
    assert.throws(() => new RecoveringBot(options), /budget/);
  }
});

test('multiple pending rooms restore separately before session readiness', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  const other = 'battle-gen9randombattle-2';
  const waiting = history() + '\n' + frame({rqid: 4, wait: true});
  s.message(waiting + '\n' + waiting.replaceAll(ID, other)); await until(() => c.bot.connection.rooms.size === 2);
  s.drop(); const next = await c.login(1);
  assert(next.sent.includes(`|/join ${ID}`)); assert(next.sent.includes(`|/join ${other}`));
  next.message(waiting); await until(() => c.restores.length === 1); assert.equal(c.bot.ready, false);
  next.message(waiting.replaceAll(ID, other)); await until(() => c.restores.length === 2);
  assert(c.bot.ready); assert.equal(c.decisions.length, 0); assert.equal(c.failures.length, 0);
});

test('invalid choices remain fatal rather than reconnecting into another action', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame(request())); await until(() => c.decisions.length);
  s.message(`>${ID}\n|error|[Invalid choice] bad action`); await until(() => c.failures.length);
  await pause(10); assert.equal(c.sockets.length, 1); assert.match(c.failures[0].message, /rejected/);
});

test('a send during the close handshake reconnects and does not count an unsent decision', async t => {
  const c = setup(t, {decisionDelayMs: 30}); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame(request())); await until(() => c.bot.connection.rooms.size);
  s.readyState = 2; // The close event has not arrived; the old decision timer will fire.
  const next = await c.login(1); assert.equal(c.failures.length, 0); assert.equal(c.decisions.length, 0);
  next.message(history() + '\n' + frame(request())); await until(() => c.decisions.length === 1);
  assert.equal(c.bot.decisions.get(ID), 1); assert.equal(c.failures.length, 0);
});

test('closing during authenticated rejoin retries through the transport classifier', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame({rqid: 4, wait: true})); await until(() => c.bot.connection.rooms.size);
  s.drop(); await until(() => c.sockets[1]); const closing = c.sockets[1];
  closing.message('|challstr|4|second'); await until(() => closing.sent.length);
  closing.readyState = 2; closing.message('|updateuser| RecoverBot|1|1|{}');
  const next = await c.login(2); next.message(history() + '\n' + frame({rqid: 4, wait: true}));
  await until(() => c.restores.length); assert(c.bot.ready); assert.equal(c.failures.length, 0);
});

test('a second disconnect after history but before the request retains the old resume floor', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  s.message(history(5) + '\n' + frame({rqid: 20, wait: true})); await until(() => c.bot.connection.rooms.size);
  s.drop(); const next = await c.login(1); next.message(history(5));
  await until(() => c.bot.connection.rooms.size); next.drop(); const last = await c.login(2);
  last.message(history(5) + '\n' + frame({rqid: 19, wait: true})); await until(() => c.failures.length);
  assert.match(c.failures[0].message, /backwards/); assert.equal(c.decisions.length, 0);
});

test('same-turn forced-switch request remains actionable after a committed restoration', async t => {
  const c = setup(t); const s = await c.login(); await c.initial;
  s.message(history() + '\n' + frame(request())); await until(() => c.decisions.length === 1);
  s.drop(); const next = await c.login(1); next.message(history() + '\n' + frame(request(), '|sentchoice|move 1'));
  await until(() => c.restores.length);
  const forced = request(5); delete forced.active; forced.forceSwitch = [true];
  forced.side.pokemon[0].condition = '0 fnt';
  forced.side.pokemon.push({ident: 'p1: Bulbasaur', details: 'Bulbasaur, L80', condition: '180/180', active: false, moves: ['vinewhip']});
  next.message(frame(forced)); await until(() => c.decisions.length === 2);
  assert.equal(c.decisions[1].turn, 1); assert.equal(c.decisions[1].choice, 'switch 2'); assert.equal(c.failures.length, 0);
});
