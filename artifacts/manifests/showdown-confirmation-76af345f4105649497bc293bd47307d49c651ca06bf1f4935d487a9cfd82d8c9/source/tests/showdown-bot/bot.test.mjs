import test from 'node:test';
import assert from 'node:assert/strict';
import {setTimeout as pause} from 'node:timers/promises';
import {legalChoices, parseFrame, PublicBattle} from '../../packages/showdown-bot/protocol.mjs';
import {chooseHeuristic} from '../../packages/showdown-bot/policy.mjs';
import {loginAssertion, ShowdownBot} from '../../packages/showdown-bot/client.mjs';

function request(rqid = 4) {
  return {rqid, side: {id: 'p1', name: 'BaselineBot', pokemon: [
    {ident: 'p1: Squirtle', details: 'Squirtle, L80', condition: '200/200', active: true,
      stats: {atk: 100, spa: 100}, moves: ['watergun', 'tackle']},
    {ident: 'p1: Bulbasaur', details: 'Bulbasaur, L80', condition: '180/180', active: false,
      stats: {atk: 100, spa: 100}, moves: ['vinewhip', 'tackle']},
    {ident: 'p1: Fennekin', details: 'Fennekin, L80', condition: '0 fnt', active: false, moves: ['ember']},
  ]}, active: [{moves: [{id: 'watergun', move: 'Water Gun', pp: 20}, {id: 'tackle', move: 'Tackle', pp: 20}], canTerastallize: 'Water'}]};
}
function fakeBot(t, opts = {}) {
  const sent = [], failures = [], bot = new ShowdownBot({decisionDelayMs: 0, ...opts});
  bot.socket = {readyState: 1, send: message => sent.push(message), close: () => {}};
  bot.ready = true; bot.on('failure', e => failures.push(e)); t.after(() => bot.close());
  return {bot, sent, failures};
}
const frame = (req, extra = '') => `>battle-gen9randombattle-1\n|init|battle\n|gametype|singles\n|tier|[Gen 9] Random Battle\n|switch|p2a: Foe|Fennekin, L80|100/100\n|request|${JSON.stringify(req)}\n${extra}`;

test('frame parser isolates rooms, including global messages and malicious room IDs', () => {
  assert.deepEqual(parseFrame('|challstr|4|abc\n>battle-one\n|turn|2\n>battle-two\n|win|B'), [
    {room: '', line: '|challstr|4|abc'}, {room: 'battle-one', line: '|turn|2'}, {room: 'battle-two', line: '|win|B'},
  ]);
  assert.throws(() => parseFrame('>x|/choose move 1\n|turn|2'));
});

test('legal singles choices handle Tera, disabled moves, trapping, wait and Struggle', () => {
  const r = request();
  assert.deepEqual(legalChoices(r), ['move 1', 'move 1 terastallize', 'move 2', 'move 2 terastallize', 'switch 2']);
  r.active[0].moves[0].disabled = true; r.active[0].trapped = true;
  assert.deepEqual(legalChoices(r), ['move 2', 'move 2 terastallize']);
  assert.deepEqual(legalChoices({wait: true}), []); assert.deepEqual(legalChoices(null), []);
  r.active[0] = {trapped: true, moves: [{id: 'struggle', pp: 1}]};
  assert.deepEqual(legalChoices(r), ['move 1']);
});

test('forced switch/revival use current request positions and opposite eligibility', () => {
  const r = request(); delete r.active; r.forceSwitch = [true];
  r.side.pokemon[0].condition = '0 fnt';
  assert.deepEqual(legalChoices(r), ['switch 2']);
  r.side.pokemon[0].condition = '50/200'; r.side.pokemon[0].reviving = true;
  assert.deepEqual(legalChoices(r), ['switch 3']);
  r.side.pokemon[1].condition = '0 fnt';
  assert.deepEqual(legalChoices(r), ['switch 2', 'switch 3']);
});

test('foreign requests and unsupported battle types reject', () => {
  const r = request(); r.side.id = 'p2'; assert.throws(() => legalChoices(r), /Foreign/);
  r.side.id = 'p1'; r.active.push(r.active[0]); assert.throws(() => legalChoices(r), /singles/);
  r.active.pop(); r.active[0].canDynamax = true; assert.throws(() => legalChoices(r), /Gen 9/);
});

test('heuristic responds to revealed type but not unobserved opponent state', () => {
  const state = new PublicBattle(); state.ingest('|switch|p2a: Foe|Fennekin, L80|100/100');
  assert.match(chooseHeuristic(request(), state), /^move 1(?: terastallize)?$/);
  const sameVisible = structuredClone(state);
  assert.equal(chooseHeuristic(request(), sameVisible), chooseHeuristic(request(), state));
  state.ingest('|-terastallize|p2a: Foe|Grass');
  assert.notEqual(chooseHeuristic(request(), state), 'move 1');
  assert.throws(() => state.ingest('|split|p1'), /Unprojected/);
});

test('public Tera remains known after switching away and back', () => {
  const state = new PublicBattle();
  state.ingest('|switch|p2a: Foe|Fennekin, L80|100/100');
  state.ingest('|-terastallize|p2a: Foe|Grass');
  state.ingest('|switch|p2a: Bench|Squirtle, L80|100/100');
  state.ingest('|switch|p2a: Foe|Fennekin, L80|100/100');
  assert.equal(state.active.p2.tera, 'Grass');
  assert.notEqual(chooseHeuristic(request(), state), 'move 1');
});

test('Illusion replacement without an HP field preserves visible HP and serializable input', () => {
  const state = new PublicBattle();
  state.ingest('|switch|p2a: Foe|Pikachu, L80|100/100');
  state.ingest('|-damage|p2a: Foe|63/100');
  state.ingest('|replace|p2a: Zoroark|Zoroark-Hisui, L80');
  assert.equal(state.active.p2.condition, '63/100');
  assert.deepEqual(JSON.parse(JSON.stringify(state)), {...state});
});

test('duplicate/stale requests submit once; wait and terminal submit nothing', async t => {
  const {bot, sent, failures} = fakeBot(t);
  await bot.receive(frame(request())); await pause(5);
  await bot.receive(frame(request())); await bot.receive(frame(request(3))); await pause(5);
  assert.equal(sent.filter(s => s.includes('/choose ')).length, 1);
  assert.match(sent[0], /\|\/choose move 1(?: terastallize)?\|4$/);
  await bot.receive(frame({rqid: 5, wait: true})); await pause(5);
  assert.equal(sent.length, 1);
  await bot.receive(frame(request(6), '|win|BaselineBot')); await pause(5);
  assert.equal(sent.length, 2); assert.match(sent.at(-1), /\/leave$/); assert.equal(failures.length, 0);
});

test('unavailable choice waits for a fresh request, then uses its new rqid', async t => {
  const {bot, sent} = fakeBot(t);
  await bot.receive(frame(request())); await pause(5);
  await bot.receive('>battle-gen9randombattle-1\n|error|[Unavailable choice] Hidden trapping'); await pause(5);
  assert.equal(sent.length, 1);
  const r = request(5); r.active[0].trapped = true; r.update = true;
  await bot.receive(frame(r)); await pause(5);
  assert.equal(sent.length, 2); assert.match(sent.at(-1), /\|5$/);
});

test('sentchoice history and revoked requests suppress replay', async t => {
  const {bot, sent} = fakeBot(t);
  await bot.receive(frame(request(), '|sentchoice|move 1')); await pause(5); assert.equal(sent.length, 0);
  await bot.receive(frame(request(5), '|request|null')); await pause(5); assert.equal(sent.length, 0);
});

test('two rooms never share request IDs or public state', async t => {
  const {bot, sent} = fakeBot(t);
  await bot.receive(frame(request()) + '\n' + frame(request()).replaceAll('battle-gen9randombattle-1', 'battle-gen9randombattle-2'));
  await pause(5); assert.equal(sent.length, 2);
  assert.notEqual(bot.rooms.get('battle-gen9randombattle-1').state, bot.rooms.get('battle-gen9randombattle-2').state);
});

test('modern challenge PMs and legacy notifications require an allowlisted sender and exact format', async t => {
  const {bot, sent} = fakeBot(t, {accept: ['Friend']});
  await bot.receive('|pm| Stranger| BaselineBot|/challenge gen9randombattle||||');
  await bot.receive('|pm| Friend| BaselineBot|/challenge gen9ou||||');
  assert.equal(sent.length, 0);
  await bot.receive('|pm|@Friend@!|@BaselineBot|/challenge gen9randombattle||||');
  await bot.receive('|pm| Friend| BaselineBot|/challenge gen9randombattle||||');
  assert.deepEqual(sent, ['|/utm null', '|/accept friend']);
  await bot.receive('|pm| BaselineBot| Friend|/challenge');
  await bot.receive('|updatechallenges|{"challengesFrom":{"friend":"gen9randombattle"}}');
  assert.equal(sent.length, 4);
});

test('remote transport requires encrypted authenticated connection; usernames cannot inject commands', () => {
  assert.throws(() => new ShowdownBot({url: 'ws://example.com/showdown/websocket'}), /WSS/);
  assert.throws(() => new ShowdownBot({url: 'wss://example.com/showdown/websocket'}), /PASSWORD/);
  assert.throws(() => new ShowdownBot({name: 'A\n/choose move 1'}), /name/);
});

test('authentication uses exact challenge, strips JSON prefix and does not echo credentials on failure', async () => {
  const password = 'secret-unprinted'; let body;
  const assertion = await loginAssertion('Bot', password, '4|abc', async (url, opts) => {
    assert.equal(url, 'https://play.pokemonshowdown.com/api/login'); body = opts.body;
    return {ok: true, text: async () => ']{"actionsuccess":true,"assertion":"signed"}'};
  });
  assert.equal(assertion, 'signed'); assert.equal(body.get('challstr'), '4|abc');
  await assert.rejects(loginAssertion('Bot', password, '4|abc', async () => ({ok: true,
    text: async () => `]{"actionsuccess":false,"error":"${password}"}`})), e => !e.message.includes(password));
});

test('explicit invalid choices and foreign battle formats fail visibly', async t => {
  const {bot} = fakeBot(t);
  await assert.rejects(bot.receive(frame(request()).replace('[Gen 9] Random Battle', '[Gen 9] OU')), /format/);
  await assert.rejects(bot.receive('>battle-gen9randombattle-1\n|error|[Invalid choice] bad choice'), /rejected/);
  const foreign = request(10); foreign.side.name = 'OtherPlayer';
  await assert.rejects(bot.receive(frame(foreign)), /another player/);
});

test('disconnect fails without replaying a pending choice or reconnecting', async t => {
  class Socket extends EventTarget {
    readyState = 1; sent = [];
    send(x) { this.sent.push(x); }
    close() {}
  }
  const socket = new Socket(); let constructions = 0;
  const bot = new ShowdownBot({socketFactory: () => { constructions++; return socket; }});
  t.after(() => bot.close());
  const ready = bot.connect();
  socket.dispatchEvent(new MessageEvent('message', {data: '|challstr|4|abc'}));
  await pause(5); assert.deepEqual(socket.sent, ['|/trn BaselineBot,0,']);
  socket.dispatchEvent(new MessageEvent('message', {data: '|updateuser| BaselineBot|1|1|{}'}));
  await ready;
  const failure = new Promise(resolve => bot.once('failure', resolve));
  socket.dispatchEvent(new Event('close'));
  assert.match((await failure).message, /Disconnected/); assert.equal(constructions, 1);
});
