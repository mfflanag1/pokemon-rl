import {EventEmitter} from 'node:events';
import {FORMAT, DEV_FORMAT, isLoopback, legalChoices, parseFrame, PublicBattle, toID} from './protocol.mjs';
import {makePolicy} from './policy.mjs';

const safeName = name => typeof name === 'string' && /^[A-Za-z0-9][A-Za-z0-9 ]{0,17}$/.test(name) && toID(name);
const identityID = value => toID(value.replace(/^[^A-Za-z0-9]+/, '').split('@')[0]);

export async function loginAssertion(name, password, challstr, fetcher = fetch) {
  const response = await fetcher('https://play.pokemonshowdown.com/api/login', {
    method: 'POST', headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: new URLSearchParams({name, pass: password, challstr}), signal: AbortSignal.timeout(15_000),
    redirect: 'error',
  });
  if (!response.ok) throw new Error(`Login HTTP ${response.status}`);
  const text = await response.text();
  let data;
  try { data = JSON.parse(text.startsWith(']') ? text.slice(1) : text); } catch { throw new Error('Malformed login response'); }
  if (!data.actionsuccess || typeof data.assertion !== 'string' || !data.assertion || /[\r\n]/.test(data.assertion)) {
    throw new Error('Login failed (check credentials or account requirements)');
  }
  return data.assertion;
}

export class ShowdownBot extends EventEmitter {
  constructor({url = 'ws://127.0.0.1:8000/showdown/websocket', name = 'BaselineBot', password,
    format = FORMAT, policy = 'heuristic', seed, accept = [], team = null, socketFactory = endpoint => new WebSocket(endpoint),
    authenticate = loginAssertion, decisionDelayMs = 750, timeoutMs = 30_000} = {}) {
    super();
    const endpoint = new URL(url);
    if (!['ws:', 'wss:'].includes(endpoint.protocol) || endpoint.username || endpoint.password || endpoint.search || endpoint.hash) throw new Error('Invalid WebSocket endpoint');
    if (!isLoopback(url) && endpoint.protocol !== 'wss:') throw new Error('Remote connections require WSS');
    if (!safeName(name) || accept.some(n => !safeName(n))) throw new Error('Invalid player name');
    if (![FORMAT, DEV_FORMAT].includes(format) || (format === DEV_FORMAT && !isLoopback(url))) throw new Error('Unsupported format');
    if (!isLoopback(url) && !password) throw new Error('Remote login requires SHOWDOWN_PASSWORD');
    if (!Number.isFinite(decisionDelayMs) || decisionDelayMs < 0 || !Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Invalid client timing');
    if (team !== null && (typeof team !== 'string' || /[\r\n]/.test(team))) throw new Error('Invalid packed team');
    this.options = {url, name, format, policy, seed, team, decisionDelayMs, timeoutMs};
    this.password = password; this.socketFactory = socketFactory; this.authenticate = authenticate;
    this.accept = new Set(accept.map(toID)); this.rooms = new Map(); this.accepting = new Set();
    this.queue = Promise.resolve(); this.failed = false; this.stopped = false; this.ready = false;
  }
  send(room, command) {
    if (this.failed || this.stopped || this.socket?.readyState !== 1) throw new Error('Connection is not active');
    if (/[\r\n]/.test(command) || (room && !/^battle-[a-z0-9-]+$/.test(room))) throw new Error('Unsafe protocol command');
    this.socket.send(`${room}|${command}`);
    // Do not emit credentials, login assertions, chat or raw private requests into diagnostic logs.
  }
  connect() {
    if (this.socket) throw new Error('Client already connected');
    return new Promise((resolve, reject) => {
      this.resolveReady = resolve; this.rejectReady = reject;
      this.loginTimer = setTimeout(() => this.fail(new Error('Login timed out')), this.options.timeoutMs);
      this.socket = this.socketFactory(this.options.url);
      this.socket.addEventListener('message', event => {
        this.lastMessageAt = Date.now();
        this.queue = this.queue.then(() => this.receive(event.data)).catch(error => this.fail(error));
      });
      this.socket.addEventListener('error', () => this.fail(new Error('WebSocket transport error')));
      this.socket.addEventListener('close', () => {
        if (!this.stopped && !this.failed) this.fail(new Error('Disconnected; no automatic replay of pending choices'));
      });
    });
  }
  fail(error) {
    if (this.failed || this.stopped) return;
    this.failed = true; clearTimeout(this.loginTimer); clearInterval(this.keepalive);
    for (const room of this.rooms.values()) clearTimeout(room.timer);
    this.rejectReady?.(error); this.socket?.close(); this.emit('failure', error);
  }
  close() {
    this.stopped = true; clearTimeout(this.loginTimer); clearInterval(this.keepalive);
    for (const room of this.rooms.values()) clearTimeout(room.timer);
    this.socket?.close();
  }
  challenge(name) {
    if (!this.ready || !safeName(name)) throw new Error('Not ready or invalid opponent');
    if ([...this.rooms.values()].some(r => !r.done)) throw new Error('Finish the current battle first');
    this.send('', `/utm ${this.options.team ?? 'null'}`);
    this.send('', `/challenge ${name}, ${this.options.format}`);
  }
  async receive(frame) {
    if (this.failed || this.stopped) return;
    const touched = new Set();
    for (const {room: id, line} of parseFrame(frame)) {
      if (!id) { await this.global(line); continue; }
      if (!id.startsWith('battle-')) continue;
      if (!this.rooms.has(id)) this.rooms.set(id, {state: new PublicBattle(), request: null, lastRqid: -1,
        sent: new Set(), decisions: 0, retries: 0, timer: null, done: false, choose: makePolicy(this.options.policy, this.options.seed)});
      const room = this.rooms.get(id);
      room.state.ingest(line); touched.add(id);
      if (line.startsWith('|request|')) {
        const request = JSON.parse(line.slice(9));
        if (request === null) { room.request = null; clearTimeout(room.timer); continue; }
        if (!Number.isSafeInteger(request.rqid) || request.rqid < 0) throw new Error('Missing/invalid server request ID');
        if (request.rqid < room.lastRqid) continue;
        if (!request.wait && toID(request.side?.name) !== toID(this.options.name)) throw new Error('Private request belongs to another player');
        if (request.side && room.seat && request.side.id !== room.seat) throw new Error('Player seat changed within a battle');
        if (request.side) room.seat = request.side.id;
        room.lastRqid = request.rqid; room.request = request;
        clearTimeout(room.timer); room.timer = null;
      }
      if (line.startsWith('|sentchoice|') && room.request) room.sent.add(JSON.stringify(room.request));
      if (line === '|deinit') { room.request = null; room.done = true; clearTimeout(room.timer); }
      if (line.startsWith('|bigerror|')) throw new Error(`Simulator reported a battle error in ${id}`);
      if (line.startsWith('|error|')) {
        if (line.startsWith('|error|[Unavailable choice]')) {
          room.request = null; clearTimeout(room.timer); room.retries++;
          this.emit('retry', {room: id, reason: 'unavailable-choice'});
          // The simulator sends a refreshed request after this error. Never guess a retry from stale information.
        } else throw new Error(`Battle ${id} rejected a choice: ${line.slice(7, 250)}`);
      }
    }
    for (const id of touched) {
      const room = this.rooms.get(id);
      if (room.state.ended) {
        clearTimeout(room.timer);
        if (!room.done) {
          room.done = true;
          this.emit('battleEnd', {room: id, winner: room.state.winner, turns: room.state.turn, decisions: room.decisions, retries: room.retries});
          this.send(id, '/leave');
        }
        continue;
      }
      if (room.done || !room.request || room.request.wait || room.timer || !this.ready) continue;
      if (room.state.gameType !== 'singles' || room.state.tier !== this.options.format) throw new Error('Unexpected battle format or game type');
      const key = JSON.stringify(room.request);
      if (room.sent.has(key)) continue;
      room.timer = setTimeout(() => {
        room.timer = null;
        try {
          if (!room.request || room.state.ended || this.failed || this.stopped || JSON.stringify(room.request) !== key) return;
          const choice = room.choose(structuredClone(room.request), structuredClone(room.state));
          if (!legalChoices(room.request).includes(choice)) throw new Error('Policy returned an inadmissible choice');
          if (++room.decisions > 20_000) throw new Error('Battle decision budget exceeded');
          room.sent.add(key);
          this.send(id, `/choose ${choice}|${room.request.rqid}`);
          this.emit('decision', {room: id, rqid: room.request.rqid, turn: room.state.turn, choice});
        } catch (error) { this.fail(error); }
      }, this.options.decisionDelayMs);
    }
  }
  async global(line) {
    const [, type, value, named] = line.split('|');
    if (type === 'challstr') {
      if (this.authStarted) throw new Error('Repeated authentication challenge');
      this.authStarted = true;
      const assertion = this.password ? await this.authenticate(this.options.name, this.password, line.slice(10)) : '';
      this.password = undefined;
      this.send('', `/trn ${this.options.name},0,${assertion}`);
    } else if (type === 'updateuser' && named === '1' && identityID(value) === toID(this.options.name)) {
      if (!this.ready) {
        this.ready = true; clearTimeout(this.loginTimer); this.lastMessageAt = Date.now();
        this.keepalive = setInterval(() => {
          try {
            if (Date.now() - this.lastMessageAt > 60_000) throw new Error('Server stopped responding');
            this.send('', `/query userdetails ${toID(this.options.name)}`);
          } catch (error) { this.fail(error); }
        }, 30_000);
        this.keepalive.unref(); this.resolveReady?.(this); this.emit('ready');
      }
    } else if (['nametaken', 'popup'].includes(type)) {
      throw new Error(type === 'nametaken' ? 'Requested username unavailable' : 'Server rejected the operation (popup)');
    } else if (type === 'updatechallenges' && this.ready) {
      const {challengesFrom = {}} = JSON.parse(line.slice(18));
      for (const name of this.accepting) if (!(name in challengesFrom)) this.accepting.delete(name);
      for (const [name, format] of Object.entries(challengesFrom)) {
        this.acceptChallenge(name, format);
      }
    } else if (type === 'pm' && this.ready) {
      const [, , from, to, message] = line.split('|');
      const fromID = identityID(from), toIDValue = identityID(to), self = toID(this.options.name);
      // Challenge removal is broadcast to both parties; its from/to direction may be reversed on acceptance.
      if (message === '/challenge' && (fromID === self || toIDValue === self)) {
        this.accepting.delete(fromID === self ? toIDValue : fromID);
      } else if (toIDValue === self && message?.startsWith('/challenge ')) this.acceptChallenge(fromID, message.slice(11));
    }
  }
  acceptChallenge(name, format) {
    name = toID(name);
    if (this.accept.has(name) && format === this.options.format && !this.accepting.has(name) &&
        ![...this.rooms.values()].some(r => !r.done)) {
      this.accepting.add(name); this.send('', `/utm ${this.options.team ?? 'null'}`); this.send('', `/accept ${name}`);
    }
  }
}
