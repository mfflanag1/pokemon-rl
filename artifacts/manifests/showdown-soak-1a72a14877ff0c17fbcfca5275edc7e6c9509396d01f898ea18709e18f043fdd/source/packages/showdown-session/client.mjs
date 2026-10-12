import {EventEmitter} from 'node:events';
import {ShowdownBot} from '../showdown-bot/client.mjs';
import {parseFrame} from '../showdown-bot/protocol.mjs';

// These are the sealed v1 client's transport failures. Protocol/authentication
// errors must remain fatal rather than repeatedly logging in or submitting moves.
const TRANSPORT = new Set(['WebSocket transport error',
  'Disconnected; no automatic replay of pending choices', 'Server stopped responding', 'Login timed out',
  'WebSocket transport not open']);

class SessionConnection extends ShowdownBot {
  constructor(options, session, expected) {
    super(options);
    this.session = session;
    this.expected = expected;
    this.initialized = new Set();
  }
  acceptChallenge(name, format) {
    if (!this.expected.size) super.acceptChallenge(name, format);
  }
  async receive(frame) {
    if (this.stopped || this.failed) return;
    const lines = parseFrame(frame), touched = new Set();
    for (const {room: id, line} of lines) {
      if (!id.startsWith('battle-')) continue;
      if (line.startsWith('|noinit|') || line === '|deinit' && this.expected.has(id)) {
        throw new Error('Battle recovery unavailable');
      }
      if (line === '|init|battle') {
        if (this.initialized.has(id)) throw new Error('Repeated battle initialization');
        if (this.initialized.size >= 64) throw new Error('Session room budget exceeded');
        this.initialized.add(id);
      }
      if (!this.initialized.has(id)) throw new Error('Battle message preceded history initialization');
      if (line.startsWith('|request|') && line !== '|request|null') {
        const request = JSON.parse(line.slice(9)), before = this.expected.get(id);
        if (before && (request.rqid < before.rqid || request.side && before.seat && request.side.id !== before.seat)) {
          throw new Error('Recovered request moved backwards or changed seat');
        }
        touched.add(id);
      }
      if (line.startsWith('|win|') || line === '|tie') touched.add(id);
    }
    await super.receive(frame);
    if (this.stopped || this.failed) return;
    for (const id of touched) {
      const before = this.expected.get(id), room = this.rooms.get(id);
      if (!before || !room) continue;
      if (room.state.turn < before.turn) throw new Error('Recovered public history moved backwards');
      this.expected.delete(id);
      this.session.emit('restored', {room: id, rqid: room.lastRqid, turn: room.state.turn,
        committed: Boolean(room.request && room.sent.has(JSON.stringify(room.request))), ended: room.done});
    }
    this.session.maybeReady(this);
  }
  send(room, command) {
    // Classify a close handshake before v1's generic send guard can turn the
    // transport outage into a permanent protocol error. No attempted action is counted.
    if (!this.failed && !this.stopped && this.socket?.readyState !== 1) throw new Error('WebSocket transport not open');
    if (command.startsWith('/choose ')) {
      const total = this.session.decisions.get(room) ?? 0;
      if (total >= 20_000) throw new Error('Session battle decision budget exceeded');
    }
    super.send(room, command);
    if (command.startsWith('/choose ')) this.session.decisions.set(room, (this.session.decisions.get(room) ?? 0) + 1);
  }
}

/** In-memory recovery of the deterministic heuristic. A new socket rebuilds each
 * active battle from the server's player-visible history and current request.
 * Never copy a stale request, action, sent set or reducer into a new connection. */
export class RecoveringBot extends EventEmitter {
  constructor({password, passwordProvider = () => password, maxReconnects = 3,
    retryDelayMs = 1_000, recoveryTimeoutMs = 30_000, ...options} = {}) {
    super();
    if (options.policy && options.policy !== 'heuristic') throw new Error('Recovery requires the stateless heuristic policy');
    if (!Number.isSafeInteger(maxReconnects) || maxReconnects < 0 || maxReconnects > 10 ||
        !Number.isFinite(retryDelayMs) || retryDelayMs < 0 || retryDelayMs > 30_000 ||
        !Number.isFinite(recoveryTimeoutMs) || recoveryTimeoutMs <= 0 || recoveryTimeoutMs > 120_000) {
      throw new Error('Invalid recovery budget');
    }
    if (typeof passwordProvider !== 'function') throw new Error('Invalid password provider');
    this.options = {...options, policy: 'heuristic'};
    this.passwordProvider = passwordProvider;
    this.maxReconnects = maxReconnects; this.retryDelayMs = retryDelayMs; this.recoveryTimeoutMs = recoveryTimeoutMs;
    this.reconnects = 0; this.generation = 0; this.expected = new Map(); this.decisions = new Map();
    this.retries = new Map(); this.finished = new Set(); this.ready = false; this.stopped = false; this.failed = false;
  }
  connect() {
    if (this.started || this.stopped) throw new Error('Session already started or closed');
    this.started = true;
    const promise = new Promise((resolve, reject) => { this.resolveReady = resolve; this.rejectReady = reject; });
    void this.attempt();
    return promise;
  }
  async attempt() {
    const generation = ++this.generation;
    try {
      let password;
      try { password = this.passwordProvider(); } catch { throw new Error('Credential provider failed'); }
      // The CLI reads its local environment synchronously; deferred credential requests are unsupported.
      if (password !== undefined && typeof password !== 'string') throw new Error('Credential provider must return a string or undefined');
      if (this.stopped || this.failed) return;
      const child = new SessionConnection({...this.options, password}, this, this.expected);
      this.connection = child;
      child.on('failure', error => this.connectionFailed(child, generation, error));
      child.on('ready', () => {
        if (!this.current(child, generation)) return;
        try {
          for (const id of this.expected.keys()) child.send('', `/join ${id}`);
          if (this.expected.size) this.recoveryTimer = setTimeout(() => this.fail(new Error('Battle recovery timed out')), this.recoveryTimeoutMs);
          this.maybeReady(child);
        } catch (error) { child.fail(error); }
      });
      child.on('decision', event => this.emit('decision', {...event, generation}));
      child.on('retry', event => {
        this.retries.set(event.room, (this.retries.get(event.room) ?? 0) + 1); this.emit('retry', event);
      });
      child.on('battleEnd', event => {
        if (this.finished.has(event.room)) return;
        this.finished.add(event.room);
        this.emit('battleEnd', {...event, decisions: this.decisions.get(event.room) ?? 0, retries: this.retries.get(event.room) ?? 0});
      });
      await child.connect();
    } catch (error) {
      if (!this.stopped && !this.failed && generation === this.generation) this.fail(error);
    }
  }
  current(child, generation = this.generation) {
    return !this.stopped && !this.failed && child === this.connection && generation === this.generation;
  }
  maybeReady(child) {
    if (!this.current(child) || child.failed || !child.ready || this.expected.size || this.ready) return;
    clearTimeout(this.recoveryTimer); this.ready = true;
    this.resolveReady?.(this); this.emit('ready', {generation: this.generation, reconnects: this.reconnects});
  }
  connectionFailed(child, generation, error) {
    if (!this.current(child, generation)) return;
    this.ready = false; clearTimeout(this.recoveryTimer);
    if (!TRANSPORT.has(error.message) || this.reconnects >= this.maxReconnects) {
      this.fail(new Error(TRANSPORT.has(error.message) ? 'Reconnect budget exhausted' : error.message)); return;
    }
    for (const [id, room] of child.rooms) {
      if (!room.done) this.expected.set(id, {rqid: Math.max(room.lastRqid, this.expected.get(id)?.rqid ?? -1),
        turn: Math.max(room.state.turn, this.expected.get(id)?.turn ?? 0), seat: room.seat ?? this.expected.get(id)?.seat});
    }
    child.close();
    // Invalidate an in-flight login promise or queued event from this socket.
    this.generation++;
    const delayMs = Math.min(30_000, this.retryDelayMs * 2 ** this.reconnects++);
    this.emit('reconnecting', {attempt: this.reconnects, delayMs, rooms: [...this.expected.keys()], reason: error.message});
    if (!this.stopped && !this.failed) this.retryTimer = setTimeout(() => void this.attempt(), delayMs);
  }
  challenge(name) {
    if (!this.ready || this.failed || this.stopped) throw new Error('Session is not ready');
    this.connection.challenge(name);
  }
  fail(error) {
    if (this.failed || this.stopped) return;
    this.failed = true; this.ready = false;
    clearTimeout(this.retryTimer); clearTimeout(this.recoveryTimer);
    this.connection?.close(); this.passwordProvider = null;
    this.rejectReady?.(error); this.emit('failure', error);
  }
  close() {
    if (this.stopped) return;
    this.stopped = true; this.ready = false;
    clearTimeout(this.retryTimer); clearTimeout(this.recoveryTimer);
    this.connection?.close(); this.passwordProvider = null;
    this.rejectReady?.(new Error('Session closed'));
  }
}
