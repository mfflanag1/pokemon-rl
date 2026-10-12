import {RecoveringBot} from '../showdown-session/client.mjs';
import {parseFrame} from '../showdown-bot/protocol.mjs';
import {createActor} from '../showdown-panel/policy.mjs';
import {TacticalView} from '../showdown-panel/view.mjs';

/** Versioned room-factory adapter for the sealed transport/session implementation.
 * Each new connection rebuilds tactical memory from its own received history. */
export class TacticalBot extends RecoveringBot {
  constructor({policy = 'tactical-v1', ...options} = {}) {
    if (policy !== 'tactical-v1') throw new Error('Adapter requires the frozen tactical-v1 policy');
    super({...options, policy: 'heuristic'});
    this.policyId = policy;
  }
  async attempt() {
    const previous = this.connection;
    // The sealed parent creates connection and installs queued message listeners
    // synchronously before its first await. Patch this instance before that queue runs.
    const pending = super.attempt(), connection = this.connection;
    if (connection && connection !== previous) {
      const receive = connection.receive.bind(connection);
      connection.receive = async frame => {
        if (connection.failed || connection.stopped) return;
        for (const {room: id, line} of parseFrame(frame)) {
          if (id.startsWith('battle-') && line === '|init|battle' && !connection.rooms.has(id)) {
            if (connection.rooms.size >= 64) throw new Error('Session room budget exceeded');
            const actor = createActor('tactical-v1');
            connection.rooms.set(id, {state: actor.state, choose: actor.choose, request: null, lastRqid: -1,
              sent: new Set(), decisions: 0, retries: 0, timer: null, done: false});
          }
        }
        await receive(frame);
        if ([...connection.rooms.values()].some(room => !(room.state instanceof TacticalView))) {
          throw new Error('Tactical room factory invariant failed');
        }
      };
    }
    return pending;
  }
}
