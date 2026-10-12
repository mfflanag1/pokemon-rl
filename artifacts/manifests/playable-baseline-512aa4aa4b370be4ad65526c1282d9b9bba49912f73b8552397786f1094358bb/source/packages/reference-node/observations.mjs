import {extractChannelMessages} from './engine.mjs';

// Only protocol timestamp lines are normalized. Requests and mechanics are unchanged.
export const normalizeProtocol = text => text.split('\n').map(line => /^\|t:\|[0-9]+$/.test(line) ? '|t:|<timestamp>' : line).join('\n');

// R1 protocol choices, not the future semantic/tensor action codec. No hidden state.
export function admissibleChoices(request) {
  if (!request || request.wait) return [];
  const pokemon = request.side.pokemon;
  const fainted = mon => mon.condition.endsWith(' fnt');
  if (request.forceSwitch) {
    if (request.forceSwitch.length !== 1) throw new Error('Only singles requests are supported');
    if (!request.forceSwitch[0]) return ['pass'];
    const reviving = Boolean(pokemon[0].reviving);
    const choices = pokemon.flatMap((mon, index) => (
      (index > 0 || reviving) && (reviving ? fainted(mon) : !fainted(mon)) ? [`switch ${index + 1}`] : []
    ));
    return choices.length ? choices : ['pass'];
  }
  if (request.teamPreview || request.active?.length !== 1) throw new Error('Unexpected request phase for the pinned singles format');
  const active = request.active[0];
  const choices = active.moves.flatMap((move, index) => move.disabled ? [] : [
    `move ${index + 1}`, ...(active.canTerastallize ? [`move ${index + 1} terastallize`] : []),
  ]);
  if (!active.trapped) pokemon.forEach((mon, index) => {
    if (!mon.active && !fainted(mon)) choices.push(`switch ${index + 1}`);
  });
  return choices;
}

export class PlayerProjection {
  constructor() {
    this.players = Object.fromEntries(['p1', 'p2'].map(side => [side, {
      side, request_id: 0, turn: 0, request: null, protocol: [], pending: false,
    }]));
    this.spectator = [];
    this.errors = [];
  }

  ingest(type, data) {
    if (type === 'update') {
      // The same splitter used by upstream getPlayerStreams; never give actors channel -1.
      const channels = extractChannelMessages(data, [0, 1, 2]);
      this.spectator.push(...normalizeProtocol(channels[0].join('\n')).split('\n'));
      for (const [side, channel] of [['p1', 1], ['p2', 2]]) this.receive(side, channels[channel].join('\n'));
    } else if (type === 'sideupdate') {
      const newline = data.indexOf('\n');
      const side = data.slice(0, newline);
      if (!this.players[side]) throw new Error(`Unexpected private channel ${side}`);
      this.receive(side, data.slice(newline + 1));
    } else if (type !== 'end') {
      throw new Error(`Unknown simulator message ${type}`);
    }
    // The end payload contains both private teams and never enters a PlayerView.
  }

  receive(side, text) {
    const player = this.players[side];
    for (const line of normalizeProtocol(text).split('\n')) {
      player.protocol.push(line);
      if (line.startsWith('|turn|')) player.turn = Number(line.slice(6));
      if (line.startsWith('|request|')) {
        player.request = JSON.parse(line.slice(9));
        player.request_id++;
        player.pending = Boolean(player.request && !player.request.wait);
      }
      if (line.startsWith('|error|')) this.errors.push({side, message: line.slice(7)});
    }
  }

  view(side) {
    const {pending, ...view} = this.players[side];
    return structuredClone(view);
  }
}
