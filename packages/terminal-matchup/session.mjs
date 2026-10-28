import {readFileSync} from 'node:fs';
import Ajv from 'ajv/dist/2020.js';
import {FullLearningSession} from '../full-learning/session.mjs';
import {TerminalRosterView} from './view.mjs';
import {choiceFor} from '../full-showdown/actions.mjs';
export {SPLITS, TACTICAL_SIDES} from '../full-learning/session.mjs';

const schema = JSON.parse(readFileSync(new URL('../../contracts/terminal-matchup-v1/player.schema.json', import.meta.url)));
const validate = new Ajv({allErrors: true, strict: false}).compile(schema);

export class TerminalMatchupSession extends FullLearningSession {
  constructor() {
    super();
    this.views = Object.fromEntries(['p1', 'p2'].map(side => [side, new TerminalRosterView(side)]));
  }

  drain() {
    for (const side of ['p1','p2']) {
      const lines = this.projection.players[side].protocol;
      while (this.cursors[side] < lines.length) {
        const line = lines[this.cursors[side]++];
        if (line.startsWith('|request|')) this.views[side].acceptRequest(JSON.parse(line.slice(9)));
        else this.views[side].ingest(line);
      }
    }
    // The inherited frozen tactical advisor retains its own independent cursor.
    super.drain();
  }

  async step(input) {
    if (this.outcome) throw new Error('Cannot step a completed session');
    const sides = ['p1','p2'], pending = sides.filter(side => this.projection.players[side].pending);
    if (!input?.actions || !input.request_ids || Object.keys(input.actions).sort().join() !== pending.join() ||
        Object.keys(input.request_ids).sort().join() !== pending.join()) throw new Error('Submit exactly the currently pending sides');
    const hints = Object.fromEntries(sides.map(side => [side,null]));
    for (const side of pending) {
      const p = this.projection.players[side];
      if (input.request_ids[side] !== p.request_id) throw new Error('Stale player request');
      const match = /^switch (\d+)$/.exec(choiceFor(input.actions[side],p.request));
      // Revival selects a bench target but does not change the active identity.
      if (match && !p.request.side.pokemon.some(mon => mon.reviving)) {
        hints[side] = p.request.side.pokemon[Number(match[1])-1].ident.split(': ').slice(1).join(': ');
      }
    }
    for (const side of sides) this.views[side].expectSwitch(hints[side]);
    try { return await super.step(input); }
    finally { for (const side of sides) this.views[side].expectSwitch(null); }
  }

  snapshot() {
    const frame = super.snapshot();
    for (const side of ['p1', 'p2']) {
      const player = this.views[side].withOwnTypes(frame.players[side]);
      if (!validate(player)) throw new Error(`Invalid dynamic player: ${JSON.stringify(validate.errors)}`);
      frame.players[side] = player;
    }
    return {...frame, schema_version: 'terminal-matchup-frame-v1'};
  }
}
