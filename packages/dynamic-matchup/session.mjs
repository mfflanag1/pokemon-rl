import {readFileSync} from 'node:fs';
import Ajv from 'ajv/dist/2020.js';
import {FullLearningSession} from '../full-learning/session.mjs';
import {DynamicRosterView} from './view.mjs';
export {SPLITS, TACTICAL_SIDES} from '../full-learning/session.mjs';

const schema = JSON.parse(readFileSync(new URL('../../contracts/dynamic-matchup-v1/player.schema.json', import.meta.url)));
const validate = new Ajv({allErrors: true, strict: false}).compile(schema);

export class DynamicMatchupSession extends FullLearningSession {
  constructor() {
    super();
    this.views = Object.fromEntries(['p1', 'p2'].map(side => [side, new DynamicRosterView(side)]));
  }

  snapshot() {
    const frame = super.snapshot();
    for (const side of ['p1', 'p2']) {
      const player = this.views[side].withOwnTypes(frame.players[side]);
      if (!validate(player)) throw new Error(`Invalid dynamic player: ${JSON.stringify(validate.errors)}`);
      frame.players[side] = player;
    }
    return {...frame, schema_version: 'dynamic-matchup-frame-v1'};
  }
}
