import {readFileSync} from 'node:fs';
import Ajv from 'ajv/dist/2020.js';

const schema = JSON.parse(readFileSync(new URL('../../contracts/full-showdown-v1/player.schema.json', import.meta.url)));
const validate = new Ajv({allErrors: true, strict: false}).compile(schema);
export function validatePlayer(player) {
  if (!validate(player)) throw new Error(`Invalid full-Showdown projection: ${JSON.stringify(validate.errors)}`);
  return player;
}
