import {readFileSync} from 'node:fs';
import Ajv2020 from 'ajv/dist/2020.js';
import {canonical, parseJSON} from './canonical.mjs';

export const contractURL = new URL('../../contracts/battle-v1/', import.meta.url);
export const schema = parseJSON(readFileSync(new URL('schema.json', contractURL)));
const ajv = new Ajv2020({strict: true, allErrors: false});
ajv.addSchema(schema);
const validators = new Map();

export function validateSchema(value, fragment = null) {
  canonical(value); // Also forbids unsafe numbers inside otherwise open provenance fields.
  const key = fragment ?? '';
  let validator = validators.get(key);
  if (!validator) {
    if (fragment && !Object.hasOwn(schema.$defs, fragment)) throw new Error('Unknown schema fragment');
    validator = ajv.compile({$ref: `${schema.$id}${fragment ? `#/$defs/${fragment}` : ''}`});
    validators.set(key, validator);
  }
  if (!validator(value)) throw new TypeError(`Contract schema: ${ajv.errorsText(validator.errors)}`);
  return value;
}
