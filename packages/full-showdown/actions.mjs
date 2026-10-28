import {legalChoices} from '../showdown-bot/protocol.mjs';

// Separate from every sealed action contract. Wait/pass is coordinator control,
// never a sampled policy action. Request slots are one-based in protocol strings.
export const ACTIONS = Object.freeze([
  ...Array.from({length: 4}, (_, i) => `move ${i + 1}`),
  ...Array.from({length: 4}, (_, i) => `move ${i + 1} terastallize`),
  ...Array.from({length: 6}, (_, i) => `switch ${i + 1}`),
]);

export function requestActions(request) {
  if (request?.teamPreview) throw new Error('Unexpected preview in full Gen9 Random Battle');
  const choices = legalChoices(request);
  if (choices.some(c => c !== 'pass' && !ACTIONS.includes(c))) throw new Error('Unsupported request choice');
  return {mask: ACTIONS.map(c => choices.includes(c)), pass: choices.length === 1 && choices[0] === 'pass'};
}

export function choiceFor(index, request) {
  if (!Number.isInteger(index) || index < 0 || index >= ACTIONS.length || !requestActions(request).mask[index]) {
    throw new Error('Illegal full-Showdown action index');
  }
  return ACTIONS[index];
}
