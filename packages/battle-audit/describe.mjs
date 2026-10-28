import {Dex} from '../reference-node/engine.mjs';
import {legalChoices, toID} from '../showdown-bot/protocol.mjs';
import {ACTIONS, requestActions} from '../full-showdown/actions.mjs';
import {chooseOpponent, damageProxy} from '../showdown-panel/policy.mjs';

/** One-player decision describer: diagnostics only. It reads one seat's own raw
 * request, that seat's TacticalView built from projected public events, and that
 * seat's full-showdown-player-v1 projection. It never receives a simulator, the
 * opponent's request, or hidden sets, and it never feeds an actor. */
export const SCHEMA_VERSION = 'battle-audit-describe-v1';
export const FAMILIES = Object.freeze(['damage', 'pivot', 'status', 'setup', 'recovery',
  'hazard', 'hazard_clear', 'protect', 'switch', 'control']);
export const TAGS = Object.freeze(['damage_choice', 'forced_replacement', 'voluntary_switch',
  'tera_timing', 'status_setup_recovery', 'hazards', 'hidden_information']);
/** Minimum best-minus-selected proxy difference that earns a damage_choice tag. */
export const DAMAGE_CHOICE_GAP = 0.2;

const HAZARDS = Object.freeze(['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']);
const CONTROL = new Set(['recharge', 'struggle']);
const CLEARERS = new Set(['rapidspin', 'mortalspin', 'defog', 'tidyup', 'courtchange']);
const HAZARD_ATTACKS = new Set(['stoneaxe', 'ceaselessedge']);
const PROTECTS = new Set(['protect', 'detect', 'kingsshield', 'spikyshield', 'banefulbunker', 'obstruct',
  'silktrap', 'burningbulwark', 'endure', 'wideguard', 'quickguard', 'matblock', 'craftyshield']);
const RECOVERY_EXTRA = new Set(['rest', 'wish', 'painsplit', 'aquaring', 'ingrain', 'strengthsap',
  'junglehealing', 'lifedew', 'shoreup', 'healingwish', 'lunardance']);
const TARGETED_HEALS = new Set(['healpulse', 'floralhealing']);
const SETUP_EXTRA = new Set(['curse', 'bellydrum', 'filletaway', 'clangoroussoul', 'noretreat', 'shellsmash',
  'geomancy', 'stockpile', 'substitute', 'focusenergy', 'charge', 'magnetrise', 'powertrick', 'minimize',
  'defensecurl', 'laserfocus', 'trickroom', 'tailwind']);
const NON_SETUP_SELF = new Set(['destinybond', 'grudge', 'splash', 'celebrate', 'holdhands', 'happyhour']);
const SELF_TARGETS = new Set(['self', 'adjacentAllyOrSelf', 'allies', 'allySide', 'allyTeam']);
const ENGINE_KEYS = ['sides', 'field', 'prng', 'queue', 'log', 'inputLog', 'makeChoices', 'getRequests',
  'commitChoices', 'commitDecisions', 'choose', 'send', 'battle', 'sim', 'stream'];
const PRIVATE_FOE_KEYS = ['stats', 'baseStoredStats', 'storedStats', 'set', 'moveSlots', 'evs', 'ivs', 'nature', 'hpType'];

const fail = message => { throw new Error(message); };
const num = x => Number.isFinite(x) ? Math.round(x * 1e6) / 1e6 : 0;
const isObject = v => v !== null && typeof v === 'object' && !Array.isArray(v);

/** Semantic family of one advertised request move id, from public Dex metadata only. */
export function classifyMove(rawId) {
  const id = toID(rawId ?? '');
  if (CONTROL.has(id)) return 'control';
  const move = Dex.moves.get(id);
  if (!move.exists) fail(`Unknown Gen9 request move: ${id}`);
  if (move.selfSwitch) return 'pivot';
  if (move.category !== 'Status') {
    if (CLEARERS.has(id)) return 'hazard_clear';
    if (HAZARD_ATTACKS.has(id) || HAZARDS.includes(move.sideCondition)) return 'hazard';
    return 'damage';
  }
  if (HAZARDS.includes(move.sideCondition)) return 'hazard';
  if (CLEARERS.has(id)) return 'hazard_clear';
  if (move.stallingMove || PROTECTS.has(id)) return 'protect';
  if (RECOVERY_EXTRA.has(id) || ((move.heal || move.flags?.heal) && !TARGETED_HEALS.has(id))) return 'recovery';
  const selfBoost = Boolean(move.boosts) && SELF_TARGETS.has(move.target) && Object.values(move.boosts).some(v => v > 0);
  if (selfBoost || move.weather || move.terrain || move.pseudoWeather || move.sideCondition || SETUP_EXTRA.has(id) ||
    (move.target === 'self' && move.volatileStatus && !NON_SETUP_SELF.has(id))) return 'setup';
  return 'status';
}

/** True when the advertised move contributes to damage proxies (control moves never do). */
export function isDamagingMove(rawId) {
  const id = toID(rawId ?? '');
  if (CONTROL.has(id)) return false;
  const move = Dex.moves.get(id);
  return move.exists && move.category !== 'Status';
}

// Verbatim copy of the frozen policy's private ownMon so proxies match its inputs.
function ownMon(mon, view, seat, active = false) {
  return {...(active ? view.active[seat] : {}), ...mon, species: mon.details.split(',')[0],
    boosts: active ? view.active[seat]?.boosts ?? {} : {},
    enteredTurn: active ? view.active[seat]?.enteredTurn : view.turn};
}

function rejectEngine(value, label) {
  if (!isObject(value)) fail(`${label} must be a plain data object`);
  for (const key of Object.keys(value)) {
    if (typeof value[key] === 'function') fail(`${label} carries executable state (${key}); engine objects are not accepted`);
    if (ENGINE_KEYS.includes(key)) fail(`${label} looks like a simulator object (${key}); engine objects are not accepted`);
  }
}

function validate(request, view, player, action) {
  rejectEngine(request, 'request'); rejectEngine(view, 'view'); rejectEngine(player, 'player');
  if (!Number.isInteger(action) || action < 0 || action >= ACTIONS.length) fail('Action must be an integer in 0..13');
  if (player.schema_version !== 'full-showdown-player-v1') fail('Expected a full-showdown-player-v1 projection');
  if (!['p1', 'p2'].includes(player.side)) fail('Projection has no valid side');
  if (!isObject(request.side) || request.side.id !== player.side) fail('Request seat does not match projection side');
  if (request.teamPreview || request.wait) fail('Only move and replacement requests are described');
  if (!Array.isArray(player.mask) || player.mask.length !== ACTIONS.length || player.mask.some(b => typeof b !== 'boolean')) {
    fail('Projection mask must be fourteen booleans');
  }
  if (player.mask[action] !== true) fail(`Action ${action} (${ACTIONS[action]}) is not allowed by the projection mask`);
  if (!requestActions(request).mask[action]) fail(`Action ${action} (${ACTIONS[action]}) is illegal for this request`);
  if (!isObject(view.active) || !isObject(view.hazards) || !Number.isFinite(view.turn)) fail('Expected a TacticalView snapshot');
  const seat = request.side.id, foeSeat = seat === 'p1' ? 'p2' : 'p1';
  if (!isObject(view.active[seat]) || !isObject(view.active[foeSeat])) fail('TacticalView lacks an active Pokémon for both seats');
  for (const key of PRIVATE_FOE_KEYS) if (key in view.active[foeSeat]) fail(`Private opponent data (${key}) in public view`);
  if (!isObject(view.hazards[seat])) fail('TacticalView lacks own hazard state');
  const phase = request.forceSwitch ? (request.side.pokemon.some(p => p.reviving) ? 'revival' : 'replace') : 'move';
  if (player.phase !== undefined && player.phase !== phase) fail('Projection phase disagrees with request');
  return {seat, foeSeat, phase};
}

function describeChoice(choice, request) {
  if (!choice || !ACTIONS.includes(choice)) return {action: null, choice: choice ?? null, family: null, move: null, tera: false, damaging: false};
  const action = ACTIONS.indexOf(choice);
  if (choice.startsWith('switch ')) return {action, choice, family: 'switch', move: null, tera: false, damaging: false};
  const slot = Number(choice.split(' ')[1]) - 1, advertised = request.active[0].moves[slot];
  const id = toID(advertised.id ?? advertised.move);
  return {action, choice, family: classifyMove(id), move: id, tera: choice.endsWith(' terastallize'), damaging: isDamagingMove(id)};
}

export function describeDecision(rawRequest, rawView, rawPlayer, action) {
  const {seat, foeSeat, phase} = validate(rawRequest, rawView, rawPlayer, action);
  // Defensive copies: the observer never mutates caller state, nor lets the policy do so.
  const request = structuredClone(rawRequest), view = structuredClone(rawView), player = structuredClone(rawPlayer);
  const foe = view.active[foeSeat];
  const selected = describeChoice(ACTIONS[action], request);
  const recommendedChoice = chooseOpponent('tactical-v1', request, view);
  const recommended = describeChoice(recommendedChoice, request);
  if (recommended.action !== null && !player.mask[recommended.action]) fail('Policy recommendation escaped the projection mask');

  // Damage proxies over legal move actions only (Tera variants included); no hidden stats.
  const proxies = [];
  if (phase === 'move') {
    const own = ownMon(request.side.pokemon.find(p => p.active), view, seat, true), active = request.active[0];
    for (const choice of legalChoices(request).filter(c => c.startsWith('move '))) {
      const slot = Number(choice.split(' ')[1]) - 1, advertised = active.moves[slot];
      const id = toID(advertised.id ?? advertised.move);
      if (!isDamagingMove(id)) continue;
      const tera = choice.endsWith(' terastallize') ? active.canTerastallize : null;
      proxies.push([choice, num(damageProxy(Dex.moves.get(id).id, own, foe, view, tera, advertised))]);
    }
  }
  const best = proxies.reduce((top, entry) => entry[1] > (top?.[1] ?? -1) ? entry : top, null);
  const bestProxy = best ? best[1] : 0;
  const selectedProxy = selected.damaging ? proxies.find(([choice]) => choice === selected.choice)?.[1] ?? 0 : null;
  const gap = num(Math.max(0, bestProxy - (selectedProxy ?? 0)));

  const ownHazards = HAZARDS.filter(id => (view.hazards[seat]?.[id] ?? 0) > 0 || (player.side_conditions?.own?.[id] ?? 0) > 0);
  const revealedDamaging = (foe.moves ?? []).filter(id => isDamagingMove(id));
  const families = [selected.family, recommended.family];
  const switching = families.includes('switch');
  const tags = [];
  if (selected.damaging && bestProxy - selectedProxy >= DAMAGE_CHOICE_GAP - 1e-9) tags.push('damage_choice');
  if (phase !== 'move' && selected.family === 'switch') tags.push('forced_replacement');
  if (phase === 'move' && switching) tags.push('voluntary_switch');
  if (selected.tera !== recommended.tera) tags.push('tera_timing');
  if (families.some(f => ['status', 'setup', 'recovery'].includes(f))) tags.push('status_setup_recovery');
  if (families.some(f => ['hazard', 'hazard_clear'].includes(f)) || (ownHazards.length && switching)) tags.push('hazards');
  if (!revealedDamaging.length) tags.push('hidden_information');

  return {
    schema_version: SCHEMA_VERSION, side: seat, opponent: foeSeat, turn: view.turn, phase,
    selected_action: selected.action, selected_choice: selected.choice, selected_family: selected.family,
    selected_move: selected.move, selected_tera: selected.tera, selected_damaging: selected.damaging,
    recommended_action: recommended.action, recommended_choice: recommended.choice, recommended_family: recommended.family,
    recommended_move: recommended.move, recommended_tera: recommended.tera,
    disagreement: selected.action !== recommended.action,
    tera_available: typeof request.active?.[0]?.canTerastallize === 'string' && request.active[0].canTerastallize.length > 0,
    selected_damage_proxy: selectedProxy === null ? null : num(selectedProxy),
    best_legal_damage_proxy: num(bestProxy), best_damage_choice: best ? best[0] : null, damage_proxy_gap: gap,
    own_hazards: ownHazards, opponent_revealed_damaging: revealedDamaging,
    tags: TAGS.filter(tag => tags.includes(tag)),
  };
}
