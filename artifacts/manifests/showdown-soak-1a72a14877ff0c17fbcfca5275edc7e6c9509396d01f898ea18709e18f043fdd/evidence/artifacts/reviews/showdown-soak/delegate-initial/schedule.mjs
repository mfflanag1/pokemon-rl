// Pure, deterministic schedule helper for the frozen showdown network-soak plan.
//
// Node builtins only. No simulator, no network, no I/O beyond reading the
// sibling plan.json once at module load. Every exported function returns fresh
// objects; the loaded plan is deep-frozen so it can never be mutated.

import { readFileSync } from 'node:fs';

const PLAN = deepFreeze(
  JSON.parse(readFileSync(new URL('./plan.json', import.meta.url), 'utf8')),
);

const NORMAL_SCENARIO = 'normal';
const SEAT_P1 = 'p1';
const SEAT_P2 = 'p2';

function deepFreeze(value) {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const key of Object.keys(value)) deepFreeze(value[key]);
  }
  return value;
}

function assertBoolean(smoke, name) {
  if (typeof smoke !== 'boolean') {
    throw new TypeError(`${name} must be a boolean, received ${describe(smoke)}`);
  }
}

function describe(value) {
  if (value === null) return 'null';
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number' || typeof value === 'bigint') return String(value);
  return typeof value;
}

/**
 * Returns a fresh deep copy of the top-level plan with smoke overrides
 * applied (when `smoke` is true), the `smoke` sub-object removed, and a
 * derived `games = sessions * games_per_session` field.
 *
 * @param {boolean} [smoke=false]
 */
export function planFor(smoke = false) {
  assertBoolean(smoke, 'smoke');
  const { smoke: smokeOverrides, ...base } = PLAN;
  const merged = smoke ? { ...base, ...smokeOverrides } : base;
  const plan = structuredClone(merged);
  plan.games = plan.sessions * plan.games_per_session;
  return plan;
}

/**
 * Returns the fully-resolved spec for a single scheduled game.
 *
 * @param {number} index safe integer in [0, games)
 * @param {boolean} [smoke=false]
 */
export function gameSpec(index, smoke = false) {
  const plan = planFor(smoke);
  if (!Number.isSafeInteger(index)) {
    throw new RangeError(`index must be a safe integer, received ${describe(index)}`);
  }
  if (index < 0 || index >= plan.games) {
    throw new RangeError(`index must be in [0, ${plan.games}), received ${index}`);
  }

  const session = Math.floor(index / plan.games_per_session);
  const within_session = index % plan.games_per_session;
  const pair = Math.floor(index / 2);
  const mirror = index % 2;
  const tactical_team = pair * 2 + mirror;
  const baseline_team = pair * 2 + 1 - mirror;
  const tactical_seat = mirror ? SEAT_P2 : SEAT_P1;
  const challenger = mirror ? 'baseline' : 'tactical';

  const parity = session % 2;
  let scenario = NORMAL_SCENARIO;
  for (let k = 0; k < plan.fault_offsets.length; k += 1) {
    if (plan.fault_offsets[k] + parity === within_session) {
      scenario = plan.faults[k];
      break;
    }
  }

  const file = `game-${String(index).padStart(4, '0')}.json.gz`;
  const names = {
    tactical: `SoakTac${session}`,
    baseline: `SoakBase${session}`,
  };

  return {
    index,
    session,
    within_session,
    pair,
    mirror,
    tactical_team,
    baseline_team,
    tactical_seat,
    challenger,
    scenario,
    file,
    names,
  };
}

/**
 * Derives the expected coverage totals by walking the full schedule.
 *
 * - rematches: every game that is not the first game of its session
 * - faults: games whose scenario is not "normal"
 * - tactical_connections: one initial login per session plus one new
 *   connection per fault recovery
 * - baseline_connections: one per session (baseline transport stays connected)
 *
 * @param {boolean} [smoke=false]
 */
export function expectedCoverage(smoke = false) {
  const plan = planFor(smoke);
  const scenarios = { [NORMAL_SCENARIO]: 0 };
  for (const fault of plan.faults) scenarios[fault] = 0;

  let rematches = 0;
  let faults = 0;
  let tactical_p1 = 0;
  let tactical_p2 = 0;

  for (let index = 0; index < plan.games; index += 1) {
    const spec = gameSpec(index, smoke);
    if (spec.within_session !== 0) rematches += 1;
    if (spec.scenario !== NORMAL_SCENARIO) faults += 1;
    if (spec.tactical_seat === SEAT_P1) tactical_p1 += 1;
    else tactical_p2 += 1;
    scenarios[spec.scenario] = (scenarios[spec.scenario] ?? 0) + 1;
  }

  return {
    games: plan.games,
    sessions: plan.sessions,
    rematches,
    faults,
    tactical_connections: plan.sessions + faults,
    baseline_connections: plan.sessions,
    tactical_p1,
    tactical_p2,
    scenarios,
  };
}
