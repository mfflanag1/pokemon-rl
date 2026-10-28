import {readFileSync} from 'node:fs';
import {PROFILE as BASE, SIDES, ROSTER, team, reference as baseReference} from '../switch-reference/profile.mjs';
export {SIDES,ROSTER,team};
export const PLAN=JSON.parse(readFileSync(new URL('../../contracts/continuation-v1/plan.json',import.meta.url)));
export const PROFILE={...BASE,id:'gen9-starter-pairs-continuation-v1',max_resolved_turns:PLAN.max_turns,max_decisions:2*PLAN.max_groups};
export function reference(){return {base:baseReference(),continuation:PLAN};}
