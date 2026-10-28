import {validatePlayer} from '../full-showdown/schema.mjs';
import {ACTIONS, requestActions} from '../full-showdown/actions.mjs';
import {chooseOpponent} from '../showdown-panel/policy.mjs';

// Diagnostic reconstruction, never an actor replacement. Inputs are one player's
// validated projection; missing public history is explicitly left unknown.
export function reconstruct(player) {
  validatePlayer(player);
  const seat=player.side, enemy=seat==='p1'?'p2':'p1';
  const team=player.own.map((p,i)=>({...structuredClone(p), ident:`${seat}: slot${i}`,
    details:`${p.species}, L${p.level}`, teraType:p.tera_type, terastallized:p.tera}));
  const own=team.find(p=>p.active), foe=player.opponent.find(p=>p?.active);
  const request={side:{id:seat,pokemon:team}};
  if(player.phase==='wait') request.wait=true;
  else if(player.phase!=='move') request.forceSwitch=[true];
  else request.active=[{moves:player.active_moves.map((m,i)=>({id:m.id,move:m.id,pp:m.pp,maxpp:m.maxpp,
    disabled:!player.mask[i],basePower:m.base_power,type:m.type})),
    trapped:!player.mask.slice(8).some(Boolean),canTerastallize:player.mask.slice(4,8).some(Boolean)?own.tera_type:undefined}];
  if(JSON.stringify(requestActions(request).mask)!==JSON.stringify(player.mask)) throw Error('Reconstructed legality differs');
  const view={turn:player.turn,weather:player.weather,terrain:player.terrain,
    hazards:{[seat]:structuredClone(player.side_conditions.own),[enemy]:structuredClone(player.side_conditions.opponent)},
    active:{[seat]:structuredClone(own),[enemy]:structuredClone(foe)}};
  return {request,view};
}
export function projectedAdvice(player, publicView=null) {
  if(!player.mask.some(Boolean)) return null;
  const {request,view}=reconstruct(player);
  const choose=v=>{const a=ACTIONS.indexOf(chooseOpponent('tactical-v1',request,v));if(a<0||!player.mask[a])throw Error('Illegal projected advice');return a;};
  const result={projection:choose(view)};
  if(publicView){
    for(const [name,fields] of Object.entries({entry:['enteredTurn'],types:['types','addedType'],boosts:['boosts'],entry_types_boosts:['enteredTurn','types','addedType','boosts']})){
      const changed=structuredClone(view);
      for(const side of ['p1','p2'])for(const field of fields){
        delete changed.active[side][field];
        if(publicView.active[side]?.[field]!==undefined)changed.active[side][field]=structuredClone(publicView.active[side][field]);
      }
      const a=ACTIONS.indexOf(chooseOpponent('tactical-v1',request,changed));if(a<0||!player.mask[a])throw Error('Illegal restored advice');result[name]=a;
    }
  }
  return result;
}
