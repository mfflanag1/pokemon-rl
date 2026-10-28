import {Dex} from '../reference-node/engine.mjs';
const types=['Normal','Fire','Water','Electric','Grass','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy','Stellar'];
const names=new Set(Dex.species.all().flatMap(s=>[s.name,...(s.otherFormes??[]),...(s.cosmeticFormes??[])]));
for(const alias of Dex.loadAliases().keys()) if(Dex.species.get(alias).exists) names.add(alias);
const species=Object.fromEntries([...names].map(name=>[name.toLowerCase().replace(/[^a-z0-9]/g,''),{types:Dex.species.get(name).types}]));
const moves=Object.fromEntries(Dex.moves.all().filter(m=>m.exists).map(m=>[m.id,{type:m.type,category:m.category,base_power:m.basePower}]));
moves.recharge={type:'???',category:'Status',base_power:0};
const effectiveness=Object.fromEntries(types.map(a=>[a,Object.fromEntries([...types,'???'].map(d=>[d,Dex.getImmunity(a,[d])?2**Dex.getEffectiveness(a,[d]):0]))]));
console.log(JSON.stringify({schema_version:'f17-public-metadata-v1',types,species,moves,effectiveness}));
