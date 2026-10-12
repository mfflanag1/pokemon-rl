
## vendor/pokemon-showdown/data/moves.ts SHA256 738b8cfc1ca491537c0188546ab697942aaa313c5e6be83a5432edd5dd8997f7

14476: 	quickattack: {
14477: 		num: 98,
14478: 		accuracy: 100,
14479: 		basePower: 40,
14480: 		category: "Physical",
14481: 		name: "Quick Attack",
14482: 		pp: 30,
14483: 		priority: 1,
14484: 		flags: { contact: 1, protect: 1, mirror: 1, metronome: 1 },
14485: 		target: "normal",
14486: 		type: "Normal",
14487: 		contestType: "Cool",
14488: 	},
14489: 	quickguard: {
14490: 		num: 501,
14491: 		accuracy: true,
14492: 		basePower: 0,
14493: 		category: "Status",
14494: 		name: "Quick Guard",
14495: 		pp: 15,
18812: 	tackle: {
18813: 		num: 33,
18814: 		accuracy: 100,
18815: 		basePower: 40,
18816: 		category: "Physical",
18817: 		name: "Tackle",
18818: 		pp: 35,
18819: 		priority: 0,
18820: 		flags: { contact: 1, protect: 1, mirror: 1, metronome: 1 },
18821: 		target: "normal",
18822: 		type: "Normal",
18823: 		contestType: "Tough",
18824: 	},
18825: 	tailglow: {
18826: 		num: 294,
18827: 		accuracy: true,
18828: 		basePower: 0,
18829: 		category: "Status",
18830: 		name: "Tail Glow",
18831: 		pp: 20,
18832: 		priority: 0,
18833: 		flags: { snatch: 1, metronome: 1 },
18834: 		boosts: {

## vendor/pokemon-showdown/data/abilities.ts SHA256 53d8558979e337c675907b76b5375185b28a0877f770d7456025b94535e1064d

3951: 	runaway: {
3952: 		flags: {},
3953: 		name: "Run Away",
3954: 		rating: 0,
3955: 		num: 50,
3956: 	},
3957: 	sandforce: {
3958: 		onBasePowerPriority: 21,
3959: 		onBasePower(basePower, attacker, defender, move) {
3960: 			if (this.field.isWeather('sandstorm')) {
3961: 				if (move.type === 'Rock' || move.type === 'Ground' || move.type === 'Steel') {
3962: 					this.debug('Sand Force boost');
3963: 					return this.chainModify([5325, 4096]);
3964: 				}

## vendor/pokemon-showdown/data/pokedex.ts SHA256 73048386b864be5aff093e9393acf32e8016299e9d7b76078bf5120b769e2fe0

410: 	rattata: {
411: 		num: 19,
412: 		name: "Rattata",
413: 		types: ["Normal"],
414: 		baseStats: { hp: 30, atk: 56, def: 35, spa: 25, spd: 35, spe: 72 },
415: 		abilities: { 0: "Run Away", 1: "Guts", H: "Hustle" },
416: 		heightm: 0.3,
417: 		weightkg: 3.5,
418: 		color: "Purple",
419: 		evos: ["Raticate"],
420: 		eggGroups: ["Field"],
421: 		otherFormes: ["Rattata-Alola"],
422: 		formeOrder: ["Rattata", "Rattata-Alola"],
423: 	},
424: 	rattataalola: {
425: 		num: 19,
426: 		name: "Rattata-Alola",
427: 		baseSpecies: "Rattata",
428: 		forme: "Alola",
429: 		types: ["Dark", "Normal"],
430: 		baseStats: { hp: 30, atk: 56, def: 35, spa: 25, spd: 35, spe: 72 },
431: 		abilities: { 0: "Gluttony", 1: "Hustle", H: "Thick Fat" },
432: 		heightm: 0.3,
433: 		weightkg: 3.8,
434: 		color: "Black",
435: 		evos: ["Raticate-Alola"],
436: 		eggGroups: ["Field"],
437: 	},
438: 	raticate: {
439: 		num: 20,
440: 		name: "Raticate",
441: 		types: ["Normal"],
442: 		baseStats: { hp: 55, atk: 81, def: 60, spa: 50, spd: 70, spe: 97 },
443: 		abilities: { 0: "Run Away", 1: "Guts", H: "Hustle" },
444: 		heightm: 0.7,
445: 		weightkg: 18.5,
446: 		color: "Brown",
447: 		prevo: "Rattata",
448: 		evoLevel: 20,
449: 		eggGroups: ["Field"],
450: 		otherFormes: ["Raticate-Alola", "Raticate-Alola-Totem"],
451: 		formeOrder: ["Raticate", "Raticate-Alola", "Raticate-Alola-Totem"],
452: 	},
453: 	raticatealola: {
454: 		num: 20,
455: 		name: "Raticate-Alola",
456: 		baseSpecies: "Raticate",
457: 		forme: "Alola",
458: 		types: ["Dark", "Normal"],
459: 		baseStats: { hp: 75, atk: 71, def: 70, spa: 40, spd: 80, spe: 77 },
460: 		abilities: { 0: "Gluttony", 1: "Hustle", H: "Thick Fat" },
461: 		heightm: 0.7,
462: 		weightkg: 25.5,

## vendor/pokemon-showdown/sim/battle-actions.ts SHA256 4342d03aa909875ce0a4c4f3f54027ae1933b60e22be778409cef992300ecb9d

1576: 	getDamage(
1577: 		source: Pokemon, target: Pokemon, move: string | number | ActiveMove,
1578: 		suppressMessages = false
1579: 	): number | undefined | null | false {
1580: 		if (typeof move === 'string') move = this.dex.getActiveMove(move);
1581: 
1582: 		if (typeof move === 'number') {
1583: 			const basePower = move;
1584: 			move = new Dex.Move({
1585: 				basePower,
1586: 				type: '???',
1587: 				category: 'Physical',
1588: 				willCrit: false,
1589: 			}) as ActiveMove;
1590: 			move.hit = 0;
1591: 		}
1592: 
1593: 		if (!target.runImmunity(move, !suppressMessages)) {
1594: 			return false;
1595: 		}
1596: 
1597: 		if (move.ohko) return this.battle.gen === 3 ? target.hp : target.maxhp;
1598: 		if (move.damageCallback) return move.damageCallback.call(this.battle, source, target);
1599: 		if (move.damage === 'level') {
1600: 			return source.level;
1601: 		} else if (move.damage) {
1602: 			return move.damage;
1603: 		}
1604: 
1605: 		const category = this.battle.getCategory(move);
1606: 
1607: 		let basePower: number | false | null = move.basePower;
1608: 		if (move.basePowerCallback) {
1609: 			basePower = move.basePowerCallback.call(this.battle, source, target, move);
1610: 		}
1611: 		if (!basePower) return basePower === 0 ? undefined : basePower;
1612: 		basePower = this.battle.clampIntRange(basePower, 1);
1613: 
1614: 		let critMult;
1615: 		let critRatio = this.battle.runEvent('ModifyCritRatio', source, target, move, move.critRatio || 0);
1616: 		if (this.battle.gen <= 5) {
1617: 			critRatio = this.battle.clampIntRange(critRatio, 0, 5);
1618: 			critMult = [0, 16, 8, 4, 3, 2];
1619: 		} else {
1620: 			critRatio = this.battle.clampIntRange(critRatio, 0, 4);
1621: 			if (this.battle.gen === 6) {
1622: 				critMult = [0, 16, 8, 2, 1];
1623: 			} else {
1624: 				critMult = [0, 24, 8, 2, 1];
1625: 			}
1626: 		}
1627: 
1628: 		const moveHit = target.getMoveHitData(move);
1629: 		moveHit.crit = move.willCrit || false;
1630: 		if (move.willCrit === undefined) {
1631: 			if (critRatio) {
1632: 				moveHit.crit = this.battle.randomChance(1, critMult[critRatio]);
1633: 			}
1634: 		}
1635: 
1636: 		if (moveHit.crit) {
1637: 			moveHit.crit = this.battle.runEvent('CriticalHit', target, null, move);
1638: 		}
1639: 
1640: 		// happens after crit calculation
1641: 		basePower = this.battle.runEvent('BasePower', source, target, move, basePower, true);
1642: 
1643: 		if (!basePower) return 0;
1644: 		basePower = this.battle.clampIntRange(basePower, 1);
1645: 		// Hacked Max Moves have 0 base power, even if you Dynamax
1646: 		if ((!source.volatiles['dynamax'] && move.isMax) || (move.isMax && this.dex.moves.get(move.baseMove).isMax)) {
1647: 			basePower = 0;
1648: 		}
1649: 
1650: 		const dexMove = this.dex.moves.get(move.id);
1651: 		if (source.terastallized && (source.terastallized === 'Stellar' ?
1652: 			!source.stellarBoostedTypes.includes(move.type) : source.hasType(move.type)) &&
1653: 			basePower < 60 && dexMove.priority <= 0 && !dexMove.multihit &&
1654: 			// Hard move.basePower check for moves like Dragon Energy that have variable BP
1655: 			!((move.basePower === 0 || move.basePower === 150) && move.basePowerCallback)
1656: 		) {
1657: 			basePower = 60;
1658: 		}
1659: 
1660: 		const level = source.level;
1661: 
1662: 		const attacker = move.overrideOffensivePokemon === 'target' ? target : source;
1663: 		const defender = move.overrideDefensivePokemon === 'source' ? source : target;
1664: 
1665: 		const isPhysical = move.category === 'Physical';
1666: 		let attackStat: StatIDExceptHP = move.overrideOffensiveStat || (isPhysical ? 'atk' : 'spa');
1667: 		const defenseStat: StatIDExceptHP = move.overrideDefensiveStat || (isPhysical ? 'def' : 'spd');
1668: 
1669: 		const statTable = { atk: 'Atk', def: 'Def', spa: 'SpA', spd: 'SpD', spe: 'Spe' };
1670: 
1671: 		let atkBoosts = attacker.boosts[attackStat];
1672: 		let defBoosts = defender.boosts[defenseStat];
1673: 
1674: 		let ignoreNegativeOffensive = !!move.ignoreNegativeOffensive;
1675: 		let ignorePositiveDefensive = !!move.ignorePositiveDefensive;
1676: 
1677: 		if (moveHit.crit) {
1678: 			ignoreNegativeOffensive = true;
1679: 			ignorePositiveDefensive = true;
1680: 		}
1681: 		const ignoreOffensive = !!(move.ignoreOffensive || (ignoreNegativeOffensive && atkBoosts < 0));
1682: 		const ignoreDefensive = !!(move.ignoreDefensive || (ignorePositiveDefensive && defBoosts > 0));
1683: 
1684: 		if (ignoreOffensive) {
1685: 			this.battle.debug('Negating (sp)atk boost/penalty.');
1686: 			atkBoosts = 0;
1687: 		}
1688: 		if (ignoreDefensive) {
1689: 			this.battle.debug('Negating (sp)def boost/penalty.');
1690: 			defBoosts = 0;
1691: 		}
1692: 
1693: 		let attack = attacker.calculateStat(attackStat, atkBoosts, 1, source);
1694: 		let defense = defender.calculateStat(defenseStat, defBoosts, 1, target);
1695: 
1696: 		attackStat = (category === 'Physical' ? 'atk' : 'spa');
1697: 
1698: 		// Apply Stat Modifiers
1699: 		attack = this.battle.runEvent('Modify' + statTable[attackStat], source, target, move, attack);
1700: 		defense = this.battle.runEvent('Modify' + statTable[defenseStat], target, source, move, defense);
1701: 
1702: 		if (this.battle.gen <= 4 && ['explosion', 'selfdestruct'].includes(move.id) && defenseStat === 'def') {
1703: 			defense = this.battle.clampIntRange(Math.floor(defense / 2), 1);
1704: 		}
1705: 
1706: 		const tr = this.battle.trunc;
1707: 
1708: 		// int(int(int(2 * L / 5 + 2) * A * P / D) / 50);
1709: 		const baseDamage = tr(tr(tr(tr(2 * level / 5 + 2) * basePower * attack) / defense) / 50);
1710: 
1711: 		// Calculate damage modifiers separately (order differs between generations)
1712: 		return this.modifyDamage(baseDamage, source, target, move, suppressMessages);
1713: 	}
1714: 
1715: 	modifyDamage(
1716: 		baseDamage: number, pokemon: Pokemon, target: Pokemon, move: ActiveMove, suppressMessages = false
1717: 	) {
1718: 		const tr = this.battle.trunc;
1719: 		if (!move.type) move.type = '???';
1720: 		const type = move.type;
1721: 
1722: 		baseDamage += 2;
1723: 
1724: 		if (move.spreadHit) {
1725: 			// multi-target modifier (doubles only)
1726: 			const spreadModifier = this.battle.gameType === 'freeforall' ? 0.5 : 0.75;
1727: 			this.battle.debug(`Spread modifier: ${spreadModifier}`);
1728: 			baseDamage = this.battle.modify(baseDamage, spreadModifier);
1729: 		} else if (move.multihitType === 'parentalbond' && move.hit > 1) {
1730: 			// Parental Bond modifier
1731: 			const bondModifier = this.battle.gen > 6 ? 0.25 : 0.5;
1732: 			this.battle.debug(`Parental Bond modifier: ${bondModifier}`);
1733: 			baseDamage = this.battle.modify(baseDamage, bondModifier);
1734: 		}
1735: 
1736: 		// weather modifier
1737: 		baseDamage = this.battle.priorityEvent('WeatherModifyDamage', pokemon, target, move, baseDamage);
1738: 
1739: 		// crit - not a modifier
1740: 		const isCrit = target.getMoveHitData(move).crit;
1741: 		if (isCrit) {
1742: 			baseDamage = tr(baseDamage * (move.critModifier || (this.battle.gen >= 6 ? 1.5 : 2)));
1743: 		}
1744: 
1745: 		// random factor - also not a modifier
1746: 		baseDamage = this.battle.randomizer(baseDamage);
1747: 
1748: 		// STAB
1749: 		// The "???" type never gets STAB
1750: 		// Not even if you Roost in Gen 4 and somehow manage to use
1751: 		// Struggle in the same turn.
1752: 		// (On second thought, it might be easier to get a MissingNo.)
1753: 		if (type !== '???') {
1754: 			let stab: number | [number, number] = 1;
1755: 
1756: 			const isSTAB = move.forceSTAB || pokemon.hasType(type) || pokemon.getTypes(false, true).includes(type);
1757: 			if (isSTAB) {
1758: 				stab = 1.5;
1759: 			}
1760: 
1761: 			// The Stellar tera type makes this incredibly confusing
1762: 			// If the move's type does not match one of the user's base types,
1763: 			// the Stellar tera type applies a one-time 1.2x damage boost for that type.
1764: 			//
1765: 			// If the move's type does match one of the user's base types,
1766: 			// then the Stellar tera type applies a one-time 2x STAB boost for that type,
1767: 			// and then goes back to using the regular 1.5x STAB boost for those types.
1768: 			if (pokemon.terastallized === 'Stellar') {
1769: 				if (!pokemon.stellarBoostedTypes.includes(type) || move.stellarBoosted) {
1770: 					stab = isSTAB ? 2 : [4915, 4096];
1771: 					move.stellarBoosted = true;
1772: 					if (pokemon.species.name !== 'Terapagos-Stellar') {
1773: 						pokemon.stellarBoostedTypes.push(type);
1774: 					}
1775: 				}
1776: 			} else {
1777: 				if (pokemon.terastallized === type && pokemon.getTypes(false, true).includes(type)) {
1778: 					stab = 2;
1779: 				}
1780: 				stab = this.battle.runEvent('ModifySTAB', pokemon, target, move, stab);
1781: 			}
1782: 
1783: 			baseDamage = this.battle.modify(baseDamage, stab);
1784: 		}
1785: 
1786: 		// types
1787: 		let typeMod = target.runEffectiveness(move);
1788: 		typeMod = this.battle.clampIntRange(typeMod, -6, 6);
1789: 		target.getMoveHitData(move).typeMod = typeMod;
1790: 		if (typeMod > 0) {
1791: 			if (!suppressMessages) this.battle.add('-supereffective', target);
1792: 
1793: 			for (let i = 0; i < typeMod; i++) {
1794: 				baseDamage *= 2;
1795: 			}
1796: 		}
1797: 		if (typeMod < 0) {
1798: 			if (!suppressMessages) this.battle.add('-resisted', target);
1799: 
1800: 			for (let i = 0; i > typeMod; i--) {
1801: 				baseDamage = tr(baseDamage / 2);
1802: 			}
1803: 		}
1804: 
1805: 		if (isCrit && !suppressMessages) this.battle.add('-crit', target);
1806: 
1807: 		if (pokemon.status === 'brn' && move.category === 'Physical' && !pokemon.hasAbility('guts')) {
1808: 			if (this.battle.gen < 6 || move.id !== 'facade') {
1809: 				baseDamage = this.battle.modify(baseDamage, 0.5);
1810: 			}
1811: 		}
1812: 
1813: 		// Generation 5, but nothing later, sets damage to 1 before the final damage modifiers
1814: 		if (this.battle.gen === 5 && !baseDamage) baseDamage = 1;
1815: 
1816: 		// Final modifier. Modifiers that modify damage after min damage check, such as Life Orb.
1817: 		baseDamage = this.battle.runEvent('ModifyDamage', pokemon, target, move, baseDamage);
1818: 
1819: 		const bypassProtect = target.getMoveHitData(move).bypassProtect;
1820: 		if (bypassProtect) {
1821: 			baseDamage = this.battle.modify(baseDamage, 0.25);
1822: 			if (bypassProtect !== true && bypassProtect.effectType === 'Ability') {
1823: 				this.battle.add('-ability', pokemon, bypassProtect.name);
1824: 			}
1825: 			this.battle.add('-zbroken', target);
1826: 		}
1827: 
1828: 		// Generation 6-7 moves the check for minimum 1 damage after the final modifier...
1829: 		if (this.battle.gen !== 5 && !baseDamage) return 1;
1830: 

## vendor/pokemon-showdown/sim/battle.ts SHA256 3eb941313832729ee043407c39ced894b4338301fed5d29a45ae8b36e49a09f2

400: 	 * 1. Order, low to high (default last)
401: 	 * 2. Priority, high to low (default 0)
402: 	 * 3. Speed, high to low (default 0)
403: 	 * 4. SubOrder, low to high (default 0)
404: 	 * 5. EffectOrder, low to high (default 0)
405: 	 *
406: 	 * Doesn't reference `this` so doesn't need to be bound.
407: 	 */
408: 	comparePriority(this: void, a: AnyObject, b: AnyObject) {
409: 		return -((b.order || 4294967296) - (a.order || 4294967296)) ||
410: 			((b.priority || 0) - (a.priority || 0)) ||
411: 			((b.speed || 0) - (a.speed || 0)) ||
412: 			-((b.subOrder || 0) - (a.subOrder || 0)) ||
413: 			-((b.effectOrder || 0) - (a.effectOrder || 0)) ||
414: 			0;
415: 	}
416: 
417: 	static compareRedirectOrder(this: void, a: AnyObject, b: AnyObject) {
418: 		return ((b.priority || 0) - (a.priority || 0)) ||
419: 			((b.speed || 0) - (a.speed || 0)) ||
420: 			((a.effectHolder?.abilityState && b.effectHolder?.abilityState) ?
421: 				-(b.effectHolder.abilityState.effectOrder - a.effectHolder.abilityState.effectOrder) : 0) ||
422: 				0;
423: 	}
424: 
425: 	static compareLeftToRightOrder(this: void, a: AnyObject, b: AnyObject) {
426: 		return -((b.order || 4294967296) - (a.order || 4294967296)) ||
427: 			((b.priority || 0) - (a.priority || 0)) ||
428: 			-((b.index || 0) - (a.index || 0)) ||
429: 			0;
430: 	}
431: 
432: 	/** Sort a list, resolving speed ties the way the games do. */
433: 	speedSort<T extends AnyObject>(list: T[], comparator: (a: T, b: T) => number = this.comparePriority) {
434: 		if (list.length < 2) return;
435: 		let sorted = 0;
436: 		// This is a Selection Sort - not the fastest sort in general, but
437: 		// actually faster than QuickSort for small arrays like the ones
438: 		// `speedSort` is used for.
439: 		// More importantly, it makes it easiest to resolve speed ties
440: 		// properly.
441: 		while (sorted + 1 < list.length) {
442: 			let nextIndexes = [sorted];
443: 			// grab list of next indexes
444: 			for (let i = sorted + 1; i < list.length; i++) {
445: 				const delta = comparator(list[nextIndexes[0]], list[i]);
446: 				if (delta < 0) continue;
447: 				if (delta > 0) nextIndexes = [i];
448: 				if (delta === 0) nextIndexes.push(i);
449: 			}
450: 			// put list of next indexes where they belong
451: 			for (let i = 0; i < nextIndexes.length; i++) {
452: 				const index = nextIndexes[i];
453: 				if (index !== sorted + i) {
454: 					// nextIndexes is guaranteed to be in order, so it will never have
455: 					// been disturbed by an earlier swap
456: 					[list[sorted + i], list[index]] = [list[index], list[sorted + i]];
457: 				}
458: 			}
459: 			if (nextIndexes.length > 1) {
460: 				this.prng.shuffle(list, sorted, sorted + nextIndexes.length);
461: 			}
462: 			sorted += nextIndexes.length;
2635: 			if (action.maxMove) {
2636: 				const maxMoveName = this.actions.getMaxMove(action.maxMove, action.pokemon);
2637: 				if (maxMoveName) {
2638: 					const maxMove = this.actions.getActiveMaxMove(action.move, action.pokemon);
2639: 					if (maxMove.exists && maxMove.isMax) {
2640: 						move = maxMove;
2641: 					}
2642: 				}
2643: 			}
2644: 			// take priority from the base move, so abilities like Prankster only apply once
2645: 			// (instead of compounding every time `getActionSpeed` is called)
2646: 			let priority = this.dex.moves.get(move.id).priority;
2647: 			// Grassy Glide priority
2648: 			const target = this.getTarget(action.pokemon, action.move, action.targetLoc);
2649: 			priority = this.singleEvent('ModifyPriority', move, null, action.pokemon, target, null, priority);
2650: 			priority = this.runEvent('ModifyPriority', action.pokemon, target, move, priority);
2651: 			action.priority = priority + action.fractionalPriority;
2652: 			// In Gen 6, Quick Guard blocks moves with artificially enhanced priority.
2653: 			if (this.gen > 5) action.move.priority = priority;
2654: 		}
2655: 
2656: 		if (!action.pokemon) {
2657: 			action.speed = 1;
2658: 		} else {
2659: 			if (this.gen <= 4 && action.choice === 'move' && action.fractionalPriority < 0) {
2660: 				// in Gen 4, Pokemon with decrease fractional priority act in reverse speed order
2661: 				// ignores Trick Room, does not ignore boosts and Simple
2662: 				action.speed = -action.pokemon.getStat('spe', false, false);
2663: 			} else {
2664: 				action.speed = action.pokemon.getActionSpeed();
2665: 			}
2666: 		}
2667: 	}
2668: 
2669: 	runAction(action: Action) {
2670: 		const pokemonOriginalHP = action.pokemon?.hp;
2671: 		let residualPokemon: (readonly [Pokemon, number])[] = [];
2672: 		// returns whether or not we ended in a callback
2673: 		switch (action.choice) {
2674: 		case 'start': {
2675: 			for (const side of this.sides) {
2676: 				if (side.pokemonLeft) side.pokemonLeft = side.pokemon.length;
2677: 				this.add('teamsize', side.id, side.pokemon.length);
2678: 			}
2679: 
2680: 			this.add('start');
2681: 
2682: 			for (const pokemon of this.getAllPokemon()) {
2683: 				this.singleEvent('BattleStart', this.dex.conditions.getByID(pokemon.species.id), pokemon.speciesState, pokemon);
2684: 			}
2685: 
2686: 			this.format.onBattleStart?.call(this);
2687: 			for (const rule of this.ruleTable.keys()) {
2688: 				if ('+*-!'.includes(rule.charAt(0))) continue;
2689: 				const subFormat = this.dex.formats.get(rule);
2690: 				subFormat.onBattleStart?.call(this);
2691: 			}
2692: 
2693: 			for (const side of this.sides) {
2694: 				for (let i = 0; i < side.active.length; i++) {
2695: 					if (!side.pokemonLeft) {
2696: 						// forfeited before starting
2697: 						side.active[i] = side.pokemon[i];
2698: 						side.active[i].fainted = true;
2699: 						side.active[i].hp = 0;
2700: 					} else {
2701: 						this.actions.switchIn(side.pokemon[i], i);
2702: 					}
2703: 				}
2704: 			}
2705: 			this.midTurn = true;
