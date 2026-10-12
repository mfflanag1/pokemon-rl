# Pinned validation distinction

Source vendor/pokemon-showdown/sim/team-validator.ts SHA256 0e2f95edb157e34a8c72b4fd1bf644ae65494b00993ccb79cd7fdf16ddb4ab94

```typescript
356: 	validateTeam(
357: 		team: PokemonSet[] | null,
358: 		options: {
359: 			removeNicknames?: boolean,
360: 			skipSets?: { [name: string]: { [key: string]: boolean } },
361: 		} = {}
362: 	): string[] | null {
363: 		if (team && this.format.validateTeam) {
364: 			return this.format.validateTeam.call(this, team, options) || null;
365: 		}
366: 		return this.baseValidateTeam(team, options);
367: 	}
368: 
369: 	baseValidateTeam(
370: 		team: PokemonSet[] | null,
371: 		options: {
372: 			removeNicknames?: boolean,
373: 			skipSets?: { [name: string]: { [key: string]: boolean } },
374: 		} = {}
375: 	): string[] | null {
376: 		const format = this.format;
377: 		const dex = this.dex;
378: 
379: 		let problems: string[] = [];
380: 		const ruleTable = this.ruleTable;
381: 		if (format.team) {
382: 			if (team) {
383: 				return [
384: 					`This format doesn't let you use your own team.`,
385: 					`If you're not using a custom client, please report this as a bug. If you are, remember to use \`/utm null\` before starting a game in this format.`,
386: 				];
387: 			}
388: 			const testTeamSeed = PRNG.generateSeed();
389: 			try {
390: 				const testTeamGenerator = Teams.getGenerator(format, testTeamSeed);
391: 				testTeamGenerator.getTeam(options); // Throws error if generation fails
392: 			} catch (e) {
393: 				return [
394: 					`${format.name}'s team generator (${format.team}) failed using these rules and seed (${testTeamSeed}):-`,
395: 					`${e}`,
396: 				];
397: 			}
398: 			return null;
399: 		}
400: 		if (!team) {
401: 			return [
402: 				`This format requires you to use your own team.`,
403: 				`If you're not using a custom client, please report this as a bug.`,
404: 			];
405: 		}
406: 		if (!Array.isArray(team)) {
407: 			throw new Error(`Invalid team data`);
408: 		}
```

The normal format with a team generator checks the generator and returns without per-set constructed legality. Our additive local format supplies already regenerated official development teams, accepts only their exact packed representations, and keeps all battle rules. It neither replaces the rejected official sets nor changes their IVs or other values.

## sim/pokemon.ts

SHA256 e36493e502858c7f7fa8e899f4b4439295fefa452f96c5017e974494f1c4a688

```
318: 		if (this.battle.format.pokemon) Object.assign(this, this.battle.format.pokemon);
319: 
320: 		if (typeof set === 'string') set = { name: set };
321: 
322: 		this.baseSpecies = this.battle.dex.species.get(set.species || set.name);
323: 		if (!this.baseSpecies.exists) {
324: 			throw new Error(`Unidentified species: ${this.baseSpecies.name}`);
325: 		}
326: 		this.set = set as PokemonSet;
327: 
328: 		this.species = this.baseSpecies;
329: 		if (set.name === set.species || !set.name) {
330: 			set.name = this.baseSpecies.baseSpecies;
331: 		}
332: 		this.speciesState = this.battle.initEffectState({ id: this.species.id });
333: 
334: 		this.name = set.name.substr(0, 20);
335: 		this.fullname = `${this.side.id}: ${this.name}`;
```

## server/index.ts

SHA256 9d1fa3391ab71aa85a938c5fd02d6a75602ef3199a9313cb68cd6dbfd59b5680

```
90: 	const { Dex } = require('../sim/dex');
91: 	global.Dex = Dex;
92: 	global.toID = Dex.toID;
93: 
94: 	const { Chat } = require('./chat');
95: 	global.Chat = Chat;
96: 
97: 	const { Rooms } = require('./rooms');
98: 	global.Rooms = Rooms;
99: 	// We initialize the global room here because roomlogs.ts needs the Rooms global
100: 	Rooms.global = new Rooms.GlobalRoomState();
101: 
102: 	const { Teams } = require('../sim/teams');
103: 	global.Teams = Teams;
104: 
105: 	const { LoginServer } = require('./loginserver');
106: 	global.LoginServer = LoginServer;
```

## server/team-validator-async.ts

SHA256 12f51e723bf6ce2676661fb953dd75481251c6380bba6d698c0b000394f97f41

```
88: 	} as typeof Monitor;
89: 
90: 	if (Config.crashguard) {
91: 		process.on('uncaughtException', (err: Error) => {
92: 			Monitor.crashlog(err, `A team validator process`);
93: 		});
94: 		process.on('unhandledRejection', err => {
95: 			Monitor.crashlog(err as any || {}, 'A team validator process Promise');
96: 		});
97: 	}
98: 
99: 	global.Dex = require('../sim/dex').Dex.includeData();
100: 	global.Teams = require('../sim/teams').Teams;
101: 
102: 	// eslint-disable-next-line no-eval
103: 	PM.startRepl((cmd: string) => eval(cmd));
```

## config/config-example.js

SHA256 05fb15ac3bf2bed1a45ba6cc30183efc37fbb5896bbb58329bcf457cbf9611e9

```
85: exports.proxyip = false;
86: 
87: // subprocesses - the number of child processes to use for various tasks.
88: //   Can be set to `0` instead of `{...}` to stop using subprocesses, if you're running out of RAM.
89: exports.subprocesses = {
90: 	/**
91: 	 * network - the number of networking child processes to spawn
92: 	 *   This should be no greater than the number of threads available on your
93: 	 *   server's CPU. If you're not sure how many you have, you can check from a
94: 	 *   terminal by running:
95: 	 *
96: 	 *   $ node -e "console.log(require('os').cpus().length)"
97: 	 *
98: 	 *   Using more workers than there are available threads will cause performance
```

## sim/dex-formats.ts

SHA256 94ac9f4ea4be767603d35336e8019d7daef4fa0a24f1b66727f2ceb8efa9e545

```
630: 		if (this.formatsListCache) return this;
631: 
632: 		const formatsList = [];
633: 
634: 		// Load formats
635: 		let customFormats;
636: 		try {
637: 			customFormats = require(`${__dirname}/../config/custom-formats`).Formats;
638: 			if (!Array.isArray(customFormats)) {
639: 				throw new TypeError(`Exported property 'Formats' from "./config/custom-formats.ts" must be an array`);
640: 			}
641: 		} catch (e: any) {
642: 			if (e.code !== 'MODULE_NOT_FOUND' && e.code !== 'ENOENT') {
643: 				throw e;
644: 			}
645: 		}
646: 		let Formats: AnyObject[] = require(`${__dirname}/../config/formats`).Formats;
647: 		if (!Array.isArray(Formats)) {
648: 			throw new TypeError(`Exported property 'Formats' from "./config/formats.ts" must be an array`);
649: 		}
650: 		if (customFormats) Formats = mergeFormatLists(Formats as any, customFormats);
651: 
652: 		let section = '';
653: 		let column = 1;
654: 		for (const [i, format] of Formats.entries()) {
655: 			const id = toID(format.name);
```

## config/formats.ts

SHA256 c10c80f8e8b32b618dcf8d15572f2573593d478ee8e7c6d6cfe22ab28cd5faea

```
26: 		section: "S/V Singles",
27: 	},
28: 	{
29: 		name: "[Gen 9] Random Battle",
30: 		desc: `Randomized teams of Pok&eacute;mon with sets that are generated to be competitively viable.`,
31: 		mod: 'gen9',
32: 		team: 'random',
33: 		bestOfDefault: true,
34: 		ruleset: ['PotD', 'Obtainable', 'Species Clause', 'HP Percentage Mod', 'Cancel Mod', 'Sleep Clause Mod', 'Illusion Level Mod'],
35: 	},
```
