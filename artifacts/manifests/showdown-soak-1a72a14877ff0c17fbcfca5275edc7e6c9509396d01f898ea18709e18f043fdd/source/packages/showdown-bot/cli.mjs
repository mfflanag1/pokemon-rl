import {parseArgs} from 'node:util';
import {ShowdownBot} from './client.mjs';

try {
  const {values} = parseArgs({options: {
    url: {type: 'string'}, name: {type: 'string'}, accept: {type: 'string', multiple: true},
    challenge: {type: 'string'}, policy: {type: 'string', default: 'heuristic'},
    help: {type: 'boolean', default: false},
  }});
  if (values.help) {
    console.log('node packages/showdown-bot/cli.mjs --name BaselineBot --accept YourName [--url ws://127.0.0.1:8000/showdown/websocket]');
    console.log('Optional --challenge Opponent, --policy heuristic|random. Remote WSS login reads SHOWDOWN_PASSWORD from the environment. No ladder searches.');
  } else {
    if (!values.accept?.length && !values.challenge) throw new Error('Specify --accept NAME or --challenge NAME');
    const bot = new ShowdownBot({...values, password: process.env.SHOWDOWN_PASSWORD});
    bot.on('failure', error => { console.error(error.message); process.exitCode = 1; });
    bot.on('battleEnd', result => console.log(JSON.stringify({event: 'battle-end', ...result})));
    bot.on('retry', result => console.log(JSON.stringify({event: 'choice-refreshed', ...result})));
    for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => bot.close());
    await bot.connect();
    console.log(`Ready: ${bot.options.name} (${bot.options.policy}); accepting challenges from ${(values.accept ?? []).join(', ') || 'nobody'}`);
    if (values.challenge) bot.challenge(values.challenge);
  }
} catch (error) { console.error(error.message); process.exitCode = 1; }
