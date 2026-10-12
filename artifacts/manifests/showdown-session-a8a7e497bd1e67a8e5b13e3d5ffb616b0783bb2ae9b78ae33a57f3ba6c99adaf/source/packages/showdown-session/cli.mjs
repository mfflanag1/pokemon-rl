import {parseArgs} from 'node:util';
import {RecoveringBot} from './client.mjs';

try {
  const {values} = parseArgs({options: {
    url: {type: 'string'}, name: {type: 'string', default: 'BaselineBot'},
    accept: {type: 'string', multiple: true}, challenge: {type: 'string'}, help: {type: 'boolean'},
  }});
  if (values.help) {
    console.log('npm run bot:recover -- --name BaselineBot --accept YourName [--url ws://127.0.0.1:8000/showdown/websocket]');
    console.log('Optional --challenge NAME. Heuristic policy; at most three reconnects. Remote login reads SHOWDOWN_PASSWORD on each connection. No ladder searches.');
  } else {
    if (!values.accept?.length && !values.challenge) throw new Error('Specify --accept NAME or --challenge NAME');
    const bot = new RecoveringBot({...values, passwordProvider: () => process.env.SHOWDOWN_PASSWORD});
    bot.on('failure', error => { console.error(error.message); process.exitCode = 1; });
    for (const event of ['ready', 'reconnecting', 'restored', 'battleEnd', 'retry']) {
      bot.on(event, detail => console.log(JSON.stringify({event, ...detail})));
    }
    for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => bot.close());
    await bot.connect();
    if (values.challenge) bot.challenge(values.challenge);
  }
} catch (error) { console.error(error.message); process.exitCode = 1; }
