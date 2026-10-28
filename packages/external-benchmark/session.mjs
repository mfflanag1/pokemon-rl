import {StructuredSession} from '../structured-policy/session.mjs';

/** F34 additive seam: only this seat's already split player protocol leaves
 * the coordinator for the external process. Never serialize Battle or end data. */
export class ExternalBenchmarkSession extends StructuredSession {
  constructor(externalSide) {
    super();
    if (!['p1', 'p2'].includes(externalSide)) throw new Error('External side must be p1 or p2');
    this.externalSide = externalSide;
    this.externalCursor = 0;
  }

  externalInput() {
    const side = this.externalSide, player = this.projection.players[side];
    const protocol = player.protocol.slice(this.externalCursor);
    for (const line of protocol) {
      if (line.startsWith('|split|')) throw new Error('Unsplit protocol at external boundary');
      if (line.startsWith('|request|')) {
        const request = JSON.parse(line.slice(9));
        if (request?.side && request.side.id !== side) throw new Error('Foreign private request');
      }
    }
    this.externalCursor = player.protocol.length;
    return {schema_version: 'f34-external-player-v1', side,
      request_id: player.request_id, pending: Boolean(player.pending && !this.outcome), protocol};
  }

  bundle(frame) { return {frame, external: this.externalInput()}; }
}
