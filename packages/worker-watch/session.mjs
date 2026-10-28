import {OpponentMixSession} from '../opponent-mix/session.mjs';
import {mark} from './diagnostics.mjs';

/** Observe start boundaries without replacing provenance checks or mechanics. */
export class WatchedSession extends OpponentMixSession {
  constructor() {
    super();
    this.starting = false;
    const write = this.stream.write.bind(this.stream);
    this.stream.write = async (...args) => {
      if (this.starting) mark('stream_write_begin');
      const result = await write(...args);
      if (this.starting) mark('stream_write_done');
      return result;
    };
  }
  planFor(...args) {
    mark('team_admission_begin');
    const plan = super.planFor(...args);
    mark('team_admission_done');
    return plan;
  }
  async start(options) {
    this.starting = true;
    mark('session_start_begin');
    try {
      const frame = await super.start(options);
      mark('session_start_done');
      return frame;
    } finally { this.starting = false; }
  }
  close() {
    mark('session_close_begin');
    super.close();
    mark('session_close_done');
  }
}
