"""Unchanged F4 collector, with observational persistence of rejected feedback."""
import json
from pathlib import Path
from pokemon_full_learning.collector import Pool as OriginalPool


class Pool(OriginalPool):
    def __init__(self,evidence,workers):
        self.feedback_logs=[];self.feedback_count=0
        super().__init__(evidence,workers)
        try:
            for i,worker in enumerate(self.workers):
                log=open(Path(evidence)/f'feedback-{i}.jsonl','x');self.feedback_logs.append(log)
                state={'last':None,'start':None};original=worker.request
                def request(op,_original=original,_state=state,_log=log,**payload):
                    before=_state['last'];frame=_original(op,**payload)
                    if op=='start':_state['start']=payload
                    for submitted in frame.get('submitted',[]):
                        if not submitted['accepted']:
                            _log.write(json.dumps({'start':_state['start'],'command':{'op':op,**payload},
                                'submitted':submitted,'before':before,'after':frame},allow_nan=False)+'\n');_log.flush()
                    _state['last']=frame;return frame
                worker.request=request
        except BaseException:
            self.close();raise

    def close(self):
        try:super().close()
        finally:
            for log in self.feedback_logs:
                if not log.closed:log.close()
