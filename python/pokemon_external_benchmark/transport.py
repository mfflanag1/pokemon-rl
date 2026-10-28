"""Bounded JSON-lines transport for the two new offline worker protocols."""
from __future__ import annotations
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


class JsonWorker:
    def __init__(self, command, stderr_path, timeout=60):
        self.sequence=0;self.buffer=b'';self.timeout=timeout;self.closed=False
        path=Path(stderr_path);path.parent.mkdir(parents=True,exist_ok=True)
        self.stderr=path.open('xb')
        try:
            env={**os.environ,'PYTHONPATH':str(ROOT/'python')}
            self.process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0,start_new_session=True)
            os.set_blocking(self.process.stdin.fileno(),False)
            os.set_blocking(self.process.stdout.fileno(),False)
        except BaseException:
            self.stderr.close();raise

    def request(self,op,**payload):
        if self.closed:raise RuntimeError('Closed external worker')
        self.sequence+=1;deadline=time.monotonic()+self.timeout
        raw=(json.dumps({'id':self.sequence,'op':op,**payload},allow_nan=False)+'\n').encode()
        written=0
        try:
            while written<len(raw):
                remaining=deadline-time.monotonic()
                if remaining<=0 or not select.select([],[self.process.stdin],[],remaining)[1]:
                    raise TimeoutError('External worker write timed out')
                try:n=os.write(self.process.stdin.fileno(),raw[written:])
                except BlockingIOError:continue
                if n==0:raise RuntimeError('External worker pipe closed')
                written+=n
            while b'\n' not in self.buffer:
                remaining=deadline-time.monotonic()
                if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                    raise TimeoutError('External worker read timed out')
                try:chunk=os.read(self.process.stdout.fileno(),65536)
                except BlockingIOError:continue
                if not chunk:raise RuntimeError('External worker exited '+str(self.process.poll()))
                self.buffer+=chunk
                if len(self.buffer)>4_000_000:raise ValueError('External worker response size')
            line,self.buffer=self.buffer.split(b'\n',1);response=json.loads(line)
            if response.get('id')!=self.sequence or response.get('ok') is not True:
                raise RuntimeError('External worker rejected command: '+json.dumps(response))
            return response['data']
        except BaseException:
            self.close();raise

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                self.process.wait(timeout=3)
        finally:
            self.process.stdin.close();self.process.stdout.close();self.stderr.close()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
