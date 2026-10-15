"""Task-scoped internal-LAN transfer: one payload GET and one bounded result PUT."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import argparse
import time
import shutil
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--max-return-bytes',type=int,default=16000000)
parser.add_argument('--max-seconds',type=float,default=86400)
args=parser.parse_args()
root=args.root.resolve()
if not 0<args.max_return_bytes<=512*1024*1024:raise ValueError('result limit outside bounded transport capacity')
if not 0<args.max_seconds<=86400:raise ValueError('bounded transfer lifetime required')


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path!='/board_payload.tar.gz':
            self.send_error(404);return
        path=root/'board_payload.tar.gz'
        self.send_response(200);self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
        with path.open('rb') as f:shutil.copyfileobj(f,self.wfile)
    def do_PUT(self):
        if self.path!='/board_return.tar.gz' or (root/'board_return.tar.gz').exists():
            self.send_error(409);return
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=args.max_return_bytes:
            self.send_error(413);return
        data=self.rfile.read(length)
        if len(data)!=length:self.send_error(400);return
        (root/'board_return.tmp').write_bytes(data)
        (root/'board_return.tmp').rename(root/'board_return.tar.gz')
        self.send_response(201);self.end_headers()


server=HTTPServer(('192.168.31.1',18773),Handler);server.timeout=1
deadline=time.monotonic()+args.max_seconds
try:
    while time.monotonic()<deadline:server.handle_request()
finally:server.server_close()
