"""Line Splitter: a local web app to upload a tablet photograph, split it into line crops with the line model,
and compare the crops with the original.

  MPLBACKEND=Agg .venv/bin/python -m line_cropping.app.server [--weights weights/linemodel/run1/model.pt] [--port 8770]
then open http://localhost:8770

Standard library only (no web framework). Uploads and results are kept per session in tablet_data/app_sessions/<id>/
(gitignored): original.jpg, display.jpg, views/<v>.jpg, crops/<v>__L<nn>.jpg, result.json.
"""
import argparse
import io
import json
import mimetypes
import os
import re
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import cv2
import numpy as np
import torch

from line_cropping.app.pipeline import detect_views, split_view
from line_cropping.linemodel.infer import load_model
from line_cropping.paths import APP_SESSIONS

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, 'static')
SESSIONS = APP_SESSIONS
DISPLAY_MAX = 2000
MAX_UPLOAD = 80 * 1024 * 1024
STATE = {}
GPU_LOCK = threading.Lock()


def session_dir(sid):
    if not re.fullmatch(r'[0-9a-f]{12}', sid or ''):
        raise ValueError('bad session id')
    return os.path.join(SESSIONS, sid)


def handle_upload(body, filename):
    arr = np.frombuffer(body, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('The file could not be read as an image. Upload a JPEG or PNG photograph.')
    sid = uuid.uuid4().hex[:12]
    d = session_dir(sid)
    os.makedirs(os.path.join(d, 'views'), exist_ok=True)
    os.makedirs(os.path.join(d, 'crops'), exist_ok=True)
    H, W = img.shape[:2]
    cv2.imwrite(os.path.join(d, 'original.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ds = min(1.0, DISPLAY_MAX / max(H, W))
    disp = cv2.resize(img, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA) if ds < 1 else img
    cv2.imwrite(os.path.join(d, 'display.jpg'), disp, [cv2.IMWRITE_JPEG_QUALITY, 88])
    views = detect_views(img)
    meta = dict(session=sid, filename=filename, width=W, height=H, display_scale=ds, views=views, created=time.time())
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'))
    return dict(session=sid, filename=filename, image=dict(url=f'/sessions/{sid}/display.jpg', width=int(disp.shape[1]), height=int(disp.shape[0])),
                original=dict(width=W, height=H), views=[dict(id=v['id'], box=[round(c * ds, 1) for c in v['box']]) for v in views])


def handle_split(req):
    sid = req.get('session')
    d = session_dir(sid)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    img = cv2.imread(os.path.join(d, 'original.jpg'))
    ds = meta['display_scale']
    if req.get('whole'):
        chosen = [dict(id='whole', box=[0, 0, meta['width'], meta['height']])]
    else:
        ids = set(req.get('views') or [])
        chosen = [v for v in meta['views'] if v['id'] in ids]
    if not chosen:
        raise ValueError('Select at least one view, or choose "whole image".')
    readout = dict(STATE['readout'])
    if req.get('kernel_thr') is not None:
        readout['kernel_thr'] = float(req['kernel_thr'])
    for f in os.listdir(os.path.join(d, 'crops')):
        os.remove(os.path.join(d, 'crops', f))
    out = []
    t0 = time.time()
    for v in chosen:
        x0, y0, x1, y1 = v['box']
        pad = int(0.02 * max(x1 - x0, y1 - y0))
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        X1, Y1 = min(meta['width'], x1 + pad), min(meta['height'], y1 + pad)
        face = img[Y0:Y1, X0:X1]
        cv2.imwrite(os.path.join(d, 'views', f'{v["id"]}.jpg'), face, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with GPU_LOCK:
            lines, info = split_view(STATE['model'], STATE['dev'], face, os.path.join(d, 'crops'), f'view{v["id"]}',
                                     rotate_per_line=bool(req.get('rotate', True)), readout=readout)
        for ln in lines:
            ln['poly'] = [[round((px + X0) * ds, 1), round((py + Y0) * ds, 1)] for px, py in ln['poly']]
            ln['crop_url'] = f'/sessions/{sid}/crops/{ln.pop("file")}'
        out.append(dict(view=v['id'], box=[round(c * ds, 1) for c in (X0, Y0, X1, Y1)], angle=info['angle'], pitch=info['pitch'], lines=lines))
    res = dict(session=sid, seconds=round(time.time() - t0, 2), readout=readout, views=out, zip_url=f'/api/zip/{sid}')
    json.dump(res, open(os.path.join(d, 'result.json'), 'w'))
    return res


def load_session(sid):
    """Upload info and, if the session was split, its result; lets the page be reopened with ?session=<id>."""
    d = session_dir(sid)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    ds = meta['display_scale']
    disp = cv2.imread(os.path.join(d, 'display.jpg'))
    out = dict(session=sid, filename=meta['filename'], image=dict(url=f'/sessions/{sid}/display.jpg', width=int(disp.shape[1]), height=int(disp.shape[0])),
               original=dict(width=meta['width'], height=meta['height']), views=[dict(id=v['id'], box=[round(c * ds, 1) for c in v['box']]) for v in meta['views']])
    rp = os.path.join(d, 'result.json')
    out['result'] = json.load(open(rp)) if os.path.exists(rp) else None
    return out


def make_zip(sid):
    d = session_dir(sid)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(os.path.join(d, 'crops'))):
            z.write(os.path.join(d, 'crops', f), f'crops/{f}')
        if os.path.exists(os.path.join(d, 'result.json')):
            z.write(os.path.join(d, 'result.json'), 'result.json')
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = 'LineSplitter/1.0'

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype='application/json', extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not os.path.isfile(path):
            return self._send(404, {'error': 'not found'})
        self._send(200, open(path, 'rb').read(), mimetypes.guess_type(path)[0] or 'application/octet-stream')

    def do_GET(self):
        p = unquote(self.path.split('?', 1)[0])
        try:
            if p in ('/', '/index.html'):
                return self._file(os.path.join(STATIC, 'index.html'))
            if p.startswith('/static/'):
                return self._file(os.path.join(STATIC, os.path.basename(p)))
            m = re.fullmatch(r'/sessions/([0-9a-f]{12})/((?:crops|views)/)?([\w.,+-]+\.jpg)', p)
            if m:
                return self._file(os.path.join(session_dir(m.group(1)), m.group(2) or '', m.group(3)))
            m = re.fullmatch(r'/api/zip/([0-9a-f]{12})', p)
            if m:
                return self._send(200, make_zip(m.group(1)), 'application/zip', {'Content-Disposition': f'attachment; filename="line_crops_{m.group(1)}.zip"'})
            m = re.fullmatch(r'/api/session/([0-9a-f]{12})', p)
            if m:
                return self._send(200, load_session(m.group(1)))
            if p == '/api/status':
                return self._send(200, dict(weights=STATE['weights'], device=STATE['dev'], readout=STATE['readout']))
            self._send(404, {'error': 'not found'})
        except ValueError as e:
            self._send(400, {'error': str(e)})

    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length', 0))
            if n <= 0 or n > MAX_UPLOAD:
                return self._send(413, {'error': f'Upload must be between 1 byte and {MAX_UPLOAD // 1024 // 1024} MB.'})
            body = self.rfile.read(n)
            if self.path == '/api/upload':
                return self._send(200, handle_upload(body, unquote(self.headers.get('X-Filename', 'upload'))))
            if self.path == '/api/split':
                return self._send(200, handle_split(json.loads(body)))
            self._send(404, {'error': 'not found'})
        except ValueError as e:
            self._send(400, {'error': str(e)})
        except Exception as e:
            self._send(500, {'error': f'{type(e).__name__}: {e}'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/linemodel/run1/model.pt')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--host', default='127.0.0.1')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfgp = os.path.join(os.path.dirname(a.weights), 'config.json')
    cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    readout = {k: cfg.get(k, d) for k, d in (('kernel_thr', 0.5), ('band_thr', 0.4), ('min_area', 12), ('max_reach', 1.5), ('max_dev', 45.0), ('merge_dy', 0.5), ('end_thr', 0.2), ('max_ext', 3.0), ('clip_tablet', 1))}
    STATE.update(model=load_model(a.weights, dev), dev=dev, weights=a.weights, readout=readout)
    os.makedirs(SESSIONS, exist_ok=True)
    print(f'Line Splitter on http://{a.host}:{a.port}  (model {a.weights}, {dev})', flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
