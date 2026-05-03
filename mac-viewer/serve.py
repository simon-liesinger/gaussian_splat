#!/usr/bin/env python3
"""Serve the gaussian splat viewer and open in browser.

Usage:
    python3 serve.py path/to/model.ply
    python3 serve.py  # then drag-drop a .ply file onto the browser
"""
import http.server
import os
import sys
import threading
import webbrowser

PORT = 8765
DIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        pass  # quiet

if __name__ == '__main__':
    ply_path = sys.argv[1] if len(sys.argv) > 1 else None

    # If a PLY file was given, symlink it into the serve directory
    if ply_path:
        ply_path = os.path.abspath(ply_path)
        link = os.path.join(DIR, '_model.ply')
        if os.path.exists(link):
            os.unlink(link)
        os.symlink(ply_path, link)

    server = http.server.HTTPServer(('', PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f'http://localhost:{PORT}/viewer.html'
    if ply_path:
        url += '?url=_model.ply'
    print(f'Serving at {url}')
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
