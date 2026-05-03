#!/usr/bin/env python3
"""Serve the gaussian splat viewer. Auto-converts .ply to .splat format.

Usage:
    python3 serve.py path/to/model.ply
    python3 serve.py path/to/model.splat
    python3 serve.py  # then drag-drop a file onto the browser
"""
import http.server
import os
import subprocess
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
        pass

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    splat_dest = os.path.join(DIR, 'model.splat')

    if model_path:
        model_path = os.path.abspath(model_path)
        if model_path.endswith('.ply'):
            print(f"Converting {model_path} to .splat...")
            subprocess.run([sys.executable, os.path.join(DIR, 'ply2splat.py'), model_path, splat_dest], check=True)
        elif model_path.endswith('.splat'):
            if os.path.abspath(splat_dest) != model_path:
                os.system(f'cp "{model_path}" "{splat_dest}"')
        print(f"Model ready at {splat_dest}")

    # Kill any existing server on this port
    os.system(f"lsof -i :{PORT} -t 2>/dev/null | xargs kill 2>/dev/null")

    import time; time.sleep(0.5)
    server = http.server.HTTPServer(('', PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f'http://localhost:{PORT}/splat_viewer.html'
    if model_path:
        url += '#model.splat'
    print(f'Viewer at {url}')
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
