"""Tiny local web server for the demo (no internet needed).

    python serve.py          -> http://localhost:8000/

Why not just double-click index.html? Browsers do not let a page opened from the disk (file://)
load the network files (.onnx) and the WebAssembly engine (.wasm). A local server solves that.
The file types are set explicitly, because Windows sometimes reports wrong ones for .mjs / .wasm.
"""
import http.server
import os
import sys
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".jpg": "image/jpeg",
        ".png": "image/png",
    }

    def log_message(self, *args):      # keep the window quiet
        pass


if __name__ == "__main__":
    url = f"http://localhost:{PORT}/"
    print(f"Demo running at {url}  (close this window or press Ctrl + C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
