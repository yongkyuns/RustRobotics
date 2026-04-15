#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path


class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".mjs"):
            return "text/javascript"
        return super().guess_type(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve RustRobotics docs/ with COOP/COEP headers.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--root", default="docs")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    handler = partial(CrossOriginIsolatedHandler, directory=str(root))

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving {root} at http://{args.host}:{args.port}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
