#!/usr/bin/env bash
set -eu

# Starts a local web-server that serves the contents of the `docs/` folder
# with cross-origin isolation headers required by pthread-enabled wasm builds.

echo "open http://127.0.0.1:3000"

python3 scripts/serve_docs.py --host 127.0.0.1 --port 3000 --root docs
