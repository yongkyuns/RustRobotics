#!/usr/bin/env bash
set -euo pipefail

rm -rf site_docs/_build/html
python3 -m sphinx -E -a -b html site_docs site_docs/_build/html
