#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${script_dir}/.." && pwd -P)
cd "${repo_root}"

venv_dir="${DOCS_VENV:-${repo_root}/.venv-docs}"
python_bin="${PYTHON:-python3}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  echo "Creating documentation environment: ${venv_dir}"
  "${python_bin}" -m venv "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --disable-pip-version-check -r site_docs/requirements.txt

rm -rf site_docs/_build/html
"${venv_dir}/bin/python" -m sphinx -W -E -a -b html site_docs site_docs/_build/html

echo "Built documentation: ${repo_root}/site_docs/_build/html/index.html"
