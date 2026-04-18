#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${script_dir}/.." && pwd -P)

pages_repo_default="${HOME}/Dev/me/blog/yongkyuns.github.io"
pages_repo="${PAGES_REPO:-$pages_repo_default}"
sim_dir="${SIM_DIR:-sim}"
tutorial_dir="${TUTORIAL_DIR:-sim-tutorial}"
build=false
dry_run=false

usage() {
  cat <<EOF
publish_pages.sh [options]

Copy the built simulator bundle and tutorial site into a GitHub Pages repo.

Defaults:
  pages repo:    ${pages_repo_default}
  simulator dir: sim
  tutorial dir:  sim-tutorial

Options:
  --repo PATH            Override GitHub Pages repo path
  --sim-dir NAME         Target subdirectory for the full simulator bundle
  --tutorial-dir NAME    Target subdirectory for the tutorial site
  --build                Rebuild simulator bundle and tutorial site first
  --dry-run              Show rsync actions without copying
  -h, --help             Show this help

Environment overrides:
  PAGES_REPO, SIM_DIR, TUTORIAL_DIR

Examples:
  ./scripts/publish_pages.sh --build
  ./scripts/publish_pages.sh --repo ~/Dev/me/blog/yongkyuns.github.io --tutorial-dir sim-docs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      pages_repo="$2"
      shift 2
      ;;
    --sim-dir)
      sim_dir="$2"
      shift 2
      ;;
    --tutorial-dir)
      tutorial_dir="$2"
      shift 2
      ;;
    --build)
      build=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${pages_repo}" ]]; then
  echo "Pages repo not found: ${pages_repo}" >&2
  exit 1
fi

if [[ "${sim_dir}" == "${tutorial_dir}" ]]; then
  echo "Simulator and tutorial targets must be different directories." >&2
  exit 1
fi

cd "${repo_root}"

if [[ "${build}" == true ]]; then
  ./build_web.sh --fast
  "${script_dir}/build_docs_site.sh"
fi

if [[ ! -f "docs/index.html" ]]; then
  echo "Missing built simulator bundle in docs/. Run ./build_web.sh first." >&2
  exit 1
fi

if [[ ! -f "site_docs/_build/html/index.html" ]]; then
  echo "Missing built tutorial site in site_docs/_build/html. Run ./scripts/build_docs_site.sh first." >&2
  exit 1
fi

sim_target="${pages_repo}/${sim_dir}"
tutorial_target="${pages_repo}/${tutorial_dir}"

mkdir -p "${sim_target}" "${tutorial_target}"

rsync_args=(-a --delete)
if [[ "${dry_run}" == true ]]; then
  rsync_args+=(--dry-run --itemize-changes)
fi

echo "Publishing simulator bundle:"
echo "  from: ${repo_root}/docs/"
echo "  to:   ${sim_target}/"
rsync "${rsync_args[@]}" "docs/" "${sim_target}/"

echo
echo "Publishing tutorial site:"
echo "  from: ${repo_root}/site_docs/_build/html/"
echo "  to:   ${tutorial_target}/"
rsync "${rsync_args[@]}" "site_docs/_build/html/" "${tutorial_target}/"

echo
echo "Done."
echo "Suggested URLs after Pages deploy:"
echo "  Simulator: /${sim_dir}/"
echo "  Tutorial:  /${tutorial_dir}/"
