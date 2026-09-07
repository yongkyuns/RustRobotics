#!/usr/bin/env bash
set -eu
script_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$script_path"

OPEN=false
FAST=false

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "build_web.sh [--fast] [--open]"
      echo "  --fast: skip wasm-opt optimization"
      echo "  --open: open the locally served simulator URL"
      exit 0
      ;;
    --fast)
      shift
      FAST=true
      ;;
    --open)
      shift
      OPEN=true
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required tool: $1" >&2
    echo "$2" >&2
    exit 1
  fi
}

require_tool cargo "Install Rust with rustup and add the wasm32-unknown-unknown target."
require_tool jq "Install jq with your platform package manager."
require_tool wasm-bindgen "Install wasm-bindgen-cli at the version used by CI."
if [[ "${FAST}" == false ]]; then
  require_tool wasm-opt "Install Binaryen or rerun with --fast to skip optimization."
fi

CRATE_NAME=rust_robotics_sim
CRATE_NAME_SNAKE_CASE="${CRATE_NAME//-/_}"

export RUSTFLAGS=--cfg=web_sys_unstable_apis

rm -f "docs/${CRATE_NAME_SNAKE_CASE}_bg.wasm"
rm -rf "docs/vendor/onnxruntime-web"
mkdir -p "docs/ort" "docs/vendor/mujoco/mt" "docs/assets/mujoco"

rm -rf "docs/assets/mujoco/go2" "docs/assets/mujoco/openduckmini"
cp -R "rust_robotics_sim/assets/mujoco/go2" "docs/assets/mujoco/"
cp -R "rust_robotics_sim/assets/mujoco/openduckmini" "docs/assets/mujoco/"

cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort.wasm.min.js" \
  "docs/ort/ort.wasm.min.js"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm.wasm" \
  "docs/ort/ort-wasm.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-threaded.wasm" \
  "docs/ort/ort-wasm-threaded.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd.wasm" \
  "docs/ort/ort-wasm-simd.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm" \
  "docs/ort/ort-wasm-simd-threaded.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd.jsep.wasm" \
  "docs/ort/ort-wasm-simd.jsep.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm" \
  "docs/ort/ort-wasm-simd-threaded.jsep.wasm"
cp "rust_robotics_sim/web/vendor/mujoco/mujoco.js" \
  "docs/vendor/mujoco/mujoco.js"
cp "rust_robotics_sim/web/vendor/mujoco/mujoco.wasm" \
  "docs/vendor/mujoco/mujoco.wasm"
cp "rust_robotics_sim/web/vendor/mujoco/mujoco_wasm.js" \
  "docs/vendor/mujoco/mujoco_wasm.js"
cp "rust_robotics_sim/web/vendor/mujoco/mujoco_wasm.wasm" \
  "docs/vendor/mujoco/mujoco_wasm.wasm"
cp "rust_robotics_sim/web/vendor/mujoco/mt/mujoco.js" \
  "docs/vendor/mujoco/mt/mujoco.js"
cp "rust_robotics_sim/web/vendor/mujoco/mt/mujoco.wasm" \
  "docs/vendor/mujoco/mt/mujoco.wasm"
cp "rust_robotics_sim/web/ort_web/_loader.js" \
  "docs/_loader.js"
cp "rust_robotics_sim/web/ort_web/_telemetry.js" \
  "docs/_telemetry.js"
cp "rust_robotics_sim/web/mujoco_runtime.js" \
  "docs/mujoco_runtime.js"
cp "rust_robotics_sim/web/ppo_trainer_bridge.js" \
  "docs/ppo_trainer_bridge.js"
cp "rust_robotics_sim/web/ppo_trainer_worker.js" \
  "docs/ppo_trainer_worker.js"

echo "Building web bundle…"
BUILD=release
cargo build -p "${CRATE_NAME}" --release --lib --target wasm32-unknown-unknown

TARGET=$(cargo metadata --format-version=1 | jq --raw-output .target_directory)

echo "Generating JS bindings for wasm…"
TARGET_NAME="${CRATE_NAME_SNAKE_CASE}.wasm"
WASM_PATH="${TARGET}/wasm32-unknown-unknown/${BUILD}/${TARGET_NAME}"
wasm-bindgen "${WASM_PATH}" --out-dir docs --target web --no-typescript

if [[ -d "docs/snippets" ]]; then
  while IFS= read -r -d '' snippet_dir; do
    if [[ -f "${snippet_dir}/_loader.js" ]]; then
      mv "${snippet_dir}/_loader.js" "${snippet_dir}/loader.js"
    fi
    if [[ -f "${snippet_dir}/_telemetry.js" ]]; then
      mv "${snippet_dir}/_telemetry.js" "${snippet_dir}/telemetry.js"
    fi
  done < <(find "docs/snippets" -mindepth 1 -maxdepth 1 -type d -print0)

  perl -0pi -e "s#/_loader\\.js#\/loader.js#g; s#/_telemetry\\.js#\/telemetry.js#g" \
    "docs/${CRATE_NAME}.js"
fi

if [[ "${FAST}" == false ]]; then
  echo "Optimizing wasm…"
  wasm-opt "docs/${CRATE_NAME}_bg.wasm" -O2 --fast-math -o "docs/${CRATE_NAME}_bg.wasm"
fi

echo "Finished web bundle: docs/${CRATE_NAME_SNAKE_CASE}.wasm"
echo "Serve it with ./start_server.sh and open http://127.0.0.1:3000/"

if [[ "${OPEN}" == true ]]; then
  url="http://127.0.0.1:3000/"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "$url"
  elif [[ "$OSTYPE" == "msys" ]]; then
    start "$url"
  else
    open "$url"
  fi
fi
