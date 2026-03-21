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
      echo "  --fast: skip optimization step"
      echo "  --open: open the result in a browser"
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
      break
      ;;
  esac
done

# ./setup_web.sh # <- call this first!

FOLDER_NAME=${PWD##*/}
# CRATE_NAME=$FOLDER_NAME # assume crate name is the same as the folder name
CRATE_NAME=rust_robotics_sim # assume crate name is the same as the folder name
CRATE_NAME_SNAKE_CASE="${CRATE_NAME//-/_}" # for those who name crates with-kebab-case

# This is required to enable the web_sys clipboard API which egui_web uses
# https://rustwasm.github.io/wasm-bindgen/api/web_sys/struct.Clipboard.html
# https://rustwasm.github.io/docs/wasm-bindgen/web-sys/unstable-apis.html
export RUSTFLAGS=--cfg=web_sys_unstable_apis

# Clear output from old stuff:
rm -f "docs/${CRATE_NAME_SNAKE_CASE}_bg.wasm"
mkdir -p "docs/vendor/onnxruntime-web/dist"
mkdir -p "docs/vendor/mujoco"
mkdir -p "docs/vendor/mujoco/mt"
mkdir -p "docs/assets/mujoco"

rm -rf "docs/assets/mujoco/go2"
rm -rf "docs/assets/mujoco/openduckmini"
cp -R "rust_robotics_sim/assets/mujoco/go2" "docs/assets/mujoco/"
cp -R "rust_robotics_sim/assets/mujoco/openduckmini" "docs/assets/mujoco/"

cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort.wasm.min.js" \
  "docs/vendor/onnxruntime-web/dist/ort.wasm.min.js"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"
cp "rust_robotics_sim/web/vendor/mujoco/mujoco.js" \
  "docs/vendor/mujoco/mujoco.js"
cp "rust_robotics_sim/web/vendor/mujoco/mujoco.wasm" \
  "docs/vendor/mujoco/mujoco.wasm"
cp "rust_robotics_sim/web/vendor/mujoco/mt/mujoco.js" \
  "docs/vendor/mujoco/mt/mujoco.js"
cp "rust_robotics_sim/web/vendor/mujoco/mt/mujoco.wasm" \
  "docs/vendor/mujoco/mt/mujoco.wasm"
cp "rust_robotics_sim/web/mujoco_runtime.js" \
  "docs/mujoco_runtime.js"

echo "Building rust…"
BUILD=release
cargo build -p "${CRATE_NAME}" --release --lib --target wasm32-unknown-unknown --no-default-features

# Get the output directory (in the workspace it is in another location)
TARGET=$(cargo metadata --format-version=1 | jq --raw-output .target_directory)

echo "Generating JS bindings for wasm…"
TARGET_NAME="${CRATE_NAME_SNAKE_CASE}.wasm"
WASM_PATH="${TARGET}/wasm32-unknown-unknown/${BUILD}/${TARGET_NAME}"
wasm-bindgen "${WASM_PATH}" --out-dir docs --no-modules --no-typescript

if [[ "${FAST}" == false ]]; then
  echo "Optimizing wasm…"
  # to get wasm-opt:  apt/brew/dnf install binaryen
  wasm-opt "docs/${CRATE_NAME}_bg.wasm" -O2 --fast-math -o "docs/${CRATE_NAME}_bg.wasm" # add -g to get debug symbols
fi

echo "Finished: docs/${CRATE_NAME_SNAKE_CASE}.wasm"

if [[ "${OPEN}" == true ]]; then
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux, ex: Fedora
    xdg-open http://localhost:8080/index.html
  elif [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    start http://localhost:8080/index.html
  else
    # Darwin/MacOS, or something else
    open http://localhost:8080/index.html
  fi
fi
