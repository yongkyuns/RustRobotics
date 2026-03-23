#!/usr/bin/env bash
set -eu
script_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$script_path"

OPEN=false
FAST=false

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "build_web_glow.sh [--fast] [--open]"
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

FOLDER_NAME=${PWD##*/}
CRATE_NAME=rust_robotics_sim
CRATE_NAME_SNAKE_CASE="${CRATE_NAME//-/_}"

export RUSTFLAGS=--cfg=web_sys_unstable_apis

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
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm.wasm" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-threaded.wasm" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm-threaded.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd.wasm" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm-simd.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd.jsep.wasm" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm-simd.jsep.wasm"
cp "rust_robotics_sim/web/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm" \
  "docs/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm"
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

echo "Building experimental Rust-owned MuJoCo web viewport bundle…"
BUILD=release
cargo build -p "${CRATE_NAME}" --release --lib --target wasm32-unknown-unknown

TARGET=$(cargo metadata --format-version=1 | jq --raw-output .target_directory)

echo "Generating JS bindings for wasm…"
TARGET_NAME="${CRATE_NAME_SNAKE_CASE}.wasm"
WASM_PATH="${TARGET}/wasm32-unknown-unknown/${BUILD}/${TARGET_NAME}"
wasm-bindgen "${WASM_PATH}" --out-dir docs --target web --no-typescript

if [[ "${FAST}" == false ]]; then
  echo "Optimizing wasm…"
  wasm-opt "docs/${CRATE_NAME}_bg.wasm" -O2 --fast-math -o "docs/${CRATE_NAME}_bg.wasm"
fi

echo "Finished experimental bundle: docs/${CRATE_NAME_SNAKE_CASE}.wasm"

if [[ "${OPEN}" == true ]]; then
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8080/index.html
  elif [[ "$OSTYPE" == "msys" ]]; then
    start http://localhost:8080/index.html
  else
    open http://localhost:8080/index.html
  fi
fi
