# AGENTS.md

This file is for coding agents working in this repository. It captures the current repo-specific context that is easy to miss from generic inspection.

## Repo Overview

Workspace root:

- `/Users/ykshin/Dev/me/RustRobotics`

Main crates:

- [`rust_robotics_algo`](./rust_robotics_algo)
  - shared robotics algorithms
  - shared MuJoCo robot-framework logic for Go2 and Open Duck Mini
- [`rust_robotics_sim`](./rust_robotics_sim)
  - UI app
  - MuJoCo world integration
  - native/web runtime glue

Other important paths:

- [`docs`](./docs)
  - built web bundle served locally and synced to the blog repo
- [`build_web.sh`](./build_web.sh)
  - builds the web app and copies the required static assets into `docs/`
- [`start_server.sh`](./start_server.sh)
  - local server launcher with COOP/COEP headers
- [`scripts/serve_docs.py`](./scripts/serve_docs.py)
  - static server used for browser MuJoCo runs

## Current Branch / State

Current working branch when this file was created:

- `feature/mujoco-tab`

Recent relevant MuJoCo commits:

- `2102d4a` `Match web MuJoCo colors to native materials`
- `91df0a7` `Share robot framework across native and web`
- `7d939cd` `Remove dead native MuJoCo glow path`
- `c58062c` `Refactor native MuJoCo runtime and wgpu viewport`

Current uncommitted local changes that should be treated as user-owned unless explicitly requested:

- modified: [`docs/assets/mujoco/openduckmini/open_duck_mini_v2.xml`](./docs/assets/mujoco/openduckmini/open_duck_mini_v2.xml)
- modified: [`docs/assets/mujoco/openduckmini/scene.xml`](./docs/assets/mujoco/openduckmini/scene.xml)
- modified: [`rust_robotics_algo/src/control/mod.rs`](./rust_robotics_algo/src/control/mod.rs)
- untracked: [`MUJOCO_LOG.TXT`](./MUJOCO_LOG.TXT)
- untracked at time of writing: [`rust_robotics_sim/MUJOCO_REMAINING_WORK.md`](./rust_robotics_sim/MUJOCO_REMAINING_WORK.md)

Do not overwrite or revert those files unless the user asks.

## MuJoCo Architecture

Reference docs:

- [`rust_robotics_sim/MUJOCO_ARCHITECTURE.md`](./rust_robotics_sim/MUJOCO_ARCHITECTURE.md)
- [`rust_robotics_sim/MUJOCO_REMAINING_WORK.md`](./rust_robotics_sim/MUJOCO_REMAINING_WORK.md)

Current architecture boundary:

- `rust_robotics_sim` owns the world
  - MuJoCo model/data
  - stepping/reset
  - raw-state extraction
  - actuation application
  - native/web rendering and UI glue
- `rust_robotics_algo` owns the robot framework
  - command semantics
  - observation building
  - recurrent/action state
  - action smoothing
  - actuation decoding
  - shared Go2 and Duck controller logic

Important files:

- native MuJoCo world/runtime:
  - [`rust_robotics_sim/src/simulator/mujoco/native.rs`](./rust_robotics_sim/src/simulator/mujoco/native.rs)
  - [`rust_robotics_sim/src/simulator/mujoco/native/sim.rs`](./rust_robotics_sim/src/simulator/mujoco/native/sim.rs)
- web MuJoCo runtime:
  - [`rust_robotics_sim/src/simulator/mujoco/wasm.rs`](./rust_robotics_sim/src/simulator/mujoco/wasm.rs)
  - [`rust_robotics_sim/web/mujoco_runtime.js`](./rust_robotics_sim/web/mujoco_runtime.js)
- shared FW:
  - [`rust_robotics_algo/src/robot_fw/mod.rs`](./rust_robotics_algo/src/robot_fw/mod.rs)
  - [`rust_robotics_algo/src/robot_fw/go2.rs`](./rust_robotics_algo/src/robot_fw/go2.rs)
  - [`rust_robotics_algo/src/robot_fw/duck.rs`](./rust_robotics_algo/src/robot_fw/duck.rs)
  - [`rust_robotics_algo/src/robot_fw/onnx.rs`](./rust_robotics_algo/src/robot_fw/onnx.rs)

## Native Runtime Notes

- Native app shell now runs on `wgpu`, not the old `glow` path.
- Native MuJoCo viewport is a `wgpu` callback renderer.
- The old native `glow` readback path has been removed from `native.rs`.
- Native mesh rendering now caches mesh assets and draws per-instance transforms instead of rebuilding mesh triangles every frame.

Expected native sanity checks:

- `cargo check -p rust_robotics_algo`
- `cargo check -p rust_robotics_sim`
- `cargo run -p rust_robotics_sim --bin rust_robotics_sim_bin --release`

## Web Runtime Notes

- Web app shell uses `wgpu`.
- Browser MuJoCo viewport is a separate JS-owned overlay canvas.
- Browser MuJoCo and ORT are currently non-threaded / GitHub-Pages-compatible.
- Web uses a shared FW bridge exported from the wasm bundle:
  - `rust_robotics_fw_create_controller`
  - `rust_robotics_fw_prepare_step`
  - `rust_robotics_fw_finish_step`
  - `rust_robotics_fw_destroy_controller`
- `docs/index.html` explicitly sets `globalThis.wasm_bindgen = wasm_bindgen` so `mujoco_runtime.js` can use the exported FW bridge.

Expected web sanity checks:

- `cargo check -p rust_robotics_sim --target wasm32-unknown-unknown --no-default-features`
- `node --check rust_robotics_sim/web/mujoco_runtime.js`
- `./build_web.sh --fast`

Local web URL typically used:

- `http://127.0.0.1:3000/`

The local server must provide:

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Resource-Policy: same-origin`

## Current Functional Baseline

Browser MuJoCo:

- Go2 and Open Duck Mini both run
- shared robot FW is used for both native and web
- Go2 uses the `facet` setpoint-ball control path
- Open Duck Mini uses the ONNX walk policy
- web colors now resolve `mat_rgba` before `geom_rgba`, matching native

Native MuJoCo:

- Go2 and Open Duck Mini both initialize
- setpoint marker is 3D
- viewport performance is acceptable after mesh-instance caching

## Deployment Notes

Built web output lives in [`docs`](./docs).

Blog repo sync target used in previous work:

- `~/Dev/me/blog/yongkyuns.github.io/sim`

Important caveat:

- the public/static hosting path currently relies on the non-threaded browser MuJoCo setup
- if browser MuJoCo starts hanging on the hosted copy, check whether the deployed bundle is stale before changing runtime code

## What To Avoid

- Do not reintroduce the old native `glow` viewport path.
- Do not assume web and native share rendering code; they currently do not.
- Do not move MuJoCo API ownership into `rust_robotics_algo`; that crate is the robot FW layer, not the world layer.
- Do not revert user-owned edits in `control/mod.rs` or the local Duck XML copies.

## Recommended Next Work

If continuing the MuJoCo architecture cleanup, the next highest-value task is:

1. remove stale JS-side controller logic from [`rust_robotics_sim/web/mujoco_runtime.js`](./rust_robotics_sim/web/mujoco_runtime.js)
2. leave JS responsible only for:
   - MuJoCo state extraction
   - ORT invocation
   - overlay rendering and pointer handling
   - shared FW bridge calls
3. add focused tests in `rust_robotics_algo` for Go2 and Duck observation/actuation semantics
