# MuJoCo Integration Architecture

This note describes the current MuJoCo and ONNX integration used by `rust_robotics_sim`.

## Overview

`MujocoPanel` is a platform facade with two backends:

- native backend: [`src/simulator/mujoco/native.rs`](./src/simulator/mujoco/native.rs)
- web backend: [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)

The facade itself is in [`src/simulator/mujoco/mod.rs`](./src/simulator/mujoco/mod.rs). The rest of the simulator talks only to `MujocoPanel`.

The simulator activates the MuJoCo backend only when the current mode is `SimMode::Mujoco`:

- [`src/simulator/mod.rs`](./src/simulator/mod.rs)
- [`src/app.rs`](./src/app.rs)

## Native Backend

The native path is mostly Rust-owned.

### Ownership

Rust owns:

- MuJoCo C objects:
  - `mjModel`
  - `mjData`
  - `mjvScene`
  - camera/render state
- ONNX Runtime session via the `ort` crate
- controller state:
  - recurrent state
  - action history
  - command state

Main runtime struct:

- [`MujocoRuntime`](./src/simulator/mujoco/native.rs)

### Load Path

Native startup loads:

1. MuJoCo XML scene from disk
2. `asset_meta.json`
3. policy config JSON
4. ONNX model through native ONNX Runtime

Relevant code:

- scene/model load: [`native.rs`](./src/simulator/mujoco/native.rs)
- ORT init and session creation: [`native.rs`](./src/simulator/mujoco/native.rs)

### Step Loop

Each app frame, when the tab is active and not paused:

1. Rust builds the policy input from current MuJoCo state
2. Rust runs ONNX inference
3. Rust smooths actions and updates recurrent state
4. Rust applies PD torques
5. Rust advances MuJoCo with `mj_step`

Relevant code:

- update entry: [`NativeMujocoBackend::update`](./src/simulator/mujoco/native.rs)
- runtime stepping: [`MujocoRuntime::step`](./src/simulator/mujoco/native.rs)
- policy input update: [`MujocoRuntime::update_policy_input`](./src/simulator/mujoco/native.rs)
- ONNX inference: [`MujocoRuntime::run_policy`](./src/simulator/mujoco/native.rs)
- control application: [`MujocoRuntime::apply_control`](./src/simulator/mujoco/native.rs)

### Threading

There is no explicit Rust-side worker split for native MuJoCo. The stepping path is synchronous from the app's point of view and runs in the normal app update flow.

## Web Backend

The web path is intentionally split between Rust and JavaScript.

### Rust Responsibilities

Rust owns:

- the RustRobotics app shell
- MuJoCo panel UI
- robot preset selection
- asset fetching
- async orchestration for browser runtime startup and stepping
- status/telemetry display
- viewport rectangle reporting

Main Rust backend:

- [`WasmMujocoBackend`](./src/simulator/mujoco/wasm.rs)

### JavaScript Responsibilities

JavaScript owns:

- MuJoCo wasm runtime
- live MuJoCo stepping
- browser-side render state
- separate browser MuJoCo overlay canvas and camera interaction
- raw-state extraction from MuJoCo
- compact per-step payload assembly for the FW bridge

Main JS runtime:

- [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)

### Shared FW Responsibilities on Web

The web path now shares the same FW ownership boundary as native:

- controller state and semantics
- observation building
- recurrent/action integration
- actuation decode
- ONNX ownership

Current web ONNX implementation detail:

- the browser FW path initializes `ort-web` for runtime loading
- per-session creation and inference currently call `onnxruntime-web` directly through Rust-side JS interop
- this avoids a broken `ort-web` session metadata wrapper path while keeping ownership in FW

Main FW implementation:

- [`../rust_robotics_algo/src/robot_fw/onnx.rs`](../rust_robotics_algo/src/robot_fw/onnx.rs)

### Browser Bridge

`index.html` exposes a small set of global bridge functions:

- `rustRoboticsOrtSmokeTest`
- `rustRoboticsMujocoInit`
- `rustRoboticsMujocoStep`
- `rustRoboticsMujocoConfigureViewport`

Relevant code:

- [`../docs/index.html`](../docs/index.html)

Rust calls these globals from `wasm.rs` using `wasm_bindgen`, `js_sys::Function`, and `Promise` conversion through `JsFuture`.

### Asset Flow

Web assets are fetched by Rust from static files. Rust prepares:

- scene files
- robot XML
- policy config JSON
- ONNX model bytes
- mesh assets

Those bytes are then passed into JS during `rustRoboticsMujocoInit(...)`, and the policy bytes are forwarded into the FW runtime creation step.

Relevant code:

- asset presets and file lists: [`wasm.rs`](./src/simulator/mujoco/wasm.rs)
- browser runtime init bridge: [`wasm.rs`](./src/simulator/mujoco/wasm.rs)

### Browser Runtime Init

JS runtime creation currently does:

1. load MuJoCo wasm module
2. configure browser ORT runtime assets
4. write fetched files into MuJoCo's virtual filesystem
5. load model from `/working/scene.xml`
6. create `State` and `Simulation`
7. resolve joint, actuator, and sensor ids
8. create the shared FW runtime and ONNX session
9. build initial render data and report

Relevant code:

- [`createRustRoboticsMujocoRuntime`](./web/mujoco_runtime.js)

### Browser Step Flow

Rust side:

1. `WasmMujocoBackend::update(...)` runs every active frame
2. if assets and runtime are ready, it starts at most one async step
3. result comes back as a summarized report

JS side:

1. update setpoint-driven command
2. extract MuJoCo state into compact typed-array payloads
3. call the FW step runtime
4. FW builds observations, runs ONNX, integrates outputs, and returns decoded actuation
5. JS applies control
6. step MuJoCo for `decimation` substeps
7. update timing counters
8. request overlay render
9. return telemetry report to Rust

Relevant code:

- Rust step orchestration: [`wasm.rs`](./src/simulator/mujoco/wasm.rs)
- JS stepping: [`rustRoboticsMujocoStep`](./web/mujoco_runtime.js)

### Rendering on Web

The current shipped web build still uses the JS-owned overlay viewport path.

Current shipped design:

- main RustRobotics app still runs on the web `wgpu` path
- MuJoCo viewport is rendered by the Rust-owned web `wgpu` path by default
- JS still provides browser MuJoCo module loading and runtime bridging

Migration path:

- the Rust-owned `web_wgpu_viewport` path is the default crate feature
- the older JS overlay path remains available only as a legacy fallback with `--no-default-features`
- a shared Rust scene-building layer now feeds both native and web Rust renderers

### Threading on Web

Current browser path is single-thread oriented:

- MuJoCo uses a browser-compatible non-threaded wasm build
- ORT is configured for `numThreads = 1`
- Rust uses async `spawn_local`, but that is still main-thread browser scheduling
- there is no worker-based simulation split yet

So:

- Rust UI is async, not worker-isolated
- JS stepping is async, but still effectively main-thread work
- browser runtime bridging is JS-side, but not on a dedicated worker

## Current Rust/JS Boundary

### Rust to JS

Rust sends:

- fetched file entries
- ONNX bytes
- controller config
- per-step state and command payloads
- viewport layout config

### JS to Rust

JS returns:

- status/telemetry reports
- initial mesh asset snapshot used by Rust-side bookkeeping

The current fast path avoids shipping full geom snapshots through Rust on every step.

Per-step FW bridge shape on web is now flatter than before:

- fixed core state packet
- dynamic joint vectors
- compact sensor packet for Duck
- compact command packet

The remaining hot-path inefficiency is that full `qpos` / `qvel` are still shipped through the bridge.

## Robot Presets

The web path currently supports:

- Go2 with the `facet` policy
- Open Duck Mini with `BEST_WALK_ONNX.onnx`

Preset selection and asset grouping live in:

- [`wasm.rs`](./src/simulator/mujoco/wasm.rs)

Controller behavior selection in JS is driven by `controller_kind` and ONNX metadata:

- [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)

## Known Architectural Weak Point

The main remaining web limitation is scheduling, with a smaller remaining payload-shape concern:

- Rust still initiates MuJoCo step batches from the app update loop
- JS runs each step batch and returns a report
- the next batch only starts after Rust re-enters that path
- the web bridge is flatter now, but still sends more state than the FW strictly needs in some cases

So the browser path is fast enough now, but it is still not a fully JS-owned simulation loop.

The next major architecture improvement would be:

- move the live MuJoCo stepping loop fully into JS
- let Rust consume telemetry and control state instead of initiating each batch
