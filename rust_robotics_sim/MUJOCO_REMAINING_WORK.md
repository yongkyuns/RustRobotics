# MuJoCo Remaining Work

This note captures the remaining MuJoCo / robot-framework work after the current native and web refactors.

Related architecture note:

- [`MUJOCO_ARCHITECTURE.md`](./MUJOCO_ARCHITECTURE.md)

## Current State

What is already true:

- native app shell uses `wgpu`
- native MuJoCo viewport uses the current `wgpu` renderer path
- native robot/controller logic is no longer hard-wired into the MuJoCo world runtime
- shared robot FW lives in `rust_robotics_algo`
- web MuJoCo uses a JS runtime plus a separate overlay canvas
- web and native now share the Go2 and Duck controller logic through the wasm FW bridge
- Open Duck Mini and Go2 both run in browser and native

Key files:

- native world/runtime:
  - [`src/simulator/mujoco/native.rs`](./src/simulator/mujoco/native.rs)
  - [`src/simulator/mujoco/native/sim.rs`](./src/simulator/mujoco/native/sim.rs)
- shared FW:
  - [`../rust_robotics_algo/src/mujoco/mod.rs`](../rust_robotics_algo/src/mujoco/mod.rs)
  - [`../rust_robotics_algo/src/mujoco/go2.rs`](../rust_robotics_algo/src/mujoco/go2.rs)
  - [`../rust_robotics_algo/src/mujoco/duck.rs`](../rust_robotics_algo/src/mujoco/duck.rs)
- web runtime:
  - [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)
  - [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)
  - [`src/mujoco_fw_web.rs`](./src/mujoco_fw_web.rs)

## Highest-Priority Remaining Work

### 1. Remove stale JS-side controller code

The browser runtime now routes controller semantics through the shared FW bridge, but `web/mujoco_runtime.js` still contains old controller logic that is no longer authoritative.

Examples:

- old Go2 history / command builders
- old Duck observation builder
- old action smoothing / motor target logic

These should be deleted so the web runtime keeps only:

- MuJoCo state extraction
- ORT invocation
- viewport / input handling
- shared FW bridge calls

Primary file:

- [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)

### 2. Make the ONNX abstraction explicit

The shared FW now has an `InferenceBackend` trait, but the web path still uses a prepare/finish bridge around JS ORT calls rather than a platform adapter that clearly implements the same abstraction.

What should exist:

- native ONNX adapter
- wasm/web ONNX adapter

Both should be thin and separate from the robot FW itself.

Primary files:

- [`../rust_robotics_algo/src/mujoco/mod.rs`](../rust_robotics_algo/src/mujoco/mod.rs)
- [`src/simulator/mujoco/native.rs`](./src/simulator/mujoco/native.rs)
- [`src/mujoco_fw_web.rs`](./src/mujoco_fw_web.rs)

### 3. Reduce web Rust/JS per-step shape further

The web path is much better than before, but it still crosses the Rust/JS boundary twice per controller step:

1. prepare observation
2. finish with policy output

This is acceptable for now, but the next architectural improvement is to make the data packets smaller and more explicit.

Good direction:

- fixed typed-array payloads instead of large generic JS objects
- fewer allocations during per-step state marshaling

Primary files:

- [`src/mujoco_fw_web.rs`](./src/mujoco_fw_web.rs)
- [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)

## Medium-Priority Work

### 4. Unify native and web MuJoCo panel semantics

The two paths are closer now, but they are still not fully aligned in UI details and status reporting.

Remaining cleanup:

- same panel terminology
- same telemetry surface where practical
- same robot-selection workflow

Primary files:

- [`src/simulator/mujoco/native.rs`](./src/simulator/mujoco/native.rs)
- [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)

### 5. Add targeted golden tests for shared FW behavior

The shared controller logic is now important enough that it should be protected from native/web drift.

Recommended tests:

- Go2 observation shape and ordering
- Go2 command-mode semantics
- Go2 output integration and torque decode
- Duck observation shape
- Duck output integration and position-target decode

Best location:

- `../rust_robotics_algo/tests/`

### 6. Clean remaining dead code and warnings

There is still non-MuJoCo warning/dead-code noise, especially in the web MuJoCo backend and path-planning module.

This is not functionally urgent, but it is worth cleaning once the runtime shape stabilizes.

Primary files:

- [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)
- [`src/simulator/path_planning.rs`](./src/simulator/path_planning.rs)

## Lower-Priority / Optional Work

### 7. Move the browser step loop fully into JS

The current web flow is still orchestrated from Rust UI cadence. The next performance/architecture step is:

- JS owns continuous stepping
- Rust sends command / viewport state
- Rust only reads telemetry

That would simplify `wasm.rs` further.

Primary files:

- [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)
- [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)

### 8. Push more config into robot data files

Some behavior still depends on code-side assumptions that could live in policy/config metadata instead.

Good candidates:

- controller kind
- command mode
- command dimensions
- joint/default pose assumptions
- actuator semantics

Primary files:

- robot asset JSONs under [`assets/mujoco`](./assets/mujoco)
- web/native config readers

### 9. Revisit deployment docs

Now that the browser path is GitHub-Pages-compatible and uses the shared FW bridge, deployment docs should reflect:

- non-threaded browser MuJoCo assumption
- current ORT setup
- local server expectations
- blog-site sync flow

Good locations:

- [`MUJOCO_ARCHITECTURE.md`](./MUJOCO_ARCHITECTURE.md)
- top-level project docs if needed

## Recommended Next Step

If only one thing is done next, it should be:

1. remove the stale JS-side controller implementation from [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)
2. keep only state extraction, ORT, overlay rendering, and FW bridge calls
3. then add focused tests in `rust_robotics_algo`

That would leave the architecture in a much cleaner state without changing the working runtime model again.
