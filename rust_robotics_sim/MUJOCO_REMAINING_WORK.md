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
- stale JS-side Go2/Duck controller-building logic has been removed from
  [`web/mujoco_runtime.js`](./web/mujoco_runtime.js), leaving JS focused on state extraction,
  overlay handling, and FW-owned step calls
- native and web now share FW-owned ONNX adapter code in
  [`../rust_robotics_algo/src/robot_fw/onnx.rs`](../rust_robotics_algo/src/robot_fw/onnx.rs)

Key files:

- native world/runtime:
  - [`src/simulator/mujoco/native.rs`](./src/simulator/mujoco/native.rs)
  - [`src/simulator/mujoco/native/sim.rs`](./src/simulator/mujoco/native/sim.rs)
- shared FW:
  - [`../rust_robotics_algo/src/robot_fw/mod.rs`](../rust_robotics_algo/src/robot_fw/mod.rs)
  - [`../rust_robotics_algo/src/robot_fw/go2.rs`](../rust_robotics_algo/src/robot_fw/go2.rs)
  - [`../rust_robotics_algo/src/robot_fw/duck.rs`](../rust_robotics_algo/src/robot_fw/duck.rs)
- web runtime:
  - [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)
  - [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)
  - [`../rust_robotics_algo/src/robot_fw/onnx.rs`](../rust_robotics_algo/src/robot_fw/onnx.rs)

## Highest-Priority Remaining Work

### 1. Reduce web Rust/JS per-step shape further

The web path no longer uses the old prepare/finish bridge, but it still marshals large generic JS objects across the boundary on every controller step.

This is acceptable for now, but the next architectural improvement is to make the data packets smaller and more explicit.

Good direction:

- fixed typed-array payloads instead of large generic JS objects
- fewer allocations during per-step state marshaling

Primary files:

- [`../rust_robotics_algo/src/robot_fw/onnx.rs`](../rust_robotics_algo/src/robot_fw/onnx.rs)
- [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)

## Medium-Priority Work

### 2. Unify native and web MuJoCo panel semantics

The two paths are closer now, but they are still not fully aligned in UI details and status reporting.

Remaining cleanup:

- same panel terminology
- same telemetry surface where practical
- same robot-selection workflow

Primary files:

- [`src/simulator/mujoco/native.rs`](./src/simulator/mujoco/native.rs)
- [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)

### 3. Add targeted golden tests for shared FW behavior

The shared controller logic is now important enough that it should be protected from native/web drift.

Recommended tests:

- Go2 observation shape and ordering
- Go2 command-mode semantics
- Go2 output integration and torque decode
- Duck observation shape
- Duck output integration and position-target decode

Best location:

- `../rust_robotics_algo/tests/`

### 4. Clean remaining dead code and warnings

There is still non-MuJoCo warning/dead-code noise, especially in the web MuJoCo backend and path-planning module.

This is not functionally urgent, but it is worth cleaning once the runtime shape stabilizes.

Primary files:

- [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)
- [`src/simulator/path_planning.rs`](./src/simulator/path_planning.rs)

## Lower-Priority / Optional Work

### 5. Move the browser step loop fully into JS

The current web flow is still orchestrated from Rust UI cadence. The next performance/architecture step is:

- JS owns continuous stepping
- Rust sends command / viewport state
- Rust only reads telemetry

That would simplify `wasm.rs` further.

Primary files:

- [`src/simulator/mujoco/wasm.rs`](./src/simulator/mujoco/wasm.rs)
- [`web/mujoco_runtime.js`](./web/mujoco_runtime.js)

### 6. Push more config into robot data files

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

### 7. Revisit deployment docs

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

1. add focused tests in `rust_robotics_algo` for the shared Go2 and Duck FW behavior
2. reduce the web step payloads to fixed typed arrays
3. clean the remaining web-backend warning noise once the interface settles

That would tighten the FW/runtime boundary further without changing the working runtime model again.
