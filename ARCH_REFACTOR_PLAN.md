# Architecture Refactor Plan

This note describes a concrete refactor direction for improving portability, reuse,
modularity, and readability across the RustRobotics workspace.

It is intentionally repo-specific. It reflects the current state of:

- [`rust_robotics_algo`](./rust_robotics_algo)
- [`rust_robotics_sim`](./rust_robotics_sim)
- [`rust_robotics_train`](./rust_robotics_train)

It does not replace the MuJoCo-specific notes in:

- [`rust_robotics_sim/MUJOCO_ARCHITECTURE.md`](./rust_robotics_sim/MUJOCO_ARCHITECTURE.md)
- [`rust_robotics_sim/MUJOCO_REMAINING_WORK.md`](./rust_robotics_sim/MUJOCO_REMAINING_WORK.md)

## Current Strengths

The current workspace split is already better than a monolith:

- `rust_robotics_algo` owns reusable robotics algorithms and shared robot FW logic
- `rust_robotics_train` owns PPO/training concerns
- `rust_robotics_sim` owns the app shell, UI, and runtime integration

There are also some good platform seams already in place:

- MuJoCo native/web split behind `MujocoPanel`
- PPO native/web split behind `simulator/ppo_trainer`
- shared robot FW kept out of MuJoCo world ownership

## Main Problems

The current pain points are mostly inside the crate boundaries, not at the top level.

### 1. `rust_robotics_algo` is too broad

It currently mixes:

- pure algorithms
- robot-framework logic
- ONNX/runtime-backed code
- platform-conditioned inference dependencies

That makes it harder to reuse pure algorithms without also inheriting heavier runtime concerns.

### 2. `rust_robotics_sim` mixes UI, domain logic, and runtime glue

Examples:

- [`rust_robotics_sim/src/simulator/pendulum.rs`](./rust_robotics_sim/src/simulator/pendulum.rs)
- [`rust_robotics_sim/src/simulator/mod.rs`](./rust_robotics_sim/src/simulator/mod.rs)

These modules currently combine:

- simulation state
- controller/domain logic
- app coordination
- training integration
- egui rendering
- browser/native test and runtime hooks

That makes the code harder to read, test, and move.

### 3. `rust_robotics_sim` depends directly on `rust_robotics_train` types

Current examples:

- `PolicySnapshot`
- `PpoTrainerConfig`
- `PpoSharedState`
- `PpoTrainerSession`

This is workable today, but it couples the app layer to the training layer more than necessary.

### 4. Browser/native runtime concerns are still spread through app-facing code

The repo already improved this in MuJoCo and PPO, but some browser-specific coordination still
leaks upward into general app modules.

## Target Architecture

The goal is to move toward this shape:

### A. Pure shared foundation

`rust_robotics_core`

Owns:

- shared math/type aliases
- serde DTOs used across crates
- lightweight traits
- generic utility helpers
- policy/action/observation snapshot types that do not require training runtime

Properties:

- no UI
- no ONNX runtime ownership
- no app shell
- minimal target-specific dependencies

### B. Pure reusable algorithms

`rust_robotics_algo`

Owns:

- inverted pendulum dynamics/control math
- path planning
- localization
- SLAM
- vehicle models

Properties:

- should depend on `rust_robotics_core`
- should not own app/UI/runtime glue
- should not need ONNX/web/native interop just to compile core algorithms

### C. Robot framework and inference boundary

Preferred end state:

- `rust_robotics_robot_fw`
- optionally `rust_robotics_inference`

Owns:

- command semantics
- observation building
- action smoothing / recurrent state
- actuation decode
- Go2 / Duck shared controller logic
- ONNX session backend adapters if they remain runtime-dependent

This keeps robot FW reusable without forcing unrelated planning/SLAM users to bring along model
runtime code.

### D. Training runtime

`rust_robotics_train`

Owns:

- PPO algorithm config and session
- Burn backend ownership
- rollout collection
- optimizer state
- training-only env wrappers

It should depend on shared core DTOs rather than being the canonical home of policy snapshot types
used by the app.

### E. App/runtime shell

`rust_robotics_sim`

Owns:

- egui app shell
- mode switching
- native/web viewport/runtime integration
- MuJoCo world ownership
- app-level browser test hooks

But inside the crate, it should be split more clearly into:

- `domain`
- `ui`
- `platform`
- `app`

## Recommended Concrete Refactors

### Refactor 1. Extract shared ML/policy DTOs from `rust_robotics_train`

Current problem:

- `PolicySnapshot` and related types live in [`rust_robotics_train/src/model.rs`](./rust_robotics_train/src/model.rs)
- `rust_robotics_sim` uses them directly

Recommended move:

- create `rust_robotics_core` or `rust_robotics_policy`
- move:
  - `LinearSnapshot`
  - `PolicySnapshot`
  - `ValueSnapshot`
  - possibly `PpoSharedState`

Effect:

- `sim` can consume model artifacts without depending on the training crate
- training becomes an implementation detail, not a shared type owner

### Refactor 2. Split `pendulum.rs` into domain / training / UI

Current file:

- [`rust_robotics_sim/src/simulator/pendulum.rs`](./rust_robotics_sim/src/simulator/pendulum.rs)

Suggested shape:

- `src/simulator/pendulum/domain.rs`
- `src/simulator/pendulum/training.rs`
- `src/simulator/pendulum/ui.rs`
- `src/simulator/pendulum/tests.rs`
- `src/simulator/pendulum/mod.rs`

Move responsibilities:

- `domain.rs`
  - `Controller`
  - state update
  - noise model
  - restart/reset logic
- `training.rs`
  - PPO coordinator wiring
  - policy sync behavior
- `ui.rs`
  - egui controls and card rendering
- `tests.rs`
  - deterministic scenario and contract tests

Effect:

- biggest readability win with limited behavioral risk
- much easier to test and review

### Refactor 3. Split `Simulator` orchestration from UI layout

Current file:

- [`rust_robotics_sim/src/simulator/mod.rs`](./rust_robotics_sim/src/simulator/mod.rs)

Suggested shape:

- `src/simulator/mod.rs`
- `src/simulator/state.rs`
- `src/simulator/ui.rs`
- `src/simulator/help.rs`
- `src/simulator/testing.rs`

Move responsibilities:

- `state.rs`
  - `Simulator` state
  - mode switching
  - pause/restart/reset
  - fixed-step advancement
- `ui.rs`
  - egui layout
  - shared controls row
  - sidebar / viewport composition
- `help.rs`
  - walkthrough/help popup structs and rendering
- `testing.rs`
  - wasm browser test bridge and test-only helpers

Effect:

- the app coordinator becomes easier to reason about
- UI changes stop colliding with runtime behavior changes in one file

### Refactor 4. Separate pure algorithms from robot FW/inference in `rust_robotics_algo`

Current hot spot:

- [`rust_robotics_algo/src/robot_fw`](./rust_robotics_algo/src/robot_fw)

Recommended split:

- keep pure algorithms in `rust_robotics_algo`
- move robot FW into `rust_robotics_robot_fw`
- if needed, move ONNX runtime adapters into `rust_robotics_inference`

Effect:

- better portability for algorithm-only consumers
- clearer ownership model
- easier to test robot FW separately from unrelated planning/SLAM code

### Refactor 5. Make platform adapters explicit

Recommended directory shape in `rust_robotics_sim`:

- `src/platform/native/...`
- `src/platform/web/...`

Candidates to move:

- [`rust_robotics_sim/src/web_ppo_worker.rs`](./rust_robotics_sim/src/web_ppo_worker.rs)
- wasm/native app test hooks
- browser-specific runtime bridge code that is not simulator-domain logic

Effect:

- cleaner separation between app logic and target-specific implementation details

## What To Keep As-Is

These boundaries are already good and should not be undone:

- MuJoCo world ownership stays in `rust_robotics_sim`
- shared robot command/observation/actuation logic stays outside MuJoCo world ownership
- native and web backends may stay separate internally if they share only partial code

Do not:

- move MuJoCo world ownership into `rust_robotics_algo`
- try to make native and web renderers share code prematurely
- split every simulator mode into its own crate right now

## Migration Order

This is the recommended order to reduce risk and keep the repo buildable after each phase.

### Phase 1. Type ownership cleanup

Do first:

1. create a small shared crate for DTOs/traits
2. move policy snapshot types there
3. update `train` and `sim` to depend on that crate

Why first:

- low churn
- immediate modularity benefit
- reduces cross-crate coupling before larger moves

### Phase 2. `pendulum` internal split

Do next:

1. split `pendulum.rs` into domain/training/ui/tests
2. keep public behavior unchanged
3. keep existing tests green

Why second:

- small enough to do safely
- provides a template for the rest of `sim`

### Phase 3. `Simulator` file split

Do after pendulum:

1. split simulator state vs layout/help/testing
2. keep the public `Simulator` entry point stable
3. move browser test hooks out of general app flow where practical

Why third:

- this is a readability refactor with broad touch points
- easier once the pendulum mode already demonstrates the internal pattern

### Phase 4. Robot FW extraction

Do after internal `sim` cleanup:

1. move `robot_fw` out of `rust_robotics_algo`
2. keep path planning/localization/SLAM in `algo`
3. move ONNX adapter ownership to the FW side

Why fourth:

- larger workspace churn
- worth doing, but not the first thing to destabilize

### Phase 5. Platform package cleanup

Do last:

1. make web/native directories more explicit
2. move loose wasm/browser helper code under a platform namespace
3. trim feature branches and compatibility shims once stable

## Proposed End-State Workspace

Reasonable target, not mandatory all at once:

- `rust_robotics_core`
- `rust_robotics_algo`
- `rust_robotics_robot_fw`
- `rust_robotics_train`
- `rust_robotics_sim`

Optional later:

- `rust_robotics_inference`
- `rust_robotics_test_support`

## Immediate Next Step

If only one refactor is started next, it should be:

1. extract shared policy snapshot / trainer-facing DTO ownership out of `rust_robotics_train`
2. then split `rust_robotics_sim/src/simulator/pendulum.rs`

That gives the best balance of:

- meaningful modularity gain
- low migration risk
- improved testability
- cleaner future boundaries for native/web/runtime work
