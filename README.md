# Rust Robotics

Rust Robotics is a Rust workspace for two closely related goals:

1. implement robotics algorithms and reusable control / estimation components
2. expose those systems through an interactive native and web simulator

The repository is intentionally split so that simulation, training, and reusable
algorithm code can evolve independently without losing a shared vocabulary.
The current architecture is especially centered on a MuJoCo-backed robot stack,
an inverted-pendulum teaching example, and a growing set of localization, SLAM,
and path-planning demos.

## Workspace Layout

### `rust_robotics_algo`

Reusable robotics logic:

- control algorithms such as LQR, MPC, PID, and vehicle dynamics
- localization and SLAM implementations
- path-planning algorithms
- shared robot-framework logic for Go2 and Open Duck Mini

This crate should own robot semantics and algorithmic policy logic, not the
simulation world or rendering backend.

### `rust_robotics_core`

Small shared crate for portable ML / policy data transfer objects:

- `LinearSnapshot`
- `PolicySnapshot`
- `ValueSnapshot`
- `PpoMetrics`
- `PpoSharedState`

This crate exists so the simulator can consume policy snapshots without taking
a deep dependency on the training runtime itself.

### `rust_robotics_train`

Training-side runtime and PPO implementation:

- pendulum environment used for PPO training
- Burn-based actor / critic models
- PPO update loop, rollout collection, and metrics
- conversion between train-time networks and portable snapshots

This crate owns optimization and training concerns. It should not own UI,
rendering, or simulator orchestration.

### `rust_robotics_sim`

Interactive application and world ownership:

- `egui` / `eframe` application shell
- native and web simulator runtime
- MuJoCo world integration and viewport glue
- mode-specific demos for pendulum, localization, SLAM, planning, and robot
- browser smoke-test hooks for the web build

This crate owns the live world state, stepping, reset semantics, and UI flow.

### `docs`

Generated web bundle output. `build_web.sh` writes the publishable browser build
here, and `start_server.sh` serves it locally with the required COOP / COEP /
CORP headers for browser MuJoCo and wasm execution.

## Architectural Overview

The most important boundary in the repository is:

- `rust_robotics_sim` owns the world
- `rust_robotics_algo` owns reusable robotics logic
- `rust_robotics_train` owns policy optimization
- `rust_robotics_core` owns portable shared snapshot formats

That means:

- MuJoCo model/data, stepping, reset, rendering, and UI belong in `sim`
- command semantics, observations, recurrent/action state, and actuation decode
  belong in `algo`
- optimizer state, rollout collection, PPO losses, and Burn modules belong in
  `train`
- serializable policy/value representations that cross crate boundaries belong
  in `core`

This split is what allows the same pendulum policy to be:

1. trained in `rust_robotics_train`
2. serialized into `rust_robotics_core::PolicySnapshot`
3. consumed in `rust_robotics_sim`
4. executed in both native and web contexts

## Simulator Structure

`rust_robotics_sim::simulator` is the application coordinator. Its current
internal layout is intentionally split by responsibility:

- `simulator/mod.rs`
  - public traits (`Simulate`, `Draw`, `SimulateEgui`)
  - top-level `Simulator` state buckets
  - mode selection and UI entrypoint
- `simulator/runtime.rs`
  - fixed-step time integration
  - pause / restart / mode switching
  - per-mode stepping and reset behavior
- `simulator/help.rs`
  - guided tutorial / highlight overlay state
- `simulator/ui/`
  - shared UI layout, sidebar, plots, and per-mode control cards
- `simulator/pendulum/`
  - `domain.rs`: plant, controller logic, state evolution
  - `training.rs`: PPO synchronization and trainer wiring
  - `ui.rs`: pendulum-specific control panels
  - `tests.rs`: contract and regression coverage

That split is deliberate: the simulator is easiest to maintain when world logic,
training integration, and presentation are readable in isolation.

## Inverted Pendulum Flow

The pendulum demo is a good example of the repo's layering:

1. `rust_robotics_algo::control::inverted_pendulum` provides the plant model
   and classical controllers.
2. `rust_robotics_train` can train a PPO policy against the same task.
3. `rust_robotics_train` exports a `PolicySnapshot`.
4. `rust_robotics_sim::simulator::pendulum` can run LQR, PID, MPC, or PPO
   against the same state vector and plot the resulting trajectories.

The discrete simulation step follows the standard linear update:

`x_(k+1) = A(dt) x_k + B(dt) u_k`

with optional measurement noise on the state seen by the controller and optional
actuation noise on the force actually applied to the plant. This makes it easy
to compare:

- controller law differences
- tuning differences
- robustness under noise
- learned policy behavior versus model-based control

## PPO Flow

The current PPO implementation is intentionally small and inspectable:

1. collect a rollout from the pendulum environment
2. compute returns and generalized advantage estimates (GAE)
3. normalize advantages
4. optimize the clipped PPO surrogate for the policy
5. optimize mean-squared value loss for the critic
6. snapshot actor / critic weights into portable DTOs

The simulator's `PpoTrainerCoordinator` can run one or more trainer replicas and
average their shared states. The simulator always consumes the exported
snapshot, not a live Burn model, which keeps the runtime boundary much cleaner.

## Native And Web Targets

### Native

Typical local checks:

```bash
cargo check -p rust_robotics_algo
MUJOCO_HOME=/path/to/mujoco cargo check -p rust_robotics_sim
MUJOCO_HOME=/path/to/mujoco cargo run -p rust_robotics_sim --release
```

### Web

Typical local checks:

```bash
cargo check -p rust_robotics_sim --target wasm32-unknown-unknown
cargo check -p rust_robotics_sim --target wasm32-unknown-unknown --no-default-features
node --check rust_robotics_sim/web/mujoco_runtime.js
./build_web.sh --fast
./start_server.sh
```

Local URL:

```text
http://127.0.0.1:3000/
```

The web server must provide:

- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Resource-Policy: same-origin`

### Publishing To GitHub Pages

The simulator bundle and tutorial site are built separately:

- full simulator bundle: `docs/`
- tutorial site: `site_docs/_build/html/`

The tutorial embeds now resolve their simulator base at runtime:

- local preview on `localhost` / `127.0.0.1`: `http://127.0.0.1:3000/`
- hosted pages: `/sim/`

So a same-origin Pages layout works well, for example:

- simulator: `/sim/`
- tutorial: `/sim-tutorial/`

Helper script:

```bash
./scripts/publish_pages.sh --build
```

Default sync target:

- `~/Dev/me/blog/yongkyuns.github.io`

Default subdirectories:

- `sim/`
- `sim-tutorial/`

You can override those with:

```bash
./scripts/publish_pages.sh \
  --repo ~/Dev/me/blog/yongkyuns.github.io \
  --sim-dir sim \
  --tutorial-dir sim-tutorial
```

## Testing Strategy

The repo now uses several layers of protection against regressions:

- unit tests in `algo`, `train`, and `sim`
- contract tests around policy snapshot handoff and averaging
- deterministic scenario tests for pendulum stabilization
- real wasm bundle builds in CI
- Playwright browser smoke tests for web startup and basic interaction

Useful commands:

```bash
cargo test --workspace
MUJOCO_HOME=/path/to/mujoco cargo test -p rust_robotics_sim --lib
./build_web.sh --fast
npm run test:web-smoke
```

## Documentation Pointers

High-value project docs:

- [ARCH_REFACTOR_PLAN.md](./ARCH_REFACTOR_PLAN.md)
- [rust_robotics_sim/MUJOCO_ARCHITECTURE.md](./rust_robotics_sim/MUJOCO_ARCHITECTURE.md)
- [rust_robotics_sim/MUJOCO_REMAINING_WORK.md](./rust_robotics_sim/MUJOCO_REMAINING_WORK.md)
- [docs/README.md](./docs/README.md)

The source code itself now carries more detailed rustdoc in the shared core,
trainer, simulator runtime, help, and pendulum modules. When in doubt, start at
`rust_robotics_sim::simulator` and follow the module-level docs outward.

## Design Rules

Some boundaries are intentional and should be preserved:

- do not move MuJoCo world ownership into `rust_robotics_algo`
- do not reintroduce the removed native `glow` viewport path
- keep browser and native rendering paths conceptually aligned, but do not
  assume they are literally the same implementation
- prefer portable snapshot DTOs over direct runtime coupling across crates

## License

This project is licensed under the MIT License.
