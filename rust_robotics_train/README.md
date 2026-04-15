# rust_robotics_train

Training workspace crate for reinforcement learning experiments.

Current scope:

- shared inverted-pendulum RL environment
- backend selection for native and wasm builds
- PPO / SAC / DQN-facing config types
- small Burn-based policy/value modules that future training loops can reuse
- custom training loops on top of Burn autodiff instead of `burn-train`, because the
  high-level `burn-train` crate is not wasm-friendly today
- wasm currently uses Burn's `ndarray` backend instead of Burn `webgpu` to avoid
  colliding with the simulator's own `wgpu` web stack during `wasm-bindgen`

Design notes:

- `rust_robotics_algo` stays the source of truth for pendulum dynamics parameters
- `rust_robotics_sim` stays UI-focused
- `rust_robotics_train` owns headless training code and browser-compatible backends
