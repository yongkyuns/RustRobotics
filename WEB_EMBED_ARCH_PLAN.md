# Web Embed Architecture Plan

This document captures the current inverted-pendulum tutorial embed structure
and the recommended path for scaling the same kind of experience to the other
simulators without turning the Rust/JS boundary into an unmaintainable pile of
UI-specific knobs.

## Current pendulum embed

The focused pendulum tutorial is currently a hybrid of three layers:

1. Rust simulator domain
   - `rust_robotics_sim/src/simulator/pendulum/`
   - Owns plant state, controller selection, PPO trainer integration, history,
     and the authoritative simulation state.
   - The plant now uses a nonlinear cart-pole model integrated with RK4.

2. egui wasm app
   - `rust_robotics_sim/src/app.rs`
   - `rust_robotics_sim/src/simulator/ui/`
   - Owns the scene rendering and egui-native plot surface.
   - Still contains full-app UI paths plus a focused embed mode.

3. Web tutorial shell
   - `docs/index.html`
   - `site_docs/_static/sim_embed.js`
   - `site_docs/tutorials/pendulum.md`
   - Owns iframe sizing, theme propagation, DOM toolbar/cards, and tutorial-page
     responsive layout.

The bridge between Rust and the DOM currently consists of:

- wasm exports in `rust_robotics_sim/src/lib.rs`
- app-level pending command / published state in `rust_robotics_sim/src/app.rs`
- mode-specific DOM rendering and polling in `docs/index.html`

That works for the pendulum tutorial, but it is already showing the scaling
cost of ad hoc exports and mode-specific host-page logic.

## Why the current direct bridge does not scale

If each simulator gets its own growing list of functions like:

- `set_speed`
- `set_noise_enabled`
- `set_noise_scale`
- `add_pendulum`
- `set_controller`
- `patch_pendulum`
- `remove_pendulum`

then each new mode adds:

- more wasm exports
- more app-side pending command plumbing
- more DOM-specific rendering logic in `docs/index.html`
- more duplicated UI semantics between Rust and JS

That is exactly the failure mode to avoid.

## Design goals

For maintainability, web embeds should follow these rules:

1. Rust remains the only source of truth.
2. The tutorial shell exposes only curated controls, not full-app parity.
3. The bridge surface is generic and mode-agnostic where possible.
4. egui continues to own simulator-native rendering.
5. DOM owns layout, page chrome, and lightweight tutorial controls.

## Recommended target architecture

### 1. Separate "sim domain" from "embed adapter"

Each simulator mode should have a Rust-side adapter layer that translates
between:

- internal simulator state
- compact tutorial-facing state
- compact tutorial-facing actions

Conceptually:

```rust
trait SimEmbedAdapter {
    type State;
    type Action;

    fn state(&self) -> Self::State;
    fn dispatch(&mut self, action: Self::Action);
}
```

The adapter should live near the simulator mode, not in `docs/index.html`.

For example:

- `pendulum/embed.rs`
- `localization/embed.rs`
- `path_planning/embed.rs`
- `slam/embed.rs`
- `mujoco/embed.rs`

### 2. Replace many exports with one generic action/state API

Instead of many pendulum-specific wasm exports, prefer:

- `rust_robotics_embed_get_state()`
- `rust_robotics_embed_dispatch_action(action_json)`

where the JSON payload is small and mode-specific inside a generic envelope.

Example:

```json
{
  "mode": "inverted_pendulum",
  "action": {
    "type": "set_controller",
    "pendulum_id": 1,
    "controller": "mpc"
  }
}
```

This keeps the bridge stable even as modes evolve.

### 3. Keep embed controls intentionally narrow

Each tutorial embed should expose only:

- one compact top toolbar
- optionally one per-entity card area
- optionally one advanced folded section
- optionally one plot toggle

Everything else stays in the full egui app.

This avoids mirroring every tuning knob into DOM.

### 4. Keep rendering in egui where it is naturally simulator-owned

Good egui responsibilities:

- scenes
- plots that depend on simulator-native history
- pointer/canvas interactions
- complex in-sim view composition

Good DOM responsibilities:

- tutorial chrome
- page layout
- simple buttons/sliders/selects
- collapsible sections
- responsive spacing

### 5. Move mode-specific DOM rendering out of the monolithic `docs/index.html`

`docs/index.html` should become a host shell, not the location of every mode's
UI logic.

Preferred split:

- `docs/embed_host.js`
  - theme sync
  - iframe sizing
  - generic polling / dispatch helpers
- `docs/embeds/pendulum.js`
- `docs/embeds/localization.js`
- `docs/embeds/path_planning.js`
- `docs/embeds/slam.js`
- `docs/embeds/robot.js`

Each mode renderer should consume the generic bridge and render only the small
tutorial UI relevant to that mode.

## Suggested state / action shape

At the bridge boundary, use a small generic envelope.

### State

```json
{
  "mode": "inverted_pendulum",
  "toolbar": {
    "paused": false,
    "sim_speed": 3
  },
  "view": {
    "show_graph": false
  },
  "payload": {
    "... mode-specific state ..."
  }
}
```

### Action

```json
{
  "mode": "inverted_pendulum",
  "action": {
    "type": "patch_entity",
    "id": 1,
    "patch": {
      "controller": "lqr"
    }
  }
}
```

This keeps:

- global actions global
- mode actions mode-scoped
- per-entity edits patch-based

without multiplying exports.

## Recommended rollout

### Phase 1: stabilize pendulum as the reference implementation

Do first:

1. Introduce a single generic embed dispatch/state API in Rust.
2. Reimplement the current pendulum DOM UI on top of that API.
3. Extract pendulum DOM code out of `docs/index.html`.

Do not add new simulator embeds until this is done.

### Phase 2: add a minimal adapter for the next simplest mode

Recommended order:

1. `path_planning`
2. `localization`
3. `slam`
4. `mujoco`

Rationale:

- path planning has simpler state than robot/MuJoCo
- MuJoCo should come last because it has the heaviest state and the most
  interaction complexity

### Phase 3: standardize embed contracts

Once 2-3 modes exist, formalize:

- common toolbar fields
- common plot toggles
- common entity list patterns
- common action envelopes

At that point the JS host can become thin and reusable.

## What to avoid

Avoid:

- exposing every egui control as a dedicated wasm function
- keeping duplicated authoritative state in JS
- building all mode-specific DOM logic directly in `docs/index.html`
- trying to give the tutorial embed full parity with the full app
- moving simulator-native rendering from egui to DOM unless there is a strong
  reason

## Practical recommendation

The current pendulum embed is a good prototype, but it should be treated as a
temporary architecture checkpoint rather than copied directly.

The next high-value refactor is:

1. collapse the pendulum-specific wasm bridge into a generic embed
   action/state API
2. extract pendulum DOM rendering into a dedicated module
3. define the adapter boundary that all future tutorial embeds must use

That is the point where "web embeds for all sims" becomes viable without
creating an endless Rust/JS UI synchronization burden.
