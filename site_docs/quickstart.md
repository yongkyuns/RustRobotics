# Quickstart

This page is the shortest path to getting useful value from Rust Robotics.

## Fastest reading path

If you want to understand the project quickly, read these in order:

1. {doc}`overview`
2. {doc}`tutorials/pendulum`
3. {doc}`tutorials/localization`
4. {doc}`tutorials/path_planning`

That gives a representative sample of control, uncertainty, estimation, and planning without
requiring the full site.

## First interactive things to try

### Control

Open {doc}`tutorials/pendulum` and start with the first guided experiment. Compare the pole angle,
cart excursion, and control effort instead of judging only whether the pole remains upright.

The chapter also explains an important implementation detail: the live plant is nonlinear while
LQR and MPC use a local linear controller model.

### Localization

Open {doc}`tutorials/localization` and vary motion and sensor noise. Look at particle spread,
recovery after drift, and the difference between a visually smooth estimate and a statistically
credible one.

### Planning

Open {doc}`tutorials/path_planning` and compare Dijkstra, A*, Theta*, and RRT. Look at search effort,
path cost, path shape, and how the environment representation changes the result.

## Local setup from a fresh checkout

Prerequisites:

- a current stable Rust toolchain
- the `wasm32-unknown-unknown` Rust target
- Node.js and npm
- Python 3 with `venv` support
- `jq`
- `wasm-bindgen-cli` matching the version used by CI when building the web bundle manually

Build and serve the simulator:

```bash
./build_web.sh --fast
./start_server.sh
```

Then open:

- simulator: `http://127.0.0.1:3000/`

Build the documentation site:

```bash
./scripts/build_docs_site.sh
```

The script creates a repository-local `.venv-docs` environment when needed, installs the pinned
documentation requirements, and builds Sphinx with warnings treated as errors.

Then open or serve:

- built docs: `site_docs/_build/html/index.html`

To test the browser bundle the same way CI does:

```bash
npm ci
npx playwright install chromium
npm run test:web-smoke
```

## What this site is for

The site is meant to explain:

- what each algorithm is for
- where it is used
- what assumptions it makes
- how it compares to alternatives
- what it costs in compute and memory
- how the current implementation differs from the idealized textbook structure
- and how to interpret its behavior in the simulator

If you want a fuller reading sequence, use {doc}`getting_started`.
