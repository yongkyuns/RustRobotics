# Quickstart

This page is the shortest path to getting useful value from Rust Robotics.

## Fastest reading path

If you want to understand the project quickly, read these in order:

1. {doc}`overview`
2. {doc}`tutorials/pendulum`
3. {doc}`tutorials/localization`
4. {doc}`tutorials/path_planning`

That gives a representative sample of:

- control
- uncertainty and estimation
- graph search and planning

without requiring the full site.

## First interactive things to try

### Control

Open {doc}`tutorials/pendulum` and compare:

- `PID`
- `LQR`
- `MPC`
- `PPO Policy`

Look for:

- overshoot
- settling behavior
- sensitivity to noise
- smoothness versus aggressiveness

### Localization

Open {doc}`tutorials/localization` and vary:

- motion noise
- sensor noise

Look for:

- particle spread
- recovery after drift
- estimate confidence versus diversity

### Planning

Open {doc}`tutorials/path_planning` and compare:

- Dijkstra
- A*
- Theta*
- RRT

Look for:

- search effort
- path shape
- sensitivity to clutter
- graph-based versus sampling-based behavior

## Local setup

For local interactive use, the standard workflow is:

```bash
./build_web.sh --fast
./start_server.sh
source /tmp/rust-robotics-docs-venv/bin/activate
./scripts/build_docs_site.sh
```

Then use:

- simulator: `http://127.0.0.1:3000/`
- built docs: `site_docs/_build/html/`

## What this site is for

The site is meant to explain:

- what each algorithm is for
- where it is used
- what assumptions it makes
- how it compares to alternatives
- what it costs in compute and memory
- and how to interpret its behavior in the simulator

If you want a fuller reading sequence, use {doc}`getting_started`.
