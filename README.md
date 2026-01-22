# Rust Robotics

A collection of robotics algorithms and interactive simulations implemented in Rust.

## Features

- **Robotics Algorithms (`rust_robotics_algo`)**:
  - **Localization**: Particle Filter implementation.
  - **Control (Inverted Pendulum)**:
    - LQR (Linear Quadratic Regulator)
    - MPC (Model Predictive Control)
    - PID Control
- **Interactive Simulator (`rust_robotics_sim`)**:
  - Built with `egui` and `eframe`.
  - Runs natively on Linux, Windows, and macOS (Intel & ARM).
  - Supports WebAssembly (WASM) for browser-based simulation.
- **Python Interface (`python_interface`)**:
  - Python bindings for core algorithms using PyO3.

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (stable)
- For Linux: `libgtk-3-dev`, `libxcb-render0-dev`, `libxcb-shape0-dev`, `libxcb-xfixes0-dev`, `libxkbcommon-dev`, `libssl-dev`

### Running the Simulator (Native)

To run the interactive simulator:

```bash
cargo run -p rust_robotics_sim
```

### Building for Web (WASM)

You can build the simulator for the web using the provided script:

```bash
./build_web.sh
```

This will generate the WASM artifacts in the `docs` folder, which can be served using any web server.

## CI/CD

The project includes a GitHub Actions workflow that automatically builds and tests the codebase across:
- Ubuntu (Linux)
- Windows
- macOS (Intel & ARM)
- WebAssembly (WASM)

## License

This project is licensed under the MIT License.