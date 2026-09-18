# PPO environment contract

## One physical system

PPO training, evaluation, and the live PPO controller use the same nonlinear point-mass cart-pole and RK4 integrator. The state is `[x, x_dot, theta, theta_dot]` in SI units. Positive force is rightward; positive angle tilts the pole left. The existing linearized `Model` is retained for classical controller prediction, not PPO environment stepping.

`PpoTrainerConfig.plant` carries `length_m`, `cart_mass_kg`, and `pole_mass_kg` into stream zero and every pooled environment. Missing `plant` in older configuration JSON defaults only the physical parameter values; it does not restore the old linear transition function. Length, masses, timestep and force limit must be finite and positive.

## Training and live execution

The live PPO path delegates transition, action clipping, observation/actuation noise, disturbances, reset sampling, and terminal/timeout behavior to `PendulumEnv`. The GUI clock chooses when a fixed step is due; it does not enter the physical transition. Snapshot inference uses the same ordered observation and one `tanh` force conversion.

The global noise toggle/scale is applied to both the current PPO training configuration and its live environment. Scale one uses the existing PPO noise amplitudes. Classical controllers retain their separate demo noise profile. A simulation-only robustness test must explicitly configure a different task rather than assume the live and training controls are independent.

Changing physical or environment/noise settings stops and destroys an incompatible trainer and prevents its retained policy from driving the new plant. An explicit restart creates a matching session. While a browser worker creates a replacement, the viewer waits for its matching snapshot rather than applying retained weights. Entering PPO policy mode uses the PPO reset distribution; ordinary snapshot publication does not reset the live trajectory. Episodes reset on failure or timeout even when training is stopped.

## Migration and historical experiments

Policies trained before this correction learned a linear forward-Euler environment. Their weight layout remains readable, but that is not a promise of equivalent behavior or verified transfer to the nonlinear plant. Retrain for the corrected environment or explicitly evaluate transfer. Bare policy weights do not include a complete physical/environment provenance certificate.

Historical learning scores and frozen reports remain measurements of the old task. A configured learning gate run on the new environment is not a reproduction of the old physical task, even with unchanged neural hyperparameters. Do not compare old linear-on-linear scores with new nonlinear-on-nonlinear scores as a same-task improvement. The correction retest evaluates retained historical weights and newly trained weights on the same nonlinear target.

## Permanent contract tests

The shared plant has equilibrium, force/sign, independent Lagrange/power-balance, upright-linearization, RK4-refinement and invalid-parameter checks. Training tests check nondefault physical propagation to all streams. Live parity tests run the actual seeded stepping wrapper alongside a direct `PendulumEnv`, comparing every physical state, observation, RNG cursor and episode boundary across several physical configurations. Policy activation, hot publication, configuration invalidation and pending-snapshot behavior have separate regressions.

Real-WASM browser tests compare worker and direct execution with nondefault physical parameters, retain legacy-config defaults, and check that a changed displayed plant invalidates the old policy until an explicit restart.

These tests verify consistency and the specified mathematical model, not reliable learned balancing over all initial states, noise conditions, or physical parameters. Learned-policy qualification is a separate acceptance question.
