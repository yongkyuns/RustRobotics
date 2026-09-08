# rust_robotics_train

Burn-backed PPO training for the inverted-pendulum environment. The crate owns
rollouts, actor/critic optimization and snapshot conversion. `rust_robotics_core`
owns the portable snapshots and metrics; the simulator owns UI and live worlds.
Native and WASM training currently use Burn's NdArray autodiff backend. The custom
loop avoids `burn-train`; NdArray avoids colliding with the simulator's web `wgpu`
stack. SAC/DQN configuration types remain preparatory, not implemented trainers.
Pendulum dynamics parameters remain owned by `rust_robotics_algo`.

## Bounded policy and objective

For raw MLP output `mu`, positive force limit `L`, and positive configured scale
`s = action_std`, training samples:

```text
sigma = s / L
z ~ Normal(mu, sigma^2)
a = L * tanh(z)
log pi(a|obs) = log Normal(z; mu, sigma^2) - log L - log(1 - tanh(z)^2)
```

Exploration is applied **before** tanh, not added to a bounded mean and clamped.
`action_std` retains force-scale units before squashing; it is not the standard
deviation of the executed action. The stochastic policy intentionally differs
from the old clipped Gaussian. Deterministic inference and the snapshot layout
remain unchanged: `L * tanh(mu)`. This squashed latent mean is not generally the
expectation of the nonlinear action distribution.

Rollouts retain `z` and its transformed log density. Replay never reconstructs a
latent with inverse tanh of an f32 action that may have rounded to a bound. The
stable log-Jacobian uses only nonpositive exponent arguments; it stays finite
for finite saturated latents. Environment actuation noise and disturbances are
separate transition dynamics, not part of the policy density.

The actor minimizes the negative clipped PPO surrogate minus
`entropy_coef * H(pi_current)`. The Gaussian scale is fixed, but bounded-policy
entropy depends on the mean. Each entropy minibatch draws fresh Gaussian noise
and differentiates through `z = mu + sigma * noise`, adding the analytic Gaussian
entropy to the sampled log-Jacobian and `log L`. Negative log likelihood of old
rollout actions is **not** substituted for current-policy entropy. Zero entropy
coefficient avoids this extra sampling and gradient term.

The critic minimizes `value_loss_coef * mean((V - returns)^2)`. A zero coefficient
skips its optimizer step, so existing Adam moments cannot move its parameters.
Positive coefficients scale the objective and gradients; with separate Adam
optimizers, gradient normalization can make positive rescalings produce similar
parameter steps. No proportional parameter-step guarantee is made.

Both coefficients must be finite and nonnegative; exploration scale and force
limit must be finite and positive. Invalid constructor inputs panic rather than
silently using a different sampling scale from the density. Metrics retain their
existing interpretation: `last_policy_loss` is the unregularized surrogate and
`last_value_loss` is raw MSE, both from the last minibatch, not total weighted loss.

## Snapshot and synchronization semantics

`PolicySnapshot` is deterministic inference state. `PpoSharedState` transfers
actor and critic weights, action limit and exploration scale between replicas.
`load_shared_state` resets optimizer moments, adopts the saved exploration scale,
and rejects a force-limit mismatch before loading weights, because the receiving
environment would otherwise clip a different distribution. It retains the local
environment trajectory and metrics. These snapshots do **not** include optimizer
moments, RNG state, environment state or rollout state; they are warm-start / weight
synchronization data, not exact-resume training checkpoints. Loading old weights
preserves deterministic inference, not the old clipped stochastic policy.

## Verification

```sh
cargo test --locked -p rust_robotics_train --lib trainer::objective_tests -- --show-output
cargo test --locked -p rust_robotics_train
cargo clippy --locked -p rust_robotics_train --all-targets -- -D warnings
```

The 17 objective tests use independent f64 change-of-variables calculations,
action-space density quadrature, seeded CDF checks, finite-difference entropy and
clipping gradients, weighted critic gradients and real optimizer steps. They cover
saturated latents, likelihood replay, zero-coefficient momentum, snapshot inference
and synchronization limits. Tolerances and sampling bounds are stated in the tests.
A passing gradient or smoke test is not evidence of policy improvement.

This is the action/objective increment of issue #5. End-to-end session/environment
seeding, reproducible short-learning qualification and broader checkpoint work
remain open. Existing rollout episode-return accounting and time-limit bootstrap
handling also need separate review before a learning-quality claim. No convergence,
performance or complete PPO-correctness claim is made here.

References: [PPO paper](https://arxiv.org/abs/1707.06347) and the
[Spinning Up squashed-Gaussian reference](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py)
(for the action transform, not a change from PPO to SAC).
