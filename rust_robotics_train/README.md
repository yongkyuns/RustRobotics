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

## Episode and rollout boundaries

`max_steps` is an external collection time limit, not a finite-horizon task
termination: remaining time is not part of the observation. `StepResult` retains
its existing serialized fields and struct-literal layout. `done` means reset is
required. `truncated` now means a **time-limit-only** end, and `terminated()`
means a task failure. If failure and timeout coincide, failure takes precedence:
`done = true`, `truncated = false`, with the existing failure reward unchanged.
Thus the old overlap case intentionally changes; the clock flag is not a second
independent termination reason. Historical payloads with both flags true cannot
identify whether a simultaneous failure occurred and are not resume checkpoints.

The collector finishes one GAE segment at every environment reset and at the
rollout buffer cutoff. A true failure supplies zero bootstrap. A timeout or
unfinished buffer supplies the critic value of the last **pre-reset** observation.
Advantages never propagate across reset into a different episode. Normalization
still occurs once across the whole rollout. A buffer cutoff alone does not reset
the environment or count an episode. Critic evaluation for bootstrapping occurs
at segment boundaries, not a second time at every ordinary step.

An unfinished episode's undiscounted return belongs to the session, not to a
rollout buffer. It survives collection/update calls and weight transfers, then
is recorded and cleared exactly once when the episode ends. Completed-episode
metrics include the whole episode (also for time limits), not bootstrapped value
targets. Empty rollouts and zero requested updates do not advance/reset episodes.

The boundary suite uses explicit weights, noise-free dynamics fixtures and a
private caller-owned action RNG to compare the real collector to forward f64
TD-error sums. This is not an end-to-end seeded training API. Fourteen trainer
tests and five environment tests cover terminal-mask combinations, pre-reset
bootstrap, reset isolation, coincident failure/timeout, cross-buffer returns,
multiple episodes, negative returns, weight transfer and empty calls. The 1,024
GAE combinations use 64 terminal masks and four gamma/lambda choices each.
Existing PPO objective tests remain unchanged.

References: [Gymnasium time-limit semantics](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
and [Spinning Up PPO path finalization](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html).
The former motivates the bootstrap distinction; the latter illustrates finishing
value targets at trajectory boundaries. These tests do not establish learning
improvement, validate a finite-horizon objective, or repair observation noise's
partial-observability effects.

## Seeded sessions and explicit environment randomness

```rust
use rust_robotics_train::{PpoTrainerConfig, PpoTrainerSession};
let mut trainer = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), 42);
trainer.train_updates(3);
```

`new_seeded(config, seed)` covers actor/critic initialization, initial environment
state, observation/action noise, random disturbances, episode resets, Gaussian
action sampling, minibatch shuffling and bounded-policy entropy sampling. Five
child StdRng streams have a fixed order: actor initialization, critic initialization,
environment, actions, optimizer. The last three continue across updates. Model
size cannot consume collection randomness; optimizer draw counts cannot consume
action or environment randomness. Shuffling and entropy share the optimizer stream.
Reset and weight transfer do not reseed. No process-global/backend seed is set.

The existing `new(config)` draws a fresh entropy seed and uses this same owned
implementation. Existing config fields, serialized layout, signatures and the
standalone `PolicyNetwork::new` / `ValueNetwork::new` APIs remain available.
Seeded MLP initialization eagerly fills weights and biases from
`U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, matching Burn 0.20.1's default LinearConfig
law, rather than relying on lazily evaluated backend-global random tensors. It
preserves the distribution, not Burn's particular random sequence or parameter IDs.

`PendulumEnv` offers `new_with_rng`, `reset_with_rng`, `step_with_rng` and
`observation_with_rng` for caller-owned streams, including evaluation fixtures.
The original methods remain entropy-backed wrappers. Mixing those original
methods into an explicit-stream trajectory opts out of deterministic replay.
The environment itself stores no RNG and gains no mutex/interior mutability.
Standalone noisy observation calls consume draws; trainer snapshot/config/metrics
readouts do not. Cloning an environment copies its state, not an external RNG.

Replay requires the same seed, configuration, inputs, calls, locked dependencies,
backend and numerical build. Tests compare exact numerical values locally within
a build, including multiple actual optimizer updates and concurrently/interleaved
sessions. Bitwise equality across architectures, compiler or dependency upgrades
is not promised; StdRng's algorithm is not a stable serialized format. Store the
seed/config and source/lockfile revisions for experiments. This is fresh-run
reproducibility, not resumable checkpointing: portable shared weights still omit
RNG/optimizer/environment/partial-episode state. The simulator's replica scheduling
and aggregation are not covered by this single-session contract.

References: [Burn 0.20.1 Linear initialization](https://github.com/tracel-ai/burn/blob/v0.20.1/crates/burn-nn/src/modules/linear.rs)
and [rand 0.8 StdRng portability](https://docs.rs/rand/0.8.5/rand/rngs/struct.StdRng.html).

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

## Short-learning evidence

[The frozen learning protocol](LEARNING_QUALIFICATION.md) compares 12 independent
training seeds after 128 default updates against their initial policies, using
32 held-out paired evaluation episodes per seed. Its initial Linux qualification
passed without adjusting seeds or thresholds: all 12 gains exceeded 25 points,
and mean return increased from 55.60 to 127.82. The complete repeat was numerically
identical within that build. Raw episodes, actor snapshots, statistical checks and
provenance are retained by the permanent **PPO learning** workflow.

```sh
python3 scripts/qualify_ppo_learning.py --output target/ppo-learning-evidence
```

The expensive endpoint is explicitly selected by that runner; normal workspace
tests run its 9 evaluator controls, including a real zero-learning-rate negative
control. The Python validator has 12 fail-closed tests. Training-run gains, not
individual episodes or repeated executions, are the independent statistical units.

**Short learning improved; sustained balancing is not qualified.** All 384 trained
policy evaluation episodes still failed before the 10-second cap. This evidence
is limited to the existing noisy linear pendulum and a fixed budget; it does not
establish global convergence, nonlinear/hardware robustness, replica-coordinator
determinism, cross-build bitwise replay or exact-resume checkpoints. Broader
checkpoint work remains open. No training-speed claim is made.

References: [PPO paper](https://arxiv.org/abs/1707.06347) and the
[Spinning Up squashed-Gaussian reference](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py)
(for the action transform, not a change from PPO to SAC).
