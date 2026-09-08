# PPO balancing development diagnostics

This is the diagnostic increment of [issue #33](https://github.com/yongkyuns/RustRobotics/issues/33),
not a sustained-balancing acceptance test. The existing
[short-learning qualification](LEARNING_QUALIFICATION.md) remains unchanged.
A higher return, a longer episode, a stable linear reference, or passing this
measurement command is not by itself evidence that PPO solves balancing.

## Frozen baseline experiment

The [plan](https://github.com/yongkyuns/RustRobotics/issues/33#issuecomment-5585901307)
was recorded before executing this development panel. Production baseline is
`2a9e0014601573029be31b3f99e96a082a8b8b39` (merged #32); the diagnostic increment
changes no production Rust code, plant dynamics, reward, noise, defaults,
model architecture, dependencies, or snapshot layout.

Use default seeded sessions for training seeds **201, 202, 203, 204**, continuing
each session to **0, 128, 512, 2048 updates**. At 512 transitions per update,
the final budget is **1,048,576 transitions per training seed**. Intermediate
measurement exports snapshots only and does not reset, reload, or otherwise
modify the live session. Report every checkpoint, never the best checkpoint.

Evaluate each checkpoint on **32 episodes per seed**, with a 1000-step external
cap (nominally 10 seconds). Every episode starts its own explicit RNG with
`0x33000000 + training_seed * 1024 + episode_index`. Initial conditions,
observation noise, action noise and disturbances remain at their default levels.
Controllers share the same random conditions until their trajectories end;
a different end time cannot shift the next episode's conditions. Evaluation
uses deterministic portable actor inference, not Gaussian exploration.

The same 128 episode conditions are also evaluated with zero force and bounded
LQR. Including four PPO checkpoints gives **768 episode records**. The first
two episode indices of each controller/seed/checkpoint have complete step
traces: **48 traced episodes**, selected by index before any result inspection.
All episodes, not only those traces, retain final physical state, maximum
absolute state coordinates, return, duration and force statistics.

These are four development runs, not 128 independent training trials. They
are disjoint from v1's exploratory seeds 0–2 and exposed confirmation seeds
101–112. No sustained-balancing confirmation cohort is used here. This panel
can guide a candidate; that candidate still needs a separately frozen budget,
completion-rate/survival target and new held-out trials.

## Classical diagnostic reference

The LQR reference uses the actual `Model::default().model(dt)` plant matrices,
computed in f32 by production code and promoted to f64 for a test-only DARE
solve. The quadratic design weights are the environment's state/action penalty
weights: `Q = diag(0.2, 0.02, 1.0, 0.05)` and `R = 0.001`.
This does not modify the repository's standalone LQR defaults.

The reference follows the standard [discrete-time Riccati equations](https://underactuated.mit.edu/lqr.html#section4).
The f64 iteration must converge with relative change below `1e-12` within
50,000 iterations and produce a positive-definite solution. A normal test
compares its gain against an independent SciPy Schur-method calculation on
the same promoted A/B/Q/R, checks the Bellman residual, and checks the spectral
radius of the *unclipped, noiseless* closed-loop matrix. SciPy is not a runtime
or CI dependency; the independently calculated reference is stored in the test.

Execution uses `clamp(-K * noisy_observation, -20, 20)`, not privileged physical
state. The same clipping, noisy environment and stopping conditions are used
for PPO and the reference. This is an **undiscounted, unconstrained design
with clipped execution**, not an optimal solution of PPO's discounted,
noise-corrupted, bounded-action task with terminal penalties. Neither the
Riccati equation nor its local eigenvalues prove constrained noisy survival;
that must be measured.

## Recorded diagnostics

- `episodes.tsv`: all 768 episodes; exact seed/checkpoint identities, undiscounted
  return, discounted stopped return, duration, physical failure category,
  absolute-force sum, near-limit count, initial observation, final state,
  coordinate maxima, and the PPO critic's initial value prediction.
- `traces.tsv`: every transition of the 48 preselected episodes; before/after
  physical state, the noisy observation actually used for control, force,
  reward, and end flags. Physical state is logged only for diagnosis, not fed
  to the controller.
- `snapshots.tsv`: all 16 actor/critic pairs as exact big-endian f32 hexadecimal
  values. Shapes are the existing `4 -> 64 -> 64 -> 1` architecture. Actor
  action-limit/exploration metadata precede the weights; each layer is stored
  as input-major weights followed by biases. These are weight snapshots, not
  complete training checkpoints.
- `metrics.tsv`: actual cumulative updates, transitions and episodes, plus
  existing raw last-minibatch policy/value losses and rolling training return.
- `lqr.txt`, `config.txt`, `measurements.log`, `results.json`, and `sources/`:
  design gain, configuration, executed-test output, independent summaries,
  source/lockfile hashes, compiler identity and evidence-file digests.

Failure categories are angle limit, cart-position limit, both, or timeout.
Limits are strict: equality does not itself terminate. A physical failure
at the cap is still failure, not successful completion. The reader independently
recomputes those categories from final state.

Near-limit means commanded `abs(force) >= 0.95 * max_force`. Group fractions
and mean absolute force are weighted by actual executed transitions; they are
not averages of equally weighted episode fractions. Applied force also includes
the unchanged environment's action noise and random disturbance.

The critic diagnostic compares an initial prediction with a stopped,
discounted **deterministic-policy** evaluation return. The critic was trained
on a **stochastic** policy and bootstrapped time limits. A discrepancy therefore
is **not automatically a critic implementation error or a calibrated value-fit
metric**. Likewise, a last-minibatch MSE is not full-dataset explained variance.
Saturation, value fit, discount horizon and observation scaling remain hypotheses
until separately investigated, not root causes established by this tool.

## Reproduction and checks

From the repository root, using the recorded source revision and lockfile:

```sh
python3 -m unittest discover -s scripts -p 'test_diagnose_ppo_balancing.py'
cargo test --locked -p rust_robotics_train --test ppo_balancing_diagnostics
python3 scripts/diagnose_ppo_balancing.py --output /tmp/ppo-balancing-fresh
```

Use a fresh output directory. The wrapper runs the exact ignored measurement
endpoint in release mode, requires its completed one-test success summary,
then checks the complete panel. It does not automatically retry failures.
The ordinary Rust command runs seven fast diagnostic controls and intentionally
ignores the expensive endpoint. That endpoint makes **no learning-quality
assertion**: a panel in which every PPO episode fails can still be a valid,
complete diagnostic measurement.

The seven controls cover boundary precedence, an independent LQR gain and
Bellman residual, stationary reward/discount accounting, an explicit noisy
observation rollout reference, independent episode streams and logging,
invalid actions, and a constant-critic reference. Focused mutation checks
restore double-counted reward, privileged-state feedback, and failure mislabeled
as timeout, and require the corresponding intended runtime assertion failures.
The six Python tests reject malformed vectors/TSV and missing or duplicated
panel data. They run in the existing PPO learning CI job without changing its
v1 fixture, acceptance rule, budget, replay procedure or existing validators.

For every retained trace, the Python reader also checks consecutive steps,
state continuity, observation-noise bounds, Euler position updates, reward
accounting and final episode consistency. It validates finite model snapshots,
complete seed/checkpoint coverage and actual transition budgets. These checks
are not a second independent implementation of PPO or a replay of all random
force draws.

## Evidence and next decision

Observed outcomes, exact execution commits, artifact IDs/checksums and the
current next action are recorded in the [#33 handoff](https://github.com/yongkyuns/RustRobotics/issues/33).
Keep unfavorable development results there as well as favorable ones. Do not
reinterpret this command's successful execution as issue closure.

Use the zero/LQR/PPO comparison to distinguish practical stabilizability under
the specified noisy observation/force contract from PPO training limitations.
Use the fixed checkpoint trajectory to decide whether more default training
resolves the failure or merely changes its type. New default/task changes,
learning recipes or statistical claims require an explicit new protocol.
Results apply to this noisy linear pendulum only; nonlinear plants, hardware,
longer horizons, browser-seeded learning, multi-replica determinism and
exact-resume checkpointing are outside this diagnostic's scope.
