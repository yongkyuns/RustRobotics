# Control Systems Tutorial

The inverted pendulum is a compact way to study a question that appears throughout robotics:
how should a controller trade state error against control effort when the plant is unstable and
its states are coupled?

This chapter uses one shared nonlinear cart-pole simulation to compare PID, LQR, MPC, and a PPO
policy. The goal is not to pick a universal winner. It is to understand what information each
controller uses, what assumptions it makes, and what changes when those assumptions stop matching
the plant.

```{raw} html
<div class="sim-embed-card">
  <iframe
    class="sim-embed-frame"
    data-sim-mode="inverted_pendulum"
    data-sim-path="?mode=inverted_pendulum&embed=focused&ui=20260907a"
    title="Rust Robotics control systems simulator"
    loading="lazy"
  ></iframe>
</div>
```

## Learning goals

By the end of this chapter, you should be able to:

- explain why the upright equilibrium is unstable
- distinguish the nonlinear simulated plant from the linear controller model
- explain what PID, LQR, MPC, and PPO use when choosing an action
- compare controllers under the same state, objective, and disturbance assumptions
- interpret overshoot, settling, cart excursion, and control effort
- identify when an apparent algorithm comparison is really a tuning or objective comparison

## The exact system used here

The simulator state is

$$
\mathbf{x} =
\begin{bmatrix}
x & \dot{x} & \theta & \dot{\theta}
\end{bmatrix}^T,
$$

where `x` is cart position, `\dot{x}` is cart velocity, `\theta` is pole angle, and
`\dot{\theta}` is pole angular velocity. The control input `u` is horizontal cart force.

The live plant is **nonlinear** and is integrated with fourth-order Runge-Kutta. The fixed
pendulum simulation step is **0.01 s**. Around the upright equilibrium, the classical model-based
controllers use the linear approximation

$$
\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u},
$$

which the reusable algorithm crate discretizes with first-order forward Euler:

$$
A_d \approx I + A_c\Delta t, \qquad B_d \approx B_c\Delta t.
$$

This distinction is deliberate. LQR and MPC therefore control a nonlinear plant using a local
linear model. Small-angle behavior is a model-matched regime; large excursions reveal model
mismatch.

```{admonition} Important comparison rule
:class: note-shell

Changing controller type is not enough to create a fair experiment. Cost weights, initial state,
noise, and constraints can change the result as much as the algorithm family. When comparing two
controllers, first decide which of those quantities you intend to hold fixed.
```

## A useful measurement set

Do not judge a controller only by whether the pole remains upright. For each experiment, watch:

- peak pole-angle error
- settling behavior
- maximum cart excursion
- residual oscillation
- peak and sustained control effort
- failure or recovery after a large excursion

The plots make these quantities visible even when two animations look superficially similar.

## PID: reactive error correction

The pendulum PID baseline acts on pole-angle error:

$$
u(t) = K_p e(t) + K_i \int e(t)\,dt + K_d \frac{de(t)}{dt}.
$$

The default implementation uses gains `Kp = 25`, `Ki = 3`, and `Kd = 3`. It is intentionally a
simple baseline: the controller directly regulates angle rather than solving a coupled optimal
control problem over the full state.

That makes PID useful for learning two things. First, very inexpensive feedback can stabilize a
surprisingly difficult plant. Second, stabilizing the pole is not identical to managing cart
position, velocity, and control effort well.

## LQR: local optimal feedback

LQR starts from the linearized discrete model and minimizes

$$
J = \sum_{k=0}^{\infty}
\left(x_k^T Q x_k + u_k^T R u_k\right).
$$

The matrices `Q` and `R` determine what the controller considers expensive. Once the Riccati
solution and feedback gain are prepared, the familiar LQR runtime law is

$$
u_k = -Kx_k.
$$

The algorithm crate offers both a one-shot `control()` API, which prepares the gain for each
call, and `PreparedLqr`, which reuses a prepared gain. The pendulum simulator uses the prepared
path: it computes the gain on the first step and rebuilds it when the controller model or timestep
changes. Resetting the simulation or stepping with a different controller clears the cache.
For unchanged configuration, LQR therefore performs the matrix-vector feedback operation without
repeating the Riccati solve on every simulation tick.

The default model uses

$$
Q = \operatorname{diag}(0, 1, 1, 0), \qquad R = 0.01.
$$

Notice that default cart-position error has zero direct state cost. This is a useful tuning lesson:
a controller can be very good at the objective you gave it while looking bad under an objective
you never encoded.

## MPC: finite-horizon optimization at runtime

MPC repeatedly solves

$$
\min_{\{x_k,u_k\}}
\sum_{k=0}^{N-1}
\left(x_k^TQx_k + u_k^TRu_k\right) + x_N^TPx_N
$$

subject to its prediction model and constraints. Only the first control action is applied before
the problem is solved again.

The pendulum demo currently uses a horizon of **12** in the focused web controls. Its default stage
cost is not identical to the default LQR cost: the simulator sets MPC state weights to
`diag(1, 1, 10, 1)`. That means a default LQR-versus-MPC run compares both the method **and** the
objective. This is fine for exploring tuned behavior, but it should not be interpreted as a
controlled algorithm-only benchmark.

For a fixed controller model and timestep, `PreparedMpc` stores the terminal Riccati solution's
contribution, assembled horizon matrices in sparse CSC form, the zero-reference linear objective,
and the constant constraint right-hand side. The pendulum runtime reuses this problem data until
its controller model or timestep changes. Resetting the simulation or stepping with another
controller discards the cache. Each solve supplies the current measured initial state.

This is **problem-data reuse**, not solver or factorization reuse: a fresh Clarabel solver is
still constructed and run for each control step. It does not warm-start the optimization.
The one-shot `try_mpc_control()` API remains available for callers that do not retain a
`PreparedMpc` instance.

MPC also has a qualitatively different failure mode: an optimizer can fail to produce a usable
solution. `PreparedMpc::try_control()` and `try_mpc_control()` return a `Result` so a solver failure
is distinguishable from a valid zero-force optimum. The pendulum runtime records the failure in
`last_control_error()` and the web card's `control_error` snapshot field, then uses a zero-force
fallback. A successful control step or reset clears the stale error. The compatibility
`mpc_control()` wrapper also retains its historical zero-force fallback. This fallback is an
educational simulator policy, not a recovery controller or a hardware-safety guarantee.

## PPO: learned feedback

The PPO controller evaluates a learned policy

$$
u = \pi_\theta(o).
$$

The main distinction is not simply "neural network versus equation." The policy is produced by a
training process and reward design rather than directly by solving a model-based control problem.
Training cost, reproducibility, action bounds, checkpoint semantics, and policy readiness therefore
belong to the comparison just as much as inference cost does.

The PPO path in this repository should be treated as an evolving educational feature while the
tracked policy-objective and bounded-action correctness work is completed. Do not use one short
training run as evidence that PPO is inherently better or worse than the classical controllers.

## Experiment 1: separate stabilization from cart regulation

**Question:** Can the pole look well controlled while the cart objective is poor?

1. Reset the pendulum and select LQR.
2. Disable added noise.
3. Observe pole angle, cart position, and control effort under the default weights.
4. Increase the cart-position weight while leaving the other weights unchanged.
5. Reset and compare the new response.

**Prediction:** increasing cart-position cost should make the controller care more explicitly about
cart excursion. The exact trajectory varies because the current reset path randomizes the initial
pole angle, so compare several resets rather than treating one run as a deterministic benchmark.

**What this teaches:** "optimal" always means optimal for a specified model and objective.

## Experiment 2: expose local-model limits

**Question:** When does the linear controller model stop describing the nonlinear plant well?

1. Select LQR and begin from an ordinary small-angle reset.
2. Observe the recovery.
3. Repeat from progressively more challenging pole angles or disturbances.
4. Compare recovery time, peak force, and whether the pole leaves the local stabilization region.

**Prediction:** behavior should degrade as the trajectory spends more time far from the upright
linearization point.

**What this teaches:** local linear control can be excellent without being globally valid.

## Experiment 3: compare objectives before algorithms

**Question:** Is MPC's visible behavior caused by planning, or by different cost weights?

1. Record the active LQR `Q` and `R` values.
2. Record the active MPC `Q`, `R`, and horizon.
3. Compare the default responses and note that both method and objective changed.
4. Then make the state/control costs as similar as the current UI permits and repeat.

**Prediction:** part of the default visual difference should disappear when the objectives become
more comparable. Any remaining difference is a more meaningful place to discuss finite-horizon
optimization versus static feedback.

## Experiment 4: robustness under measurement and actuation noise

**Question:** Which controller degrades gracefully under the same disturbance model?

1. Choose a controller and establish a no-noise baseline.
2. Enable noise at a low level.
3. Increase it gradually without changing controller tuning.
4. Repeat for the other controllers.
5. Compare angle error, cart excursion, control effort, and failure rate across multiple resets.

The simulator perturbs both measured state and applied force when pendulum noise is enabled. This
is therefore a combined sensing-and-actuation robustness experiment, not a pure sensor-noise test.

## Complexity and implementation reality

| Method | Algorithmic online work after preparation | Current demo caveat | Main tradeoff |
| --- | --- | --- | --- |
| PID | constant scalar arithmetic | angle-focused baseline | simple but weakly expresses coupling |
| LQR | matrix-vector multiply | preparation repeats when the cached configuration changes | excellent local linear feedback |
| MPC | optimization every step | fixed QP data is reused, but the solver is constructed and run each tick | constraints and anticipation at runtime cost |
| PPO | neural-policy inference | correctness/reproducibility work is still tracked | flexible learned behavior after training |

## Common mistakes

- comparing controller names while silently changing objectives
- calling local stabilization "global stability"
- looking only at pole survival and ignoring cart excursion or effort
- drawing conclusions from one randomized reset
- comparing PPO inference cost without including training and reproducibility
- treating solver fallback as though it were an intended MPC action

## Where to go next

After this chapter, use the localization tutorial to see the same experimental discipline applied
to uncertainty: define the state and sensor model, change one assumption at a time, and measure how
the estimator responds rather than relying only on visual plausibility.
