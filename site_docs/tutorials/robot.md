# Robot Runtime Tutorial

The earlier chapters isolate one algorithm at a time. A robot runtime has to connect state,
observations, policy/controller execution, action decoding, and physics while respecting timing and
interface contracts. This chapter uses the Go2 path as the main concrete example.

```{raw} html
<div class="sim-embed-card">
  <iframe
    class="sim-embed-frame"
    data-sim-mode="robot"
    data-sim-path="?mode=robot&embed=focused&ui=20260907a"
    title="Rust Robotics robot runtime simulator"
    loading="lazy"
  ></iframe>
</div>
```

## Learning goals

By the end of this chapter, you should be able to:

- describe the runtime path from raw robot state to joint torques
- explain why policy observation ordering is part of the model contract
- identify recurrent/history state that must survive between control steps
- distinguish normalized policy actions from actuator commands
- explain where latency, jitter, and model-loading failures enter the loop
- understand which parts of the current robot stack belong in reusable algorithm code versus runtime/inference adapters

## The concrete Go2 control path

The reusable Go2 controller follows this sequence:

$$
\text{raw state}
\rightarrow \text{history update}
\rightarrow \text{observation + command features}
\rightarrow \text{policy inference}
\rightarrow \text{action smoothing}
\rightarrow \text{PD torque decoding}.
$$

That is more precise than saying "the policy controls the robot." The policy is one component in a
stateful control pipeline.

## Observation contract

The Go2 policy observation is a dense vector of **117 values** built from three samples of:

- body-frame gravity: `3 x 3 = 9`
- 12 joint positions: `3 x 12 = 36`
- 12 joint velocities: `3 x 12 = 36`
- previous actions: `3 x 12 = 36`

Total:

$$
9 + 36 + 36 + 36 = 117.
$$

The controller also carries a recurrent hidden-state buffer of **128 values**. Resetting the robot
controller therefore means more than zeroing a command: history buffers, previous actions, recurrent
state, and initialization state must all be reset coherently.

```{admonition} Observation order is an ABI for the policy
:class: note-shell

A neural policy does not know the semantic names of its inputs. If joint order, history order,
coordinate frame, scaling, or command layout changes while the model stays the same, inference can
still run and produce completely wrong behavior. Golden tests for these layouts are therefore
correctness tests, not cosmetic unit tests.
```

## Command features

The Go2 controller currently supports two high-level command styles.

**Velocity mode** rotates commanded planar velocity into the robot body frame, includes yaw/yaw-rate
features, and appends oscillator features used by the locomotion policy.

**Impedance-style mode** converts a world-space target into the body frame, bounds the local target
radius, and builds a richer target feature vector.

The key engineering point is that world commands should not be passed directly to a policy whose
training contract expects robot-centric features.

## Policy output is not torque

The policy produces at least 12 action values plus recurrent output. The controller low-pass filters
new actions with the previous action:

$$
a_{used} = 0.2\,a_{prev} + 0.8\,a_{new}.
$$

Each used action becomes a joint-position offset around a nominal posture. The final torque is then
computed by a PD law

$$
\tau_i = k_{p,i}(q_{target,i}-q_i) + k_{d,i}(0-\dot q_i).
$$

This separation matters when debugging. If the robot moves badly, the cause can be observation
construction, policy inference, action smoothing/scaling, nominal joint targets, gains, state
ordering, or the physics model. Looking only at the neural-network output hides most of the runtime.

## Experiment 1: trace one command through the stack

**Question:** What actually changes when you change a high-level command?

1. Select the Go2 robot.
2. Start from a reset state and use a small velocity command.
3. Identify the requested world/body motion in the UI.
4. Observe the resulting locomotion.
5. Increase one command component at a time rather than changing several simultaneously.

While watching the robot, mentally trace:

`command -> body-frame command features -> policy -> smoothed action -> target joints -> torque`.

**What this teaches:** a high-level command is transformed several times before it reaches the
simulated actuator.

## Experiment 2: stateful inference and reset behavior

**Question:** Why must reset clear more than the physics pose?

The Go2 path keeps three-step histories and a 128-value recurrent state. A robust reset should clear
those controller states consistently with the simulated robot reset.

1. Run the robot long enough for histories/recurrent state to become nontrivial.
2. Reset the demo.
3. Apply the same small command again.
4. Watch for any obvious transient that suggests stale controller state survived reset.

This is also a good automated regression: reset should restore deterministic controller state even
when the subsequent physics trajectory is not bit-for-bit identical across platforms.

## Experiment 3: action smoothing versus responsiveness

**Question:** What does smoothing buy, and what does it cost?

The current controller blends 20% previous action with 80% new action. That is light smoothing, but
it still changes the temporal response.

1. Apply a steady command and observe normal motion.
2. Change command direction abruptly.
3. Watch how quickly the robot changes behavior.
4. Consider what would happen with no smoothing or much stronger smoothing.

**Tradeoff:** smoothing can reduce abrupt target jumps and torque spikes, but excessive smoothing
adds lag.

## Experiment 4: runtime failures should be visible

**Question:** What happens when the policy contract is not satisfied?

The controller explicitly rejects policy outputs shorter than 12 actions or shorter than 128
recurrent values. Model-loading/inference failures also need to remain distinguishable from a valid
zero or neutral action.

A polished runtime should surface those errors in the UI and tests instead of silently substituting
behavior that looks like a legitimate policy decision.

## Performance and timing

A robot-control step has several costs:

- copying/transforming raw state
- maintaining history buffers
- building observations and commands
- inference
- smoothing/decoding actions
- physics stepping and rendering in the simulator

For control quality, average frame rate is not enough. Measure control-step latency and jitter
separately from rendering. A controller that usually runs quickly but occasionally stalls can behave
very differently from one with consistent timing.

## Portability boundary

Rust Robotics wants control semantics to be reusable across native and web runtimes. That goal is
best served by keeping these layers explicit:

1. robot semantics and observation/action contracts
2. inference backend adapters
3. simulator/world integration
4. UI/rendering

The current `rust_robotics_algo` crate still includes robot-framework and inference-related
dependencies alongside pure control/localization/planning/SLAM algorithms. Extracting that boundary
remains useful cleanup because an A* or particle-filter user should not need the robot policy
runtime dependency graph.

## What to measure

For runtime experiments, prefer:

- observation/action dimensions and contract checks
- inference latency
- complete control-step latency
- jitter/worst-case latency
- reset correctness
- command tracking error
- joint/torque saturation where available
- model-loading and inference error rate

## Common mistakes

- treating simulator state as though it can be fed directly to a trained policy
- changing input ordering without changing/retraining the model
- assuming policy action equals actuator torque
- resetting physics while leaving recurrent controller state intact
- hiding inference failure behind a neutral-looking action
- measuring rendering FPS instead of control-loop timing
- mixing pure algorithm reuse with heavyweight inference/runtime dependencies

## What this chapter is really teaching

Robotics software becomes useful when mathematical components are connected by explicit contracts.
Observation layout, state lifetime, coordinate frames, inference semantics, decoding, and timing are
part of control correctness—not plumbing around it.
