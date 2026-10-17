# Issue #42 — Frozen Gaussian training gain x evaluation gain crossover

The completed [training-time gain experiment](ppo-gaussian-training-gain-results-20261016.md)
established that gain 1.5 and 2.0 during training improve hard-case
balancing versus gain 1.0, but improve LESS than applying the same gain
only after ordinary 1.0 training. This crossover measures whether a
Gaussian policy adapts to the force map it was trained with.

**All choices are development-exposed** and specified before this matrix
is executed. Reuse seeds 201–204 and the existing ordinary and difficult
per-seed and common 64-key reset panels; no fresh independent #33 confirmation.

## Matrix and control

- **Training gain rows**: 1.0, 1.5, 2.0 (applied consistently for the entire
  976-update, 499,712-interaction training budget).
- **Absolute evaluation gain columns**: 1.0, 1.5, 2.0. At the 976-update
  checkpoint, switch the physical force map of the frozen, fully trained
  actor during read-only evaluation. Gain 2.0 means exactly
  \`clip(2*20*tanh(z), -20,20)\`, NOT twice the training gain.
- **Same** native nonlinear noisy plant, rewards, GAE/Adam/optimizer,
  action latent distribution, original bounded force and 15-second
  horizon. The actor never receives true simulator state or access
  to evaluation rewards for training.
- **No additional training** or checkpoint selection between evaluating
  different columns for the same trained policy.
- Evaluate all 4 reset panels × 64 keys, deterministic/stochastic, all four
  exposed training seeds: **12 policies trained once**, **18,432**
  evaluation episodes at the final checkpoint.

## Qualifying evidence

Download the exact artifact from successful
[workflow #38979375155](https://github.com/yongkyuns/RustRobotics/actions/runs/38979375155)
and verify the 12 immutable prior CSV SHA256 digests. The **diagonal**
(train gain = evaluation gain) must match all 512 *individual*
checkpoint-976 results, including initial conditions, raw returns,
survival, cart margin and policy switching, for **every** training
seed and gain. A numerical mismatch INVALIDATES the crossover regardless
of its apparent outcome. Unsuccessful policies or off-diagonal losses
remain visible; the workflow imposes **no success threshold**.

This is an evaluation-only **within-trained-policy** causal change to the
force map. It can show adaptation to the training force scale, but cannot
prove whether adaptation comes specifically from squared-force reward,
critic state coverage, entropy or PPO action credit. It is NOT hardware
qualification, production continuous Gaussian PPO, or independent #33
confirmation. PR #41 remains draft and unmerged.
