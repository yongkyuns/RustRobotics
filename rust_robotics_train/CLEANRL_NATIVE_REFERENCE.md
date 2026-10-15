# Native CleanRL-reference PPO — experimental

**Status: prototype; no validated 499,712-interaction training result; not the production trainer.**

This is a port of the successful *recipe*, not a byte-for-byte port of
CleanRL's PyTorch implementation. It lives behind
\`rust_robotics_train/cleanrl-reference\` and **does not replace** the
existing continuous-action PPO trainer, change its config defaults or update
the simulator/browser training UI.

Base: \`fix/ppo-shared-nonlinear-plant\` (PR #38), not \`master\`. The base
includes PR #37's persistent shared-policy rollout fix. Do **not** compare
against the older linear-Euler PPO training dynamics on master.

## Source of truth

- Upstream [CleanRL \`ppo.py\`](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo.py) at pinned commit \`fe8d8a03c41a7ef5b523e2e354bd01c363e786bb\` (MIT license).
- Recorded [native comparison](https://github.com/yongkyuns/RustRobotics/blob/audits/cleanrl_comparison_20261015/RESULTS.md) (issue #35, protocol comment 6201479730). **That experiment trained in PyTorch, not Rust.**
- Physical environment \`rust_robotics_train/src/env.rs\` on PR #38; the 256-step cutoff is *truncation*, not failure.

## Implemented in the opt-in diagnostic module

- **Categorical** left/right force \`-max_force\`/\`+max_force\`, two policy
  logits, stable log-softmax, explicit sampled-action log-probabilities.
- Independent 64/64 \`tanh\` policy and value MLPs, zero biases,
  gain-controlled orthogonal initialization with a numerical Gram-Schmidt
  generator. Correct statistical initialization, not the exact Torch QR RNG.
- One combined actor-critic Burn autodiff module and one persistent Adam
  optimizer, epsilon \`1e-5\`; native module-level norm clipping to \`0.5\`.
- Four persistent independent rollout streams; 128 samples per environment
  and update; 512 samples per update.
- GAE \`gamma=0.99\`, \`lambda=0.95\`, minibatch sample-standard-deviation
  advantage normalization; batch=128, epochs=4, clipped PPO ratio=0.2.
- Critic value clipping against rollout old values, entropy coefficient
  0.01, joint loss weights matching the reference.
- Linear rate annealing \`0.00025 * (1 - update_index / 976)\` using one
  monotonic global update counter and a persistent Adam state.
- **Timeout-only** target correction: add
  \`gamma * V(final, pre-reset noisy observation)\` once to the reward used
  for GAE. True physical failures receive zero bootstrap and all episode
  boundaries stop advantage propagation. Original physical rewards are not
  rewritten.

## Semantic and numerical differences still requiring qualification

This is **not** claimed to be identical to CleanRL:
1. Orthogonal weight generation differs from Torch's QR numerical and RNG
   identity, so identical seeds do not imply identical initial matrices.
2. The pinned CleanRL source uses a global PyTorch gradient norm; Burn's norm
   clipping configuration must be independently checked for the same scope.
3. The Rust rollout stream seeding and minibatch permutation order are
   independent of Torch, and no exact trajectory-equality is expected.
4. Gym vectorized evaluation and native model export/snapshot and UI
   integration have **not** been ported.
5. Actual 4-seed, full-budget Rust training, the original independent native
   difficult-case panels, retained checkpoint evaluation and direct
   step-by-step PPO gradient/Adam comparisons **have not yet been run**.
6. Live tests can verify timeout targets and finite forward/backward, but
   cannot establish performance, long-run retention, or deployment safety.

## How to check the prototype

\`\`\`bash
cargo clippy --locked -p rust_robotics_train --features cleanrl-reference --lib --tests -- -D warnings
cargo test --locked -p rust_robotics_train --features cleanrl-reference --lib cleanrl_reference::tests -- --nocapture
\`\`\`

GitHub Actions \`Experimental CleanRL native PPO\` triggers on the
experimental branch. No task should be promoted to the supported product path
unless these tests pass and a separate predeclared held-out multi-seed
qualification demonstrates sustained balancing, including all regressions.

Protected MuJoCo and control source, current runtime snapshots and regular
continuous trainer are untouched. The experimental module intentionally has no
user-facing switch; the goal remains one supported reliable trainer, not two
competing production pathways.
