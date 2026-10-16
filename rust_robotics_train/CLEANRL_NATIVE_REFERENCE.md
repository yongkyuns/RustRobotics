# Native CleanRL-reference PPO — experimental

**Status: experimental. Seed 201 completed its full 976-update / 499,712-interaction
native training budget in [run #38844353112](https://github.com/yongkyuns/RustRobotics/actions/runs/38844353112);
this does not qualify multi-seed or production reliability.**

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
2. The pinned CleanRL source clips the single GLOBAL L2 gradient norm
   across all actor/critic parameters. Burn 0.20's built-in norm clipping is
   per-tensor and therefore incorrect for that recipe. This branch now uses
   an explicit **shared global coefficient across all 12 parameter tensors**,
   before Adam. This algorithm still requires numerical comparison against
   pinned PyTorch to qualify its floating-point results.
3. The Rust rollout stream seeding and minibatch permutation order are
   independent of Torch, and no exact trajectory-equality is expected.
4. Gym vectorized evaluation and native model export/snapshot and UI
   integration have **not** been ported.
5. Seed 201 completed its full-budget native development-panel run. The four-seed
   common-panel qualification and direct full-network, step-by-step PPO
   gradient/Adam comparisons **have not yet passed**.
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

## Fixed full-cohort evaluator (manual only)

A source-controlled `rust_robotics_train/examples/cleanrl_reference_cohort.rs`
runner is provided for the four exposed development seeds. It uses the
read-only, **64-lane batched** native evaluator (identical batch size across
completed/inactive lanes), preserving original physical rewards, bounded
rail/angle terminations and independently seeded action/actuation/observation
noise. The full training clock never resets between checkpoints.

```bash
cargo run --release -p rust_robotics_train --features cleanrl-reference \\
  --example cleanrl_reference_cohort -- 201 /tmp/native-seed201.csv
```

Fixed checkpoints at updates 0/64/128/256/512/768/976 correspond to
0/32768/65536/131072/262144/393216/499712 actual transitions. Two
predeclared panels of 64 reset keys each use independent key namespaces:
ordinary is a fixed 64-key prefix; difficult is selected solely by initial
physical-state thresholds (angle magnitude ≥ 0.15, angular-rate magnitude
≥ 0.25 and outward motion). Both deterministic and stochastic policy
outputs are evaluated for each checkpoint, with all physical failures,
returns, rail clearances and force switches stored in create-only CSV files.
This is **not** the previous Python evaluator's identical state/noise-key
panel or equivalent RNG generator; the results must not be presented as a
byte-matched Python-versus-Rust experiment. It is a new fixed task-native
panel for checking retained recovery as learning proceeds. The seed-201
development run completed, while the additional four-seed common-panel
qualification remains separate from routine CI.


## Frozen four-seed + common held-out protocol (opt-in)

The one-seed release workflow on commit `8d0d9d31` passed source gates
(17 targeted tests, formatting, strict Clippy), 976 updates and full CSV
verification. Seed 201's final development-panel completions were 64/64
ordinary and 63/64 difficult in **both** deterministic/stochastic modes.
This is a single development seed only, not four-seed reproducibility,
matching Torch parameters or held-out deployment readiness.

The optional `--shared-heldout` flag appends two **common** panels to
the unchanged existing seed-specific development panels. Every checkpoint
(0/64/128/256/512/768/976) evaluates each panel in deterministic and
stochastic modes for 1500 ticks and records **all** 64 individual outcomes.
The two new reset-key sets are identical across training seeds 201–204.
The ordinary panel starts at key `4_000_100_000`; the difficult panel is
selected from keys starting at `4_100_100_000` with the same predeclared
physics-only predicate: abs(angle) >= 0.15, abs(angular rate) >= 0.25, and
angle * angular rate > 0. No seed is dropped or checkpoint selected after
seeing learning outcomes. Held-out evaluations are read-only: their
observations/rewards never contribute to PPO optimization.

The separate create-once [four-seed workflow](../.github/workflows/cleanrl-native-four-seed.yml)
builds the release example **once**, trains the four seeds sequentially
without restarting each seed's Adam state between checkpoints, and validates
4 × 976 updates, 4 × 499,712 training transitions and 14,336 evaluation
records (4 seeds × 7 checkpoints × 4 panels × 2 modes × 64 resets). It
verifies the *same* common-panel reset keys across seeds, publishes all
per-seed CSVs and logs including incomplete failure evidence, and reports
the 512/768/976 checkpoint completion counts. The workflow has a 90-minute
cap, a one-off push-path trigger and manual dispatch; it is NOT in normal
PR CI. The **workflow conclusion means evidence completeness only**.
Whether the policy meets a sufficiently strict independent held-out success,
variance and retention threshold requires interpretation of the actual
results, and matching Torch gradients/Adam remains a distinct gate. No
merge/default change is authorized by a green workflow.
