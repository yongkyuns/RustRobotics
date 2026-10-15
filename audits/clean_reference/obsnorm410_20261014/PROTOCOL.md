# Observation normalization: fixed from-scratch development screen

Protocol: issue #35 comment 6181775559. This is a new same-host paired cohort, not a replay of the historical AMD training trajectory and not issue #33 held-out qualification.

## Question

Can standard training-observation normalization improve ordinary vanilla-PPO learning and retained recovery clearance without changing the task, physical reward or training algorithm?

Compare the unchanged scaled-2048 reference recipe with stock SB3 VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10, epsilon=1e-8). Statistics come only from training observations, are frozen during evaluation, and are saved alongside every model. No reward shaping, privileged actor state, teacher, recovery curriculum, selected data, rollback or extra critic/optimizer pass.

## Frozen recipe and budget

Development seeds 201, 202, 203, 204; both arms start from equal freshly initialized actor/critic tensors. Each runs exactly 1,048,576 interactions. Checkpoints: 0, 65,536, 262,144, 524,288, 1,048,576. No best-checkpoint selection or outcome-adaptive extension.

Retain original nonlinear native plant, noisy observations and disturbed actuation, clipped Gaussian actions, physical rewards and failures; 256-step external training timeout; inherited reward-unit scale 1/1000 and divided initial critic head; gamma .999, lambda 1, 2048 rollout rows, batch 64, ten epochs, learning rate .0003, Adam epsilon .00001, clip .2, gradient norm .5, entropy coefficient zero, two 64-unit tanh layers. Use original SB3 PPO.train, collector and GAE methods. The reference is the previously investigated SB3 recipe, not a claim of updated production Rust defaults.

Four concurrent process slots; within each seed, raw then normalized for odd seeds and normalized then raw for even seeds. The ordering is fixed before outcomes, and all numerical kernels use one thread.

## Evaluation

Freeze all source panels before main training. Per training seed and checkpoint: 64 ordinary and 64 conditional difficult original native resets, each in deterministic and stochastic mode, 1500 ticks, no critic bootstrap. Difficult predicate: abs(theta)>=.15, abs(omega)>=.25, theta*omega>0. Select the first 64 qualifying ascending keys, at most 65,536 considered. Base key = 4200142600 + training_seed*1000000 + panel_index*100000. Stochastic draws = default_rng(key XOR 0x5E1410AA).standard_normal(1500).astype(float32). Same keys/draws across arms/checkpoints, 64 fixed inference lanes including inactive lanes.

Record raw and gamma .999 return, completion, signed minimum cart clearance including initial state, successful-case clearance quantiles, and completion with at least 50mm clearance. The last metric is a declared descriptive readout, NOT a changed physical boundary or a qualified safety specification. Failed cases remain in denominators. No evaluation observation updates normalization statistics.

## Development progression rule

At the final checkpoint require all of: no reduction in either difficult-mode completion count in any seed; no reduction in either ordinary-mode mean raw return in any seed; and at least five percentage points mean improvement in difficult completion-with-50mm across seed/mode pairs. A pass only makes a candidate promising for subsequent qualification. It does not authorize a default change, deployment or closure of #33/#35. Report all failures/checkpoints and use training seeds, not trajectories, for uncertainty.

## Controls and scope

Before main runs, require same-host exact raw versus monitored policy/Adam/RNG/live-state replay; disabled-normalizer identity; independent running-statistics, timeout-final-observation, unchanged-reward and frozen-evaluation controls. Validate recorded physical reward bits, action inference, endings, padding, source identities and margin arithmetic independently. Failures are retained rather than repaired by changing acceptance criteria.

The current host is Intel Xeon 8573C. New paired construction avoids falsely calling the current results exact historical AMD measurements. Software remains pinned to Python3.11.14, Torch2.8.0+cpu, NumPy2.2.6, SB3 2.9.0 and Gymnasium1.3.0, with original native and SB3 source checks.

No production trainer/default, master, reward/noise/reset/physics, protected MuJoCo/control source or deployment change is part of this audit. Main training outcomes were not observed when the protocol was registered. Executed source, inputs, controls and all result records will accompany the report; this protocol file is not an assertion that the measurements have completed.
