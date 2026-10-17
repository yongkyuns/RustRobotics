# Issue #42 — Frozen matched native PPO action-distribution comparison

This is an **experimental development cohort**, not an independent acceptance
test and not a patch to production continuous Gaussian PPO. Keep PR #41 draft.

## Matched quantities

Both arms use the same underlying 4-observation native noisy nonlinear plant,
reward, RK4 integrator (10 ms), observation and actuation noise, termination,
four independent training environments, 128 samples/environment/update, 976
updates (499,712 training transitions) per seed, 64×64 tanh actor and critic,
two actor outputs, same seeded initialization, one persistent Adam epsilon
1e-5, single global gradient norm 0.5, batch 128, 4 epochs, gamma .99,
GAE lambda .95, clipped actor/critic loss, entropy coefficient .01 and linear
learning-rate annealing from 2.5e-4. No evaluation resets or held-out rewards
enter optimization.

All four existing exposed development seeds 201–204 are run with both policies.
Four fixed evaluation panels (per-seed ordinary/difficult plus shared ordinary/
difficult) each contain 64 keys; 1500 ticks; deterministic and stochastic
inference; checkpoints 0/64/128/256/512/768/976; no checkpoint selection.
All failed episodes and intermediate snapshots of *evaluation measurements*
are retained. No model serialization/resume is claimed.

## Single comparison dimension, with unavoidable mathematical differences

**Categorical:** two actor logits, categorical action -20 N or +20 N, log
softmax likelihood and Shannon entropy.

**Continuous Gaussian:** exactly two actor outputs denote mean and log std
of pre-tanh scalar Gaussian latent z. Clamp logstd [-2, 1] for numerical
stability; physical action = 20*tanh(z) N. Store z to compute PPO probability
ratios from its Gaussian log density: the tanh Jacobian cancels for the same
stored z. Entropy regularization uses Gaussian pre-tanh differential entropy
rather than exact squashed-action entropy. Therefore an equal coefficient 0.01
does not imply equal exploration pressure measured in action units.

The two models start from matching neural weight tensors for each seed, but
the different sampling laws consume RNG differently and trajectories diverge.
This ablation holds the **PPO recipe** fixed and changes policy law; it does
not make categorical/continuous entropy identical or explain historical
production Gaussian PPO regressions by itself.

## Evidence criteria

The opt-in workflow passes only when both distributions complete all four
budgets, all 28,672 CSV evaluation records and common reset-key comparisons
are verified, and the categorical final common-panel successes replicate
the prior four-seed results. Loss of balancing by either arm must appear in
the records even if all CI checks pass. Statistical confirmation would need
new training/evaluation seeds and predeclared criteria under #33. Production
integration and a matched **original** continuous PPO baseline remain separate.
