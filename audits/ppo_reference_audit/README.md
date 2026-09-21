# Independent PPO correctness audit

This audit is a correctness check, not another learning/tuning experiment.

It uses the immutable seed 41006 regression-window artifact from run 35554192268 and focuses on the independently confirmed harmful update 4140. No policy is retrained and no reward, gamma, lambda, sampling mixture, normalization, learning rate, clipping rule, or production source is changed.

The reference implementation is deliberately independent of Burn and the Rust PPO implementation. It uses NumPy only for dense linear algebra and implements the following from the recorded bytes/data:

1. actor and critic snapshot decoding and forward inference;
2. tanh-squashed Gaussian log probabilities;
3. the exact PPO clipped surrogate;
4. critic MSE;
5. GAE/return recursion from the recorded physical rollout rows;
6. global advantage normalization, including a sequential-f32 reconstruction;
7. analytic MLP backpropagation for actor and critic losses;
8. finite-difference directional checks against the independent analytic gradients.

For all 16 minibatch transactions in update 4140 it compares the independently reconstructed pre-update policy/value losses with the Rust-recorded losses, and independently checks the post-update exported network means against the recorded Burn means. It also checks old likelihoods against the incoming actor and tests whether the action-space likelihood ratio agrees with the algebraically equivalent latent-Gaussian ratio.

This audit does **not** claim to reproduce the Adam parameter update from weight files alone: Adam moments are not exported by the historical artifact. It therefore does not infer or fabricate them. A passing result establishes the forward/loss/target/gradient semantics around the update; it does not prove that every optimizer-internal state is correct. Any unexplained mismatch is retained as a candidate bug and stops the clean-pass classification.

The workflow additionally runs the repository's existing PPO/native tests on the pinned corrected plant source. Results are written to an artifact and are not merged into production automatically.

Known simulator/UI issues discovered separately (running-trainer config propagation, classical-vs-PPO noise equivalence, and deployment timeout semantics) are intentionally not used to explain update 4140 because the captured audit transaction bypasses those UI/runtime paths.
