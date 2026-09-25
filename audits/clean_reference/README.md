# Clean PPO reference comparison

September 25, 2026. This branch is an isolated diagnostic. It does not add a new production trainer or a task-specific learning intervention.

## Production corrections

PR #37, head `3de2e27663cf2d4630c61a8b6cddd17709140e12`, removes coordinator parameter averaging and implicit optimizer resets. One learner retains the actor, critic and both Adam histories. The optional environment pool collects under one frozen policy and feeds the ordinary PPO update; default collection is one environment. Snapshot publication/readout is not a learner mutation.

PR #38, head `95c670b9f4618a11dc5439b026caf385a39718c8`, includes #37 and one shared nonlinear cart-pole/PendulumEnv contract for training and live control. Current master `8514317e3b5cd57dc37bee2aa4c950af674751e1` and its stack-algebra migration are preserved. Reconciliation adapts vector construction and test references, not physics equations or PPO settings. Both PRs are ready for review, unmerged and undeployed at this report.

All four workflows on each current PR head pass. Combined tree `4a8e24061690e9d34fb1cd496e4b81cedea987fc` passes 109 algorithm/physics tests, 68 trainer unit tests, 7 balancing controls and 9 ordinary learning/evaluator controls in run36128017064. Two pre-existing expensive integration tests remain ignored. PR38 platform run36128207071 passes Linux/Windows/macOS, strict formatting/Clippy, documentation and rebuilt WASM/browser, including15 browser tests. Its tested synthetic merge6b35e65e80a8941a9dcaa4de853a233106ffb32a has the identical tree. Native success and browser parity are not evidence that sustained balancing is solved.

## Fixed comparison

Protocol: issue35/comment5831510896, registered before the new learning outcomes.

Executable/driver source: `da6623ae4b5878ae5079bf05c18c3d327570bc16`. Workflow36129074098. Underlying production source95c670b9f4618a11dc5439b026caf385a39718c8.

Three arms, four exposed development seeds201–204, all from scratch:

- `rust`: ordinary unchanged native PPO; one512-step environment, minibatch128, four epochs, gamma.99/lambda.95, clip.2, value coefficient.5, entropy0, learning rate3e-4, ReLU64x64 and fixed pre-squash force-scale2.0 (latent standard deviation.1).
- `matched`: actual SB3 PPO.train and collect_rollouts, the same randomly initialized actor/critic weights and recipe. Whole-rollout population advantage normalization is applied after SB3's own return/GAE computation. Log standard deviation is frozen; gradient norm clipping is disabled. No hand-written Python PPO optimizer is substituted.
- `sb3-defaults`: stock SB3 continuous-action PPO using normalized force actions, learned standard deviation and its default update/initialization/activation choices. This is a package-level comparison, not a single-variable experiment or a proposed default.

All use the actual same noisy nonlinear Rust PendulumEnv through the checked C ABI. No Python approximation of the dynamics. No LQR initialization, pretrained critic, oracle labels, selected trajectories, reset curriculum, extra future-support collection, policy rollback, or checkpoint selection.

Each arm/seed uses exactly1,048,576 training interactions. Checkpoints are0,65536,262144,1048576. Evaluation is the same fixed ordinary32 deterministic10-second and64 stochastic15-second episodes per checkpoint with independent keys. Rust evaluation uses native snapshot inference, not a claim of bit-identical Torch inference. These are short learning panels, not five-minute stability qualification. Stock SB3 uses more optimization work; equal interaction count does not mean equal compute.

The previous C3 study was on the old linear plant and used epsilon1e-8 in its nominally matched SB3 arm. Do not reuse it as a current-task exact recipe match. This build serializes actual Burn AdamConfig and requires epsilon1e-5 in both implementations. The new generated reference includes this explicit check and retains the original matched initial weights and ordinary update equations.

## Preflight completed

Native job108051887081 succeeded before the measurement matrix. It passes74 Rust tests including the unchanged68 production unit tests and six ABI/configuration/readout controls, plus the existing learning/evaluator integration controls. The independent Python controls pass real Gymnasium environment checks, ABI/noise replay, whole-rollout normalization, GAE terminal/cutoff handling, current-policy ratio equivalence, zero-update preservation, evaluation RNG immutability, matched two-update replay and native/stock learning smoke tests.

Actual serialized Adam: beta1=.9, beta2=.999, epsilon=.00001, no weight decay, no gradient clipping. Reference dependencies are Torch2.8.0+cpu, SB3 2.9.0, Gymnasium1.3.0, NumPy2.2.6 and Python3.11; actual source modules and freeze output are retained.

Build artifact10860907319 (`clean-ppo-native`), ZIP SHA256 `58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0`. All142 manifest-listed build payloads have been independently verified after download. It includes the exact generated bridge/reference code, source provenance, checked binary, dependency versions, Adam receipt, preparation diff and test logs.

## Status and recovery

At this report, all four Rust measurements completed, all four matched SB3 measurements were training, and the stock SB3 measurements were queued. The reference comparison is not complete. Retain every arm/seed/checkpoint and every contrary result. Do not substitute an incomplete cohort for the registered twelve runs.

Completed Rust artifact IDs:201=10861835890,202=10861456056,203=10861765844,204=10860937552. These results contain384 episode outcomes each, checkpoints, original native actor/critic bytes, complete native update-loss/cost history, and the exact qualified build manifest. Their raw records confirm that the ordinary recipe still has failures; architecture correction is not being presented as reliable learning.

Production source restoration is an explicit workflow gate. The diagnostic adds only disposable cdylib/ABI wiring during compilation; protected trainer/GAE/model/environment/optimizer sources remain identical to95c670b. The preparation/workflow files are never merged into the production PRs.

GitHub artifacts expire. Preserve the downloaded originals and exact generated source when retaining the study. Source preparation is derived from immutable reference commit a29060c3180cd603347fffafde98f0709054c9c2 and additionally verifies its retained artifact manifest; the generated actual source is in the new build artifact. Outcome accounting must validate source identity, all manifests, exact budgets and initial weights, complete episode keys/endings, and all twelve runs before any full-cohort interpretation. Four training seeds, not pooled evaluation episodes, are the sampling unit for training comparisons. No production adoption is authorized by this diagnostic alone.
