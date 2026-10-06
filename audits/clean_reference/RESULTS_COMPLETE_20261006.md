# RustRobotics PPO: completed clean-reference comparison

Analysis date: October 6, 2026. Measurements were executed September 25, 2026.

## Conclusion

The three-arm, twelve-run comparison is complete. No tested recipe qualifies reliable sustained balancing. Stock SB3 learns substantially better early behavior than the Rust-matched recipe, but loses that behavior during continued training in every one of the four exposed development seeds. This is not evidence that a new recovery curriculum is needed, nor proof of a Rust-specific optimizer defect. The next learning investigation should localize degradation in the clean training path, without adding task-specific training mechanisms.

No training, production-default change, merge, or deployment was performed in this continuation. All results below are reductions of the original fixed-run artifacts; all seeds, checkpoints, and failures are retained.

## Fixed experiment and source

Registered protocol: [issue #35, comment 5831510896](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5831510896).

Original Actions run: [36129074098](https://github.com/yongkyuns/RustRobotics/actions/runs/36129074098), executed source `da6623ae4b5878ae5079bf05c18c3d327570bc16`, production base `95c670b9f4618a11dc5439b026caf385a39718c8` (PR #38). All arms use the same corrected, noisy, native nonlinear PendulumEnv through the checked ABI. Seeds 201–204 start from scratch and receive exactly 1,048,576 training interactions each. The fixed checkpoints are 0, 65,536, 262,144, and 1,048,576 interactions. Each checkpoint has 32 deterministic 10-second and 64 stochastic 15-second evaluation episodes per training seed, with separate recorded evaluation keys. Rust evaluation uses the native exported policy calculation.

The matched SB3 arm uses the same initial actor/critic weights, ReLU 64x64 networks, fixed latent standard deviation 0.1, 512-step rollouts, 128-row minibatches, four epochs, whole-rollout normalization, Adam epsilon 1e-5, and no gradient-norm clipping. Stock SB3 is a package-level contrast: its own initialization, tanh 64x64 networks, learned standard deviation, 2,048-step rollouts, 64-row minibatches, ten epochs, minibatch normalization, and gradient-norm limit 0.5. It is not a one-variable intervention. The archived SB3 version is 2.9.0; Torch is 2.8.0+cpu, Gymnasium 1.3.0, and NumPy 2.2.6.

## Complete endpoint results

All rows use the prescribed 1,048,576-interaction endpoint, not a selected best checkpoint.

| Training seed | Rust deterministic /32 | Matched SB3 deterministic /32 | Stock SB3 deterministic /32 | Rust stochastic /64 | Matched SB3 stochastic /64 | Stock SB3 stochastic /64 |
|---|---:|---:|---:|---:|---:|---:|
|201|22|13|19|46|34|44|
|202|23|14|13|48|36|32|
|203|17|24|12|35|43|14|
|204|28|11|18|55|15|42|
|Total|90/128|62/128|62/128|184/256|128/256|132/256|

| Arm | Deterministic mean return | Deterministic mean discounted return | Stochastic mean return | Stochastic mean discounted return |
|---|---:|---:|---:|---:|
|Rust|682.834|79.434|1033.123|78.697|
|Matched SB3|546.951|79.775|780.887|78.821|
|Stock SB3|476.636|75.203|760.458|76.542|

Matched SB3 has slightly higher mean discounted return than Rust despite substantially fewer completions. Completion, undiscounted return, and discounted return are different measures; none is silently substituted for another. Final stock-SB3 completion counts are below Rust in each of these four seeds, but all final paired-arm descriptive 99% training-seed t intervals span zero. These observations establish neither general superiority nor equivalence.

## All fixed checkpoints, including contrary results

| Interactions | Rust det /128 | Matched det /128 | Stock det /128 | Rust stoch /256 | Matched stoch /256 | Stock stoch /256 |
|---|---:|---:|---:|---:|---:|---:|
|0|0|0|0|0|0|0|
|65,536|0|0|122|0|0|236|
|262,144|17|1|95|23|0|173|
|1,048,576|90|62|62|184|128|132|

Stock SB3's early result is important: this task is learnable with an ordinary independent PPO package on these development seeds. That does not turn the early checkpoint into a qualified supported policy or a license for evaluation-selected early stopping.

| Stock seed | Deterministic: 65,536 -> 262,144 -> endpoint | Stochastic: 65,536 -> 262,144 -> endpoint |
|---|---|---|
|201|29 -> 32 -> 19|60 -> 63 -> 44|
|202|31 -> 27 -> 13|60 -> 51 -> 32|
|203|31 -> 4 -> 12|58 -> 1 -> 14|
|204|31 -> 32 -> 18|58 -> 58 -> 42|

Seed 203 partially recovers from its intermediate collapse. Seeds 201 and 204 improve or hold between the first two measured checkpoints before degrading. We do not claim monotonic deterioration at every update.

Comparing the same recorded evaluation keys at 65,536 interactions and the endpoint, stock SB3 loses 60 deterministic and 104 stochastic previously successful cases, and gains none in either panel. The lost cases end at angle limits in 86 cases and position limits in 78; these 164 panel cases are not 164 independent training runs.

Every stock-SB3 seed also loses mean discounted return in both evaluation modes over that comparison. Pooled deterministic discounted return falls 85.411 -> 75.203 and stochastic discounted return 81.423 -> 76.542. Thus the observed degradation is not solely a deterministic-completion versus stochastic-discounted-objective mismatch.

The four-seed mean deterministic completion change is -46.875 percentage points, with a nominal descriptive 99% t interval [-85.595, -8.155]. The stochastic change is -40.625 points, interval [-101.163, 19.913]; the unbounded t interval is reported without clipping. Discounted-return intervals span zero despite all four observed seed changes being negative. The 65,536-to-endpoint emphasis is a post-hoc descriptive localization, not a new predeclared acceptance test. All adjacent checkpoint contrasts are also retained in results.json.

Learned normalized-action standard deviations at 65,536 -> endpoint are 0.250245 -> 0.042006, 0.230965 -> 0.020595, 0.256208 -> 0.030175, and 0.231475 -> 0.034262 for seeds 201–204. This is an observation, not a causal attribution. The actor changes as well, and deterministic outcomes deteriorate too.

## Compute is not matched for the stock arm

At the endpoint, Rust and matched SB3 each make 4,194,304 sample visits per network and 32,768 minibatch updates per network. Stock SB3 makes 10,485,760 sample visits per network and 163,840 minibatch updates. SB3 uses one optimizer over both networks, while Rust has separate actor/critic optimizers. Equal environment-interaction counts do not imply equal training compute. No runtime-speed ranking is claimed.

## Validation and reproducibility

A new read-only analyzer downloaded and verified all 13 original ZIP digests against GitHub artifact metadata and all 434 manifest-listed payloads, including the 142 build payloads. It validates all 4,608 evaluation records, all twelve run identities, four fixed budgets, exact driver/library-source receipts, actual native Adam epsilon, matched initial weights including Torch orientation, checkpoint policy shapes/standard deviations, complete native update histories, reference episode continuity, finite outcomes, evaluation keys, and physical episode boundaries.

The analyzer uses no new PPO implementation, native training, critic fitting, oracle labels, or checkpoint selection. Its 28 actual-data/corrupted-input tests pass. A fresh extracted evidence package repeats the analysis byte-for-byte and passes all 28 tests. These are validation/replay checks, not additional independent training runs. The analysis environment is Python 3.13.5 and NumPy 2.3.5, separate from the archived training environment.

The conversation package contains the unmodified original ZIPs, analyze.py, test_analyze.py, results.json, aggregate/per-seed CSV files, logs, checksums, and reproduction commands. Original artifact IDs are listed in ARTIFACTS.json and embedded with their verified SHA256 values in analyze.py. Artifacts have finite retention; do not assume another chat has this continuation's sandbox paths.

## Production integration review

PR #37 head `3de2e27663cf2d4630c61a8b6cddd17709140e12` and PR #38 head `95c670b9f4618a11dc5439b026caf385a39718c8` remain open, non-draft, mergeable, and unmerged. The base is `8514317e3b5cd57dc37bee2aa4c950af674751e1`. Existing current-head Rust/platform, numerical, configured-learning, and dependency-audit workflows are completed successfully for both PRs; these are the September 25 executions, rechecked October 6, not new test runs.

Reviewed the combined diff's shared nonlinear physics, trajectory-local GAE and pooled normalization, persistent optimizer ownership, native/browser lifecycle, configured-plant propagation, stale-policy invalidation, and accompanying regression tests. No new blocking source-review finding was identified. This is the implementing assistant's review, not independent approval or formal verification. The integration fixes need not be held back as though they were an unproven learning trick: they repair demonstrated contracts, and #38 already contains #37. Any eventual merge must preserve the qualified combined source tree and recheck resulting-head CI if it changes.

No production policy, reward, training recipe, protected MuJoCo/control file, or master branch was modified here. #33 and #35 remain learning-acceptance work, independent of merge readiness.

## Next bounded investigation

Localize the clean-path loss of acquired behavior, starting from these retained checkpoints and all lost-case counterparts. An evaluation-only trace replay can identify physical failure sequences without changing training. Causal update attribution then requires continuing or exactly replaying the original seeded training prefix with actor, critic, Adam, environment and RNG state intact; weight-only checkpoints are not exact-resume training checkpoints. Distinguish critic/action-credit error from policy changes that damage behavior outside the fitting batch. Historical linear-task witnesses must not be treated as proof of the same cause on this corrected nonlinear task.

Do not add a new curriculum, selected fitting rows, extra critic-repair phase, policy rollback, or another broad hyperparameter sweep merely because an endpoint is poor. No new replay or training job has been launched by this report. A supported default still requires a fixed recipe and new, predeclared held-out sustained-balancing qualification.
