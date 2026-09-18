# PPO boundary investigation — 2026-09-18

## Result and limits

**A concrete error was found in the reference comparison, not a demonstrated fix for the production learning failure.** C3's purported recipe-matched Stable-Baselines3 arms used Adam epsilon `1e-8`, while the actual Burn 0.20.1 production default is `1e-5`. Correcting this explains the large first-step mismatch and makes the explicit terminal and timeout boundary cases pass. The two longer default cases still fail the original consecutive-update tolerances. They are retained as failures; no tolerance was relaxed.

A supplementary conditional update replay agrees closely with the native updates, but resets its incoming parameters to the recorded native state. It is not free-running equivalence or a learned-policy result. Issues #35 and #33 remain unresolved. No production training equation/default change, merge, or sustained-balancing qualification is claimed.

## Provenance

- Production baseline: `ff4934a636974727632355371f575e651dc44a61`, the existing draft PR #37 plus the lockfile-audit workflow correction.
- Original experiment: `19ac962de95ac6d038fa8aeb00daaf3a57f7759a`, [run 35373817260](https://github.com/yongkyuns/RustRobotics/actions/runs/35373817260).
- Runtime-matched experiment: `dd39c023c3fc0462fcc63789f54c0d5ddbe52d5f`, [run 35374729957](https://github.com/yongkyuns/RustRobotics/actions/runs/35374729957).
- [Original pre-execution protocol](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5733581178) and [recorded configuration correction before rerunning](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5733722073).

The actual reference is SB3 2.9.0, Torch 2.8.0+cpu, NumPy 2.2.6, Gymnasium 1.3.0, Python 3.11. Rust is 1.98.1 with the committed lockfile. Actual installed SB3 collector, buffer, policy and PPO source files are archived.

`prepare.py` installs observation-only native hooks in a disposable checkout. An ordinary unobserved twin session must remain bitwise identical, including metrics and the next rollout. Protected production source paths and Cargo.lock are verified before preparation and after restoration. Only test-time hooks are inserted; native training arithmetic is unchanged.

## The configuration error

Production initializes both optimizers using `AdamConfig::new()`. [Burn v0.20.1 source](https://github.com/tracel-ai/burn/blob/v0.20.1/crates/burn-optim/src/optim/adam.rs) declares epsilon `1e-5`. The corrected run exports the actual compiled configuration:

```json
{"weight_decay":null,"grad_clipping":null,"beta_1":0.9,"beta_2":0.999,"epsilon":0.00001}
```

[C3's actual reference constructor](https://github.com/yongkyuns/RustRobotics/blob/a29060c3180cd603347fffafde98f0709054c9c2/audits/ppo_crosscheck/reference.py#L130-L149) instead explicitly supplied `optimizer_kwargs=dict(eps=1e-8)`. The original boundary replay inherited that mistaken assumption. `matched.py` now configures the constructor from the recorded native settings, retaining the original comparator and unmodified SB3 `collect_rollouts` and `PPO.train` methods. The original incorrect comparator remains reproducible with `compare.py` and is exercised as a negative control.

C3's measurements remain observations of the configurations actually executed. Its label "recipe-matched" was incorrect, so its performance gap cannot be interpreted as an implementation-only comparison. This correction does not establish that either epsilon yields better long-run task performance.

## Main consecutive-update replay

The native probe records actual observations, latents, rewards, terminal/timeout information, final-before-reset and next-after-reset observations, target/advantage tensors, exact minibatches, pre-step losses and post-step actor/critic weights.

The SB3 playback environment supplies the recorded observations, rewards and endings, and the distribution hook supplies only recorded action draws. SB3 computes its own network outputs, behavior log probabilities, timeout correction, GAE and updates. Its documented buffer adapter normalizes in native environment-major order and follows the native minibatch order. Weights are loaded only at initialization; Adam persists across updates.

**Important interpretation boundary:** observations, actions, rewards and minibatch order are fixed, but derived behavior log probabilities and targets are independently recomputed. Once the networks drift, these derived training inputs can also differ. This is not strict equality of all derived tensors across the complete free-running replay.

All original thresholds are retained: inference absolute/relative `2e-4/2e-5`; targets `5e-4/2e-5`; normalized advantages `1e-4/2e-5`; losses `5e-4/1e-4`; parameters `2e-4/0`.

| Case | Native transitions | Updates / optimizer transactions | Corrected result | First failed update | Maximum actor parameter difference | Maximum critic parameter difference |
|---|---:|---:|---|---:|---:|---:|
| Default seed 201, 1 x 512 | 4,096 | 8 / 128 | FAIL | 3 | 0.00610584 | 0.0000377223 |
| Default seed 204, 1 x 512 | 4,096 | 8 / 128 | FAIL | 4 | 0.00612972 | 0.000364449 |
| True-terminal control, seed 201, 3 x 7 | 84 | 4 / 24 | PASS | — | 1.34e-7 | 1.08e-7 |
| Timeout control, seed 201, 3 x 7 | 84 | 4 / 24 | PASS | — | 7.14e-8 | 1.23e-7 |

Default cases use batch128/four epochs. Boundary controls use batch8/two epochs, including the partial final batch of five. The timeout cap is three steps with high failure bounds; the terminal control sets the angle limit to zero. These are contract controls, not learning-quality environments.

Incorrect epsilon is detected at update 1; deliberate Adam resets with otherwise matched settings are detected at update 2. These do not turn the longer positive cases into passes. The first run failed all four cases. Both original and corrected runs completed their intended comparisons without a harness exception, and the corrected workflow correctly remains red.

## Supplementary conditional checks

The independent NumPy trajectory replay reproduces every native return and globally normalized advantage exactly in f32. The exported set contains 179 true-terminal paths, 27 timeout paths and 24 live-buffer cutoffs. This is verification of targets from recorded values/bootstrap estimates, not an independent validation of the physical model or critic accuracy.

A separate first-step Torch/closed-form Adam check reproduces the large mismatch at epsilon `1e-8`; using `1e-5` reduces the maximum first-step parameter discrepancy across all four cases to `6.90e-8`.

A local Torch diagnostic then feeds each recorded native minibatch and incoming parameter state through independent actor/critic losses and persistent Torch Adam moments. Across **304 actor/critic update pairs**, maximum conditional post-step differences are `2.98e-8` for the actor and `5.96e-8` for the critic. Its maximum policy-loss discrepancy is `2.47e-7`; maximum unweighted value-loss discrepancy is `3.05e-5` or less.

That diagnostic deliberately restores native parameters before each step while retaining optimizer moments. It therefore tests local update consistency without letting parameter errors feed back through subsequent gradients; it does not reproduce a free-running reference policy. It uses local Torch **2.10.0+cpu**, NumPy **2.3.5**, Python **3.13.5**, distinct from the pinned main experiment. Versions, scripts and per-step results are retained in the evidence bundle.

A second local mode retains free-running parameters but supplies exact native derived rollout tensors. Its maximum actor differences are `9.51e-5` and `1.31e-4` for default seeds201/204; the seed204 critic still reaches `3.64e-4`. This is supplementary evidence, not a substituted pass for the main SB3 test. Feedback amplification of small numerical differences is a plausible interpretation, not a uniquely established cause of the remaining free-running divergence.

## Verification, cost and retained evidence

Both primary runs pass strict native Clippy, 65 unit tests, all four observer/twin-session controls, formatting and protected-source restoration. The native JSON traces in the original and corrected runs are **byte-for-byte identical**. No cases, checkpoints or seeds were selected after observing outcomes.

The four recorded cases contain 8,360 physical training transitions and 304 actor-plus-critic optimizer transactions. Counting unobserved twin sessions and subsequent-rollout equality checks gives 18,852 native transitions per execution, excluding unit-test work. Reference playback consumes recorded data rather than independently interacting with a new physical environment. Repeated traces are not independent learning trials.

Raw evidence:

| Artifact | ID | Archive SHA256 |
|---|---:|---|
| Original failed comparison | 10559841260 | `2b7082dfdf9c4830a09af04560bb8e0324736893b6ea8ecfe6f88505a72eb4ce` |
| Runtime-matched comparison | 10560032269 | `8e6948cf6cd032ea8777096e92010f5e592507df6ab414bb19b713df00e3465e` |

Archive digests and all 44 original / 47 corrected manifest entries were verified. The self-contained conversation bundle `rustrobotics-ppo-boundary-evidence.zip` retains both unchanged archives, summaries, independent verification scripts, supplementary per-step results and provenance. Intermediate setup commits also triggered Actions runs; these are not additional independent experiments or favorable replacements.

## Next unresolved work

Do not retune production merely to make two numerical trajectories agree. The next bounded diagnostic should control all stored likelihood/target tensors through the real reference update, distinguish local gradient/optimizer consistency from free-running feedback, and retain the present failures. A separate epsilon-only, fixed-budget from-scratch ablation is needed before claiming epsilon improves balancing. Neither early boundary checks nor reused development seeds qualify held-out sustained balancing.

PR #37's lifecycle correction is a separate scope. Its exact head `ff4934a` passes native/platform, rebuilt-WASM/browser, numerical and configured PPO-learning checks; cargo-audit remains blocked by rustls0.23.44 / RUSTSEC-2026-0285. This investigation did not update Cargo.lock, suppress the advisory, approve or merge the PR. See [PR review record](https://github.com/yongkyuns/RustRobotics/pull/37#issuecomment-5733640579).
