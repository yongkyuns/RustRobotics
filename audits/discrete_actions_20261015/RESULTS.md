# Discrete left/right PPO: stronger final balancing, but regressions remain

October 15, 2026. RustRobotics issue #35. Completes the before-outcome protocol [6200292905](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6200292905).

## Finding

A genuine categorical left/right policy learns the native nonlinear cart-pole task and finishes with higher aggregate difficult-case completion than the continuous comparator. It does **not** eliminate learn-then-collapse: one discrete seed falls from 62/64 to 38/64 difficult stochastic completions before partially recovering, and three discrete seeds lose previously successful recoveries during the final continued-training stage.

This is a completed diagnostic, not a production fix or a claim that the task is now reliably solved. Both arms use the same stock SB3 PPO implementation against the actual native Rust plant. The continuous arm is the existing normalized, constant-learning-rate, clipped-Gaussian reference, **not a new measurement of the production Rust/Burn tanh-Gaussian trainer**. A categorical policy has not been added to the production Rust trainer or browser training UI.

## Final results

All eight from-scratch runs finished once, with exit code zero. Four exposed development seeds, 201–204; each arm receives exactly 524,288 training interactions. Each cell below is completed 15-second episodes out of 256: 64 cases for each of four separately trained policies.

| Evaluation panel | Continuous | Discrete left/right |
|---|---:|---:|
| Ordinary, deterministic | 254/256 | 255/256 |
| Ordinary, stochastic | 254/256 | 254/256 |
| Difficult, deterministic | 239/256 (93.36%) | 253/256 (98.83%) |
| Difficult, stochastic | 238/256 (92.97%) | 244/256 (95.31%) |

Ten-second survival counts happen to equal these 15-second counts at the final checkpoint; no claim about longer-than-15-second survival follows. These are four already exposed training seeds and evaluation panels, not a new held-out qualification. Evaluation episodes are not additional independently trained agents. The measured aggregate advantage is not proof of population-wide superiority.

Final difficult-case results, each out of 64:

| Seed | Continuous deterministic | Discrete deterministic | Continuous stochastic | Discrete stochastic |
|---:|---:|---:|---:|---:|
| 201 | 61 | 64 | 62 | 62 |
| 202 | 64 | 62 | 64 | 61 |
| 203 | 60 | 63 | 59 | 63 |
| 204 | 54 | 64 | 53 | 58 |

Seed 202 is a contrary result: the continuous policy outperforms the discrete policy in both modes. No seed is excluded.

## Learning and continued training

Difficult-case completions at every fixed checkpoint, all denominators 256:

| Training interactions | Continuous deterministic | Discrete deterministic | Continuous stochastic | Discrete stochastic |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 65,536 | 250 | 122 | 204 | 122 |
| 131,072 | 250 | 228 | 237 | 217 |
| 262,144 | 241 | 250 | 234 | 228 |
| 524,288 | 239 | 253 | 238 | 244 |

Continuous control learns faster in the early checkpoints. At the primary 262,144 endpoint, discrete control is better in aggregate deterministic completion but worse in stochastic completion. The additional 262,144 interactions change that final comparison; no earlier checkpoint was selected for deployment.

The clearest non-monotonic discrete run is seed 204: difficult stochastic completion is **62/64 at 131,072, 38/64 at 262,144, and 58/64 at 524,288**. Two fixed actions do not prevent a large observed recovery loss.

Between 262,144 and 524,288 interactions, discrete seed 201 loses two difficult stochastic cases, seed 202 loses one, and seed 203 loses one in each evaluation mode. None of those panels gains a case to offset its loss. Seed 204 gains four deterministic and twenty stochastic cases during this stage. Consequently, an improved aggregate final score does not establish casewise retention. Complete gained/lost keys are retained in RETENTION.json.

## What changed, and what did not

DISCRETE uses a genuine two-logit categorical PPO policy. Index 0 maps to normalized command -1, hence -20 N; index 1 maps to +1, hence +20 N. There is no neutral action, Gaussian sign rounding, force-magnitude search, teacher, safety override, curriculum, rollback or extra optimizer pass.

Both arms preserve the original nonlinear native physics, noisy observations, disturbed actuation, reset distribution, rewards, physical termination and 10 ms timestep. The external training timeout is 256 steps; evaluation runs to first physical failure or 1,500 ticks. The actor never receives privileged physical state. Common actor hidden tensors and the complete initial critic are paired across arms. The action heads, initial action variance and sampling laws necessarily differ; identical initial policies are not claimed.

Both retain the existing reference recipe: observation-only VecNormalize, training-only statistics frozen at evaluation, reward-unit scale 1/1000 with divided initial critic head, gamma .999, GAE lambda 1, 2,048 rollout rows, batch 64, ten epochs, constant learning rate 3e-4, Adam epsilon 1e-5, clip .2, global gradient norm .5, entropy coefficient zero and two 64-unit tanh layers. Adam history persists across all checkpoints. This is the established reference recipe, not unmodified SB3 default hyperparameters.

The original 64 ordinary and 64 difficult reset cases per seed are reused completely, in deterministic and stochastic modes. Each evaluator uses a fixed 64-lane inference batch, including inactive lanes. Native reset/noise keys are paired. Gaussian innovations and categorical uniforms are each fixed across checkpoints; different action distributions do not share identical sampled actions. Evaluation asserts that policy, optimizer, normalizer, training RNG, live observations and training counters remain unchanged.

## Control effort and reward

The unchanged nonterminal reward includes `-0.001 * force^2`. Fixed +/-20 N therefore pays **0.4 reward units per nonterminal step**, even when balancing perfectly; true terminal reward remains -10. The penalty was not removed or retuned.

At the final difficult stochastic checkpoint, mean physical return is 1289.93 continuous versus 783.72 discrete, despite the higher discrete completion count. Average per-trajectory commanded-force RMS is 4.89 N continuous versus 20 N discrete. Mean command-sign switching is approximately 46.36/s continuous versus 63.35/s discrete. These are descriptive simulation measurements, not actuator or hardware qualification. Force records denote commanded force, not the hidden noisy applied-force realization.

Thus the discrete result is stronger final balancing on this development panel, **not better control effort, higher reward, faster early learning, or established reliable continued learning**.

## Verification

Seven preflight controls pass: exact 4,096-step continuous-baseline reproduction; all four shared-initialization pairs; exact paired native dynamics/noise/reset/ending behavior for explicit left/right commands; categorical log-probability, entropy and gradient formulas; invalid-action rejection; actual categorical learning with persistent Adam; and frozen evaluation with exact original continuous-evaluator raw-array reproduction.

Fifteen independent NumPy reader/genuine-fixture/corruption tests and ten reusable adapter-contract tests pass. The full reader checks **160 evaluation groups, 10,240 trajectories and 11,380,205 physical evaluation transitions**, including original reset keys, first endings, zero padding, float32 physical reward bits, command/force mapping, noisy-observation bounds, normalizer/source/checkpoint identities and all named Adam clocks. Discrete actions match an independent float64 categorical forward calculation exactly. Maximum continuous normalized-command discrepancy is 3.566752220240943e-7, below the unchanged 3e-6 bound.

Every continuous policy, Adam export and normalizer at the four checkpoints shared with the preceding constant-rate LR study matches that retained study exactly: 48 object comparisons across four seeds. This supports preservation of the reference; it is not new independent population evidence.

The reader does not independently reintegrate native physics or regenerate private native noise. Policy/Adam/normalizer exports are not complete native-simulator-resume checkpoints. The full offline replay needs only the retained arrays plus NumPy, not Torch, native execution, an optimizer or pickle.

## Costs and retained setup failures

Main training: **4,194,304 native interactions and 655,360 Adam transactions**. Main evaluation: **11,380,205 native transitions**. Preflight attempts add 20,480 training interactions, 2,400 native mapping transitions and 16,339 evaluation transitions: **15,613,728 total native transitions**, including setup. No main run was restarted or dropped.

The first preflight stopped before main training on a bit-equality assertion between two algebraically equivalent float64 log-probability formulas. Before main execution this was changed to an explicit 1e-14 absolute/relative comparison, matching the other categorical mathematical checks. The failure log and costs are retained; no physical reward, completion rule, budget or physical-reader tolerance changed.

A self-contained HTML replay contains the first three accepted cases, chosen without outcomes, for both final-stage checkpoints, all seeds, panels and modes. It replays recorded native traces and does not train or resimulate physics. JavaScript syntax checks pass. Interactive browser verification did not complete: the container Chromium blocked the local file URL with ERR_BLOCKED_BY_ADMINISTRATOR. No browser pass, screenshot, public hosting or deployment is claimed.

## Source and disposition

Exact executed collector Git blob: `41675eba1b56a9b016c891ce4833210f3684cc62`; SHA256 `7774d3191a92239560bd872528797a77ae6807b78484b98aecbd306aa5aad7b5`. Published under `audits/discrete_actions_20261015/code/study.py` on `audit/ppo-discrete-actions-20261015`, together with the reusable adapter, tests and execution README. Complete local source, inherited native inputs, independent reader, reports, checkpoints and traces are delivered in evidence archives rather than claimed all committed to GitHub.

Native archive SHA256 `58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0`, all 142 payload hashes checked. Production plant base PR #38 head `95c670b9f4618a11dc5439b026caf385a39718c8`; diagnostic native bridge source `da6623ae4b5878ae5079bf05c18c3d327570bc16`. Pinned runtime Python 3.11.14 / Torch 2.8.0+cpu / NumPy 2.2.6 / SB3 2.9.0 / Gymnasium 1.3.0 on AMD EPYC 9V74. Runtime archive SHA256 `c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1`, all 11 payloads checked.

No production Rust categorical implementation, default change, protected-file edit, master change, PR merge, deployment or issue closure. This experiment narrows the conclusion: continuous force magnitude is not required for good balancing, but removing it alone does not remove the observed PPO regressions.
