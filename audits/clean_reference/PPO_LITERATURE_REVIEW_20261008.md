# PPO learning failures: literature, local evidence, and next experiments

**Research note — sources checked October 8, 2026.** Related work: [#35](https://github.com/yongkyuns/RustRobotics/issues/35); sustained-control acceptance: [#33](https://github.com/yongkyuns/RustRobotics/issues/33). This document does not change the trainer, environment, reward, production defaults, or acceptance rules.

## Executive conclusion

The observed symptoms have close precedents in policy-gradient research: noisy update directions, inaccurate values despite good fitting loss, surrogate/return disagreement, and sensitivity to scaling and optimization details. They are not evidence, by themselves, of a Rust arithmetic defect or an unusually hard physical plant. The literature does **not** establish their prevalence in our exact noisy continuous cart-pole task or identify one cause for every local failure. Do not turn a familiar symptom into a diagnosis without a matched test. [R1–R5]

The most directly useful methodology is Ilyas et al.'s separation of **gradient quality, value calibration, and actual reward change**. In this repository these must also be separated by routine balancing versus early recovery. Keep the negative larger-batch result and reasonably calibrated surviving-timeout tails visible; they contradict overly broad explanations.

### How the sources map to the recorded evidence

| Local observation | Relevant sources | Supported interpretation and limit |
|---|---|---|
| Raw tanh critic saturates; value scaling helps from initialization | R2, R4, R7 | Scaling can alter conditioning and optimizer dynamics. Neither paper proves our precise saturation mechanism. |
| A PPO surrogate improves while real control worsens | R1, R3, R5 | A known empirical possibility; ratio clipping is not a hard per-state trust-region constraint. |
| Independent 2,048-row rollout estimates disagree on directional-credit sign | R1, R6 | Examine estimator variance and independent trajectories, not just sample-row count. An interval spanning zero does not prove the true mean is zero or variance caused the failure. |
| Increasing n_steps to 8192 does not pass held-out superiority/early-learning screens | R1, R4 | More data per update is not automatically better learning; update cadence and data freshness also change. |
| About 85% of difficult-timeout targets comes from estimated continuation | R6, R8 | Correct external-truncation bootstrapping may legitimately dominate a target; test calibration rather than declare the tail fictitious. |
| Persistent LQR recovery succeeds where one-action substitutions fail | R6, R12 | An action's advantage depends on its continuation policy. LQR feasibility is not an oracle advantage label for PPO. |
| Average return improves but some previously successful cases fail | R9, R13 | Expected return, completion probability, and low-tail risk are different metrics. |
| Gaussian samples are clipped before applying force | R5, R10 | Distinguish latent-action likelihoods from boundary probability mass; a relevant variance-reduction lead, not a proven bug. |

## Annotated primary reading list

### R1 — A Closer Look at Deep Policy Gradients

**Andrew Ilyas et al., ICLR 2020** (preprint first submitted 2018). [Paper](https://arxiv.org/abs/1811.02553v4); [full text](https://arxiv.org/html/1811.02553v4); [conference record](https://iclr.cc/virtual_2020/poster_ryxdEkHtPS.html).

Sections 2.1–2.3 experimentally separate noisy gradient estimates, value-target fitting versus return prediction, and surrogate versus true-return landscapes. Their high-sample reference is an empirical approximation, not analytic ground truth. The MuJoCo examples are not our cart-pole environment.

**Use here:** compare the same frozen parameter direction against independently sampled return changes, with separate state strata. Measure uncertainty across independent rollouts/future draws. Do not equate lower critic training error with calibrated continuation values or assume a larger batch alone repairs the objective. This is the closest methodological match, not proof that every local witness shares one mechanism.

### R2 — Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO

**Logan Engstrom et al., ICLR 2020.** [Paper](https://arxiv.org/abs/2005.12729v1); [full text](https://arxiv.org/html/2005.12729v1).

The ablations show that implementation-level choices can substantially affect performance and the effective behavior of the algorithm. Their studied details include reward scaling, initialization, value clipping, learning-rate schedules and joint gradient clipping.

**Use here:** treat our fixed reward/value scaling as an intervention in optimization, not merely a change in displayed loss units. Shared actor/critic norm clipping means changing critic-gradient scale can also change actor optimization. Preserve exact source/configuration and test recorder noninterference. The paper's normalization is not identical to our fixed S=1/(1−gamma) transformation.

### R3 — Truly Proximal Policy Optimization

**Yuhui Wang, Hao He, Xiaoyang Tan; UAI proceedings, PMLR 115 (2020).** [Proceedings](https://proceedings.mlr.press/v115/wang20b.html).

The paper analyzes why ordinary PPO clipping need not strictly bound likelihood ratios or enforce a well-defined trust region; it proposes a modified method. Its proposed method's conditions must not be transferred to ordinary PPO.

**Use here:** retain actual conditional KL, maximum/quantile policy changes, and independent return measurements around a harmful update. A small mean KL or a clipped sampled objective does not certify preservation in sparsely represented recovery states. This is not authorization to add rollback or declare a KL threshold sufficient.

### R4 — What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study

**Marcin Andrychowicz et al., ICLR 2021.** [Author-institution publication](https://research.google/pubs/what-matters-for-on-policy-deep-actor-critic-methods-a-large-scale-study/); [2020 preprint, under the earlier title “What Matters In On-Policy Reinforcement Learning?”](https://arxiv.org/abs/2006.05990).

The study investigates over 50 implementation/design choices through more than 250,000 agents across five continuous-control environments.

**Use here:** an organized checklist for observation/action scaling, policy/value architecture, optimization and data collection. It does not provide a universally optimal PPO recipe, nor justify selecting another hyperparameter on the already-exposed held-out seeds.

### R5 — The 37 Implementation Details of Proximal Policy Optimization

**Shengyi Huang, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, Weixun Wang; ICLR Blog Track, March 25, 2022.** [Maintainer/implementer article](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

A practical, source-linked implementation reference covering initialization, minibatch advantage normalization, value fitting, global gradient clipping, observation/reward transforms and continuous actions. Its implementation stores the sampled **unclipped latent action** even when a clipped command is sent to the environment.

**Use here:** audit the precise sample/action/log-probability contract and normalization grouping. A whole-rollout-normalized diagnostic is not the actual minibatch-normalized historical loss. This article is a practical reference rather than a theorem or a task-specific fix. For clipping-estimator theory use R10 rather than overgeneralizing prose about “clipping bias.”

### R6 — High-Dimensional Continuous Control Using Generalized Advantage Estimation

**John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel; 2015 preprint / ICLR 2016.** [Paper](https://arxiv.org/abs/1506.02438).

GAE trades variance against bias through exponentially weighted temporal-difference residuals. At lambda=1, the residual sum telescopes to observed discounted rewards plus any valid cutoff value, minus the initial baseline.

**Use here:** distinguish a terminal reward-to-go, an externally truncated target, the initial baseline, and an expected action advantage. A failed sampled trajectory can have positive advantage relative to a low baseline; that is not automatically wrong expected credit. A successful controller's isolated action is evaluated under the *subsequent policy actually used*, not under that successful controller forever.

### R7 — Learning values across many orders of magnitude

**Hado van Hasselt et al., NeurIPS 2016.** [Paper](https://arxiv.org/abs/1602.07714v2).

Introduces adaptive target normalization to address the scale sensitivity of value learning without simply changing the task through reward clipping.

**Use here:** a direct reference for scale-aware value learning. Our fixed scaling experiment is **not PopArt**, and this paper does not validate retrofitting a saturated critic, an adaptive implementation, or our particular scaling constant. Retain raw-unit predictions and evaluation rewards in every comparison.

### R8 — Time Limits in Reinforcement Learning

**Fabio Pardo, Arash Tavakoli, Vitaly Levdik, Petar Kormushev; ICML 2018.** [Proceedings](https://proceedings.mlr.press/v80/pardo18a.html).

Separates a task whose objective truly ends at a deadline from a continuing task cut into shorter training episodes. A task deadline may require remaining time in the state; an artificial training cutoff calls for continuation bootstrapping.

**Use here:** do not replace external truncations with physical terminals, assume every short surviving recovery is complete, or diagnose a large tail contribution as a bug. The local same-policy timeout calibration is stronger evidence than the tail percentage alone. Correct boundary semantics do not guarantee the estimated value is accurate elsewhere.

### R9 — Worst Cases Policy Gradients

**Yichuan Charlie Tang, Jian Zhang, Ruslan Salakhutdinov; CoRL proceedings, PMLR 100 (2020).** [Proceedings](https://proceedings.mlr.press/v100/tang20a.html).

Studies risk-sensitive actor-critic learning using conditional Value-at-Risk of return distributions, with driving-simulation experiments.

**Use here:** explains why optimizing average return is not identical to preventing rare recovery failures. Risk-sensitive learning would be a separately declared objective change, not a neutral numerical PPO repair or an already-authorized production modification. CVaR of return is also not automatically the same as our physical-failure probability.

### R10 — Clipped Action Policy Gradient

**Yasuhiro Fujita, Shin-ichi Maeda; ICML 2018.** [Proceedings](https://proceedings.mlr.press/v80/fujita18a.html); [paper](https://arxiv.org/abs/1802.07564).

CAPG exploits action bounds to obtain an unbiased, lower-variance policy-gradient estimator under its stated assumptions. Multiple out-of-bound latent samples can produce the same clipped physical command.

**Use here:** a focused potential variance diagnostic for saturated Gaussian force commands. The conventional score on the **original latent action** is not automatically biased just because execution clips it. Replacing that latent action with the clipped number inside a Gaussian log density is not the correct boundary-mass likelihood either. A CAPG-style estimator or mixed boundary/continuous likelihood requires explicit derivation, tests, and separate qualification; its variance result does not automatically establish properties of our minibatch-normalized, multi-epoch PPO update. No such trainer change is made here.

### R11 — Standard CartPole and implementation guidance

[Gymnasium CartPole documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/) specifies two discrete actions, default initial coordinates in (−0.05, 0.05), and a 500-step v1 truncation. Our audited plant has continuous clipped force, broader resets, observation/actuation disturbances, learned Gaussian noise, and different training/evaluation horizons. Therefore “works on CartPole-v1” is a useful smoke test, not qualification of this task.

[Stable-Baselines3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) discusses training instability, independent evaluation, normalization and time-limit handling. It lists SAC/TD3 among continuous-action options. A matched alternative-algorithm baseline could test PPO specificity, but neither is guaranteed monotonic or a current project replacement. These online docs are rolling references; the actual experiments remain pinned to SB3 2.9.0 and their archived sources.

### R12 — Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator

**Maryam Fazel, Rong Ge, Sham Kakade, Mehran Mesbahi; ICML 2018.** [Proceedings](https://proceedings.mlr.press/v80/fazel18a.html).

Provides convergence results in a specified linear-quadratic policy-optimization setting. Those results are not a theorem about arbitrary neural PPO on nonlinear dynamics with clipped commands, observation noise and physical-failure boundaries.

**Use here:** keep LQR as a feasibility comparator. Successful model-based control establishes useful reachable behavior on the tested panel; it does not imply model-free sampled learning must discover it reliably, nor that LQR first actions are valid teacher advantages.

### R13 — Deep Reinforcement Learning at the Edge of the Statistical Precipice

**Rishabh Agarwal et al., NeurIPS 2021.** [Paper](https://arxiv.org/abs/2108.13264); [author project](https://agarwl.github.io/rliable/).

Shows how conclusions based on a small number of training runs and aggregate point estimates can change when uncertainty is analyzed.

**Use here:** distinguish independently trained agents from repeated evaluation futures of a frozen agent. Keep all checkpoints and failures, predeclare inference, and report uncertainty across the correct experimental units. A conditional interval over future-noise draws does not certify population-wide training reliability. Failure to establish superiority is not proof of equivalence.

### R14 — Original PPO algorithm and firsthand practitioner report

**John Schulman et al., 2017**, [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), defines the sampled-surrogate/multiple-minibatch-epoch method. The empirical benchmark results are not a guarantee that every update preserves real return.

[SB3 issue #1616, July 2023](https://github.com/DLR-RM/stable-baselines3/issues/1616) is a firsthand report of clipping questions and performance collapse during custom-environment training. It confirms that practitioners recognize the symptom, **not its frequency or a verified cause**. Anonymous social-media anecdotes from the earlier conversation are deliberately not used as technical evidence here.

## Evidence that must survive the literature narrative

The following are local study results, not findings attributed to the external papers:

- [Value scaling](https://github.com/yongkyuns/RustRobotics/blob/43ce58bb4ae49607cf300941e939aa08b216cf59/audits/clean_reference/VALUE_SCALE_20261007.md) improves conditioning and exposed-seed performance, but retains a preservation failure. The current output-head ceiling is a bound with that head held fixed, not total neural-network capacity.
- [8192 held-out qualification](https://github.com/yongkyuns/RustRobotics/blob/beaefe2b8415305ea73a4ed90d6e3a5ce6991a4c/audits/clean_reference/VALUE_BATCH_8192_HELDOUT_20261007.md) passes its reliability composite but fails superiority and early-learning criteria. Do not keep increasing batch size based on the noisy-gradient narrative.
- [Recovery substitution](https://github.com/yongkyuns/RustRobotics/blob/27d885971c86c0c4c85fe6ed7164957116008f41/audits/clean_reference/PPO_RECOVERY_CREDIT_20261007.md) shows genuinely harmful one-action substitutions with zero tail bootstrap alongside feasible persistent recovery. This is not a proof of a policy-gradient barrier.
- [Actual timeout calibration](https://github.com/yongkyuns/RustRobotics/blob/36a5fb4e72cdc6f1a915fa257216a3ccf5f09b7e/audits/clean_reference/PPO_RECOVERY_TIMEOUT_CALIBRATION_20261007.md) finds reasonably close predictions and 1024/1024 surviving futures at 32 selected cutoffs. These survivor-conditioned results do not clear earlier failing states.
- [Matched early trajectories](https://github.com/yongkyuns/RustRobotics/blob/96a4cd237d75dca619ced64128f2745afe289be2/audits/clean_reference/PPO_EARLY_RECOVERY_ACTIONS_20261008.md) reveal a response difference, but do not hold policy age, physical state or disturbances exactly fixed.

## Mathematical guardrails for the next causal test

For a fixed incoming policy and observed input o, a state-only baseline satisfies E[grad(log pi(a|o)) b(o) | o] = 0 when actions are sampled from that policy and the baseline is held fixed. This identity concerns the on-policy score-function expectation. It does not guarantee that a finite batch, a data-dependent normalized/clipped surrogate, or a warm Adam displacement improves control. A large baseline error may increase variance without introducing a bias in that ideal score identity. [R1, R6]

For a lambda=1 episode with m observed transitions, the target is the discounted observed prefix plus gamma^m V(cutoff) at an external truncation, or zero continuation at a true terminal. The advantage subtracts V(start). Separately check the prefix, cutoff value and start baseline. A measured value at a fixed hidden physical state and noisy observation is not automatically the observation-only posterior expectation learned by a memoryless critic. [R6, R8; local model contract]

PPO ratio clipping, environment action clipping, and gradient-norm clipping are three different operations. A Gaussian latent-action density, a clipped command distribution, and an invertibly squashed policy are also different likelihood contracts. Do not exchange them in a diagnostic or patch. [R3, R5, R10]

## Ordered next steps, without another recipe sweep

1. **Complete the already registered historical-policy cross-over**, [protocol 6061891252](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6061891252). Recover the 32 contemporaneous incoming actors for the fixed 16 matched pairs. Enforce all six original checkpoint/reward-trace matches, the full 512-batch digest, exact episode-ledger values and the selected state/observation digest. A comparison-format error must be fixed with regression tests, not by disabling a data gate.
2. **Compare both actors from both original physical states using paired future innovations**, preserving each actor's learned standard deviation. Equal Gaussian innovations are not equal force perturbations when policy sigmas differ. Report the complete cross-over matrix, failures, early behavior, centring, and raw gamma=.999 returns. This tests policy-version effects on selected states; it does not identify a particular optimizer update or validate a production policy.
3. **Then register an Ilyas-style direction test**, separating early recovery and routine balancing. Use an actual captured update direction; report unclipped score direction, actual minibatch-normalized surrogate, finite parameter-step effects and same-state return differences separately. Pair future randomness and keep the continuation policy/discount/occupancy weighting explicit. A finite difference or an LQR action substitution is not the exact global performance gradient.
4. **Choose at most one development-only intervention from the measured discrepancy.** If clipped tails dominate estimator variance, first test the R10 estimator algebra and variance without changing behavior. If expected gradients oppose harmful optimizer movement, inspect warm moments and update geometry. If good aggregate updates hurt a recovery subset, quantify coverage and the objective/reliability tradeoff before changing the objective. Otherwise retain a null result rather than force a diagnosis.
5. A controlled SAC/TD3 comparison remains a separate optional algorithm baseline, not a substitute for the pending matched test. Any corrective candidate still requires from-scratch training, fresh qualification seeds, unchanged physical evaluation and preservation of contrary results.

**Disposition:** durable literature guidance and a bounded experimental sequence, not a claim of a solved PPO defect. No teacher labels, fallback controller, larger-batch retuning, reward change, deployment or production-default promotion is authorized by this note.
