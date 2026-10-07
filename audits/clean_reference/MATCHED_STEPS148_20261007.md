# Update148: momentum-clearing rescue is not explained by smaller steps alone

October 7, 2026. Fixed LOCAL conditional diagnostic; not a qualified PPO correction.

## Decision

The matched-length control registered before new outcomes in issue35/comment6033941815 is complete. Smaller retained-momentum updates reduce the collapse, but do not reproduce the momentum-cleared result. Enlarging momentum-cleared updates to the original large lengths still largely avoids collapse. This rejects a parameter-step-size-only explanation of the previous LOCAL rescue, without establishing a universal cause or safe new training recipe.

## Exact local controls and scope

Input: prior local factorial package SHA256b97ace9a969f16af240a6976f43c9577be603c27efb22a2b3cd5e4fd88fcfdc5. All168 payloads verify and its original offline analysis/tests replay. Runtime ZIP7418edd50c11ae67ef761610a15c4a0c6ee70f2d3fa032609d6394eb745b5dd1 and31 payload hashes verify. Host AMD EPYC9V74; Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB32.9.0/Gymnasium1.3.0.

Restore the same retained LOCAL post313 actor/critic/full-Adam/RNG state and execute only original minibatches314–320. The exact historical Intel experiment remains unqualified; no historical equality gate is loosened. No prefix retraining, weight-only approximation, rollout collection, new training seed or gamma/lambda sweep.

A and B reproduce the previous LOCAL parameter tensors after EVERY step and full final policy/Adam/RNG states exactly. A additionally reproduces the full archived Adam state at every step. Previously exposed start313/A/B evaluator panels reproduce all288 records exactly before any new evaluation.

## Four fixed cells

| Cell | Initial actor first moment | Actor parameter-step length |
|---|---|---|
|A keep/native|Retain|Ordinary Adam|
|B clear/native|Clear once before314|Ordinary Adam|
|E keep/small|Retain|Match each corresponding archived B length|
|F clear/large|Clear once before314|Match each corresponding archived A length|

All cells use joint clipping and the identical LOCAL original critic path, including raw/scaled gradients, weights, first/second moments and clocks. Holding this recorded critic trajectory fixed is a diagnostic control, not a deployable trainer. Initial state differs only in actor exp_avg for B/F; weights, second moments, optimizer clocks, betas, epsilon, learning rate and critic state are preserved. Subsequent momentum accumulation is ordinary Adam.

For E/F, compute each cell's own stock Adam proposal, then apply ONE positive scalar to the concatenated actual actor displacement, including log_std. Compute differences/norms/scales in float64 and cast parameters back to float32. No per-layer scaling, gradient rescaling, or post-step modification of Adam moments/clocks. A/B are untouched native steps. Largest absolute target-norm error8.174e-9; largest direction-cosine error3.091e-9. Fixed tolerances were norm rtol2e-4/atol1e-6 and cosine>=.99999, separate from bitwise control checks.

Every subsequent gradient/moment is recomputed at the cell's own current parameters. Thus this is a stateful crossed-norm-schedule experiment, not transplantation of the original direction sequence. Matching parameter norms does not also match action changes, policy KL, or per-layer norms.

## All new physical outcomes

Eight new common domains8960204+100000*k,k0..7. Each uses the unchanged32 deterministic10s and64 stochastic15s cases, original inference shapes/noise/native physics, and gamma.99 evaluation. Training gamma.999/lambda1 and the actual fitting batch/targets are unchanged. Training's256-step cap does not shorten evaluation.

| Policy | Deterministic /256 | Stochastic /512 | Mean stochastic seconds | Mean stochastic discounted return |
|---|---:|---:|---:|---:|
|Common start313|250|486|14.585508|74.634158|
|A keep/native large steps|155|195|10.534297|73.784618|
|B clear/native small steps|252|493|14.686113|74.893899|
|E keep/matched small steps|220|410|13.712383|74.898513|
|F clear/matched large steps|249|486|14.625234|75.246550|

**Small matched lengths, B versus E:** net+32 deterministic and+83 stochastic completions. Exact paired gains/losses are33/1 deterministic and87/4 stochastic. Matching small parameter-step lengths does not remove the benefit of clearing inherited momentum.

**Large matched lengths, F versus A:** gains94 deterministic and291 stochastic completions, with ZERO losses against A. Restoring the large original parameter-step lengths does not restore the collapse.

**Step size still matters:** E versus A gains65 deterministic and215 stochastic completions with zero losses. Nevertheless E remains below the common start:31 deterministic losses/1 gain and80 stochastic losses/4 gains. The result is not that step magnitude is irrelevant.

Contrary data remain: B has8 stochastic gains and1 loss versus start313, not perfect retention of every case. F ties start313's pooled stochastic completion through9 gains AND9 losses; it is not identical behavior. F loses one deterministic completion net. B-E mean stochastic discounted effect is-0.004613 despite B's83 extra completions. F has the highest discounted score but B the highest completion. Do not silently swap the survival metric for discounted return.

A/E endpoint failures are exclusively position-limit endings. B has16 position and3 angle failures stochastically; F has26 stochastic position failures. All physical ending labels are independently checked. These are conditional comparisons of one selected fixed-batch witness, not512 independently trained agents or held-out training evidence.

## Actor optimizer diagnostics

| Cell | Current-minibatch actor-loss ascent steps /7 | Final whole-batch actor loss |
|---|---:|---:|
|A|5|+.015187507|
|B|1|-.001193899|
|E|4|+.001089063|
|F|1|-.002034670|

Incoming whole-batch loss+.000182899; lower is better. This uses whole-batch advantage normalization and is not labelled the identical objective of every separately normalized minibatch. E retains four local ascent steps with smaller lengths; F has one with large lengths. The matched physical and optimization outcomes support an inherited-moment/stateful-direction contribution in this tail, not an Adam arithmetic defect.

An explicitly post-result descriptive read of the saved first proposals gives A/B direction cosine.973506 at their common initial parameters. No extra actor was evaluated. This is not a claim that the first direction reverses completely; the seven-step trajectories matter.

## Verification, execution cost and evidence

Independent NumPy-only reader checks3840 primary records,28 transactions/post-step policies, eight complete initial/final optimizer exports, exact state continuity/critic paths, clocks, clipping, first/second-moment recurrence, Adam proposals, scalar displacement transforms and frozen norm schedules. Independently ordered float64 MLP/Gaussian/clipped-surrogate calculations check112 actor losses, maximum discrepancy1.088e-7. Maximum independent Adam-proposal discrepancy over all tensors is4.747e-7, within float32 rounding of large critic parameters; live local control equality remains bitwise.

**26 actual-record tests plus8 rescaling unit tests pass.** Missing/duplicate cases, wrong domains/checkpoints, nonfinite outcomes, false/shortened completions, optimizer-order/clock/moment/gradient corruption, altered proposals/applied weights, critic-path changes and wrong scaling metadata are rejected.

All40 new panel processes completed ONCE with immediately saved outcomes and exact simulator-call receipts. No learning/evaluation retry. The initial interactive-session tool request was rejected before executing its shell; it produced no numerical work. A non-interactive launcher then completed the original fixed panel set.

New costs: **0 training-environment transitions;28 fixed-batch Adam transactions;4,711,160 primary evaluation calls;342,851 identity-control calls;5,054,011 total native evaluation calls.** Earlier experiment costs are separate.

Evidence package `ppo-tail148-matched-steps-evidence-20261007.zip`: **21,780,109 bytes**, SHA256 **0ff29503e7dc2cf0f409915284bd4f0c822f7eb56bba7181ceba7285509b70c7**. FRESH extraction verifies305 new package payloads and the unchanged prior ZIP/168 payloads, reproduces VERIFIED.json BYTE-FOR-BYTE, and passes all34 tests. It includes the unchanged prior local package, full report, executed measurement/launcher code, independent reader/tests, every new outcome/optimizer trace and runtime receipts. Third-party runtime binaries are excluded. No simulator/optimizer is executed by offline verification.

Default replay: `PYTHONDONTWRITEBYTECODE=1 python code/REPLAY.py .` with Python3.13.5/NumPy2.3.5 runs the26 NumPy reader tests. Add `--runtime-python /path/to/preserved/python3.11` to also run8 Torch rescaling unit tests. Full measurement reproduction is a separate exact-local-control-gated operation described in README.md.

Human report: `ppo-tail148-matched-steps-results-20261007.md`. Detached receipt: `ppo-tail148-matched-steps-verification-20261007.json`. Official PyTorch2.8 Adam/clip_grad_norm_ documentation was consulted; the measurement loader checks the actual pinned library sources against original evidence.

## Next decision

A lower learning rate alone is no longer an adequate explanation of this local rescue. The next practical question is how an ordinary-PPO optimizer configuration handles carried directions across changing minibatches/rollouts without a witness-timed momentum reset. Any proposed configuration needs fixed from-scratch early-learning/late-retention comparisons and fresh held-out qualification. No new recipe or sweep is registered by this result.

**Diagnostic COMPLETE; historical Intel replay UNQUALIFIED; training correction UNQUALIFIED. Master, defaults, reward/noise/reset laws, protected sources, production PRs and deployments unchanged. No merge. Issues33/35 remain open.**
