# Update410 fresh action-credit calibration: diagnostic complete, signs unresolved

October14,2026. Preregistered issue35/comment6175230162. Continues step-direction report40e6a79f. No training correction, production merge or deployment.

## Decision

This experiment **does not establish a critic-bootstrap-induced sign reversal** along the saved410→411 direction. Surviving256-step cutoff values are modest underestimates of actual continuation return. Replacing them with measured long returns changes the score estimates much less than their large uncertainty. Both primary gradient signs and both paired bootstrap differences remain unresolved.

This is negative/inconclusive evidence for the tested explanation, not proof that the critic is universally correct or that estimator noise caused the historical collapse. The existing finite-step recovery regression remains real. No default or correction is selected from these results.

## Fixed experiment

Frozen incoming410 controls every action. Incoming411 only defines the actor-network/log-sigma direction for analytic scores; it never controls these new trajectories. This continues the stock-SB3 reference witness used in the RustRobotics investigation, not a new measurement of the production Rust optimizer. Development training seed201 only; no use of held-out training seeds830101–830108.

New native reset panels:512 ordinary keys3610142600+i; first512 difficult qualifying keys3710142600+j in ascending order, predicate abs(theta)>=.15,abs(omega)>=.25,theta*omega>0. Difficult selection considered5,136 keys, ending3710147735. Both panels froze before policy trajectories. All considered reset states, noisy observations, packets and selection flags are retained.

Original noisy nonlinear native plant, disturbed actuation and clipped command mapping, fixed64 inference lanes including inactive lanes, policy innovations default_rng(key XOR0xAC141041).standard_normal(8192).astype(float32), first physical ending or8,192 ticks. No256-step timeout/reset inside a continuation, no final critic bootstrap. Exactly1,024 completed trajectories /7,441,504 native transitions. Zero training/optimizer transactions.

Pinned runtime: Python3.11.14 / Torch2.8.0+cpu / NumPy2.2.6 / SB3 2.9.0 / Gymnasium1.3.0. Original actual PPO, buffer, policy and collector source bytes match; original native archive and all142 payload hashes pass the strict loader. The runtime outer ZIP c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and all11 runtime payloads pass. Original localization/recovery archive hashes and all306/648 payload hashes also pass.

## Primary estimand and result

Let T=min(256,physical episode length). Sum gamma^t times the Gaussian action score along theta411−theta410, multiplied by either measured finite8,192-tick reward-to-go or the256-step lambda1 return with frozen critic bootstrap. Both subtract exactly the same state-value baseline. Gamma=.999; the scaled critic is multiplied by1,000 to recover physical reward units.

The paired difference is checked against the independent identity:

    D_boot − D_measured
      = gamma^T * (V_at_cutoff − measured_tail_return) * sum(first-T scores).

It is exactly zero for physical failures at/before256. This isolates bootstrap substitution on identical data, not a changed initial-state baseline or a different policy rollout.

|Reset panel|Critic-bootstrap projection [99% interval]|Actual-return projection [99% interval]|Bootstrap minus actual [99% interval]|
|---|---:|---:|---:|
|Ordinary|−10.364 [−29.923,+9.195]|−10.653 [−30.281,+8.976]|+0.289 [−0.291,+0.869]|
|Difficult|−22.773 [−99.890,+54.344]|−23.574 [−100.782,+53.635]|+0.800 [−0.357,+1.957]|

All six intervals contain0. Negative point estimates are not demonstrated harmful expected direction. No equivalence margin was declared, so unresolved paired differences do not establish absence of material bootstrap effects.

Intervals are the fixed pointwise two-sided99% Student intervals across512 independent episode/reset/noise cases per panel, not trained-agent-population or simultaneous multiple-comparison intervals. No sample extension after observing uncertainty.

This score estimand concerns perturbing the first256 actions and then reverting to the old policy over a finite8,192-tick reward horizon. It is NOT the previous1,500-tick whole-policy finite difference, an infinite-horizon gradient, or the exact original minibatch-normalized PPO loss. Unweighted-prefix projections are retained separately for comparison with ordinary PPO's sampling convention.

## Physical outcomes and cutoff calibration

|Panel|8,192-tick completions|Physical failures by256|Predicted cutoff tail|Measured cutoff tail|Mean prediction error|
|---|---:|---:|---:|---:|---:|
|Ordinary|507/512|5|978.952|981.522|−2.570|
|Difficult|400/512|112|973.546|977.221|−3.675|

All117 physical failures occur before256 and hit the cart-position limit. All907 paths alive at256 survive through8,192 ticks,81.92 seconds. These are outcomes for one frozen old policy, not a new sustained-learning qualification.

Surviving-cutoff mean-error intervals: ordinary[−2.722,−2.418], difficult[−4.112,−3.238]. RMSE2.889 and4.987; mean absolute error2.572 and3.678. Largest overpredictions are+0.391/+0.350; largest underpredictions−12.191/−17.233.

This does not resemble grossly optimistic cutoff values assigning high return to immediate subsequent failures. It does NOT certify earlier recovery-state values: survivor-conditioned cutoff calibration cannot clear credit at failed starts. Measured tails are finite, not exact infinite-horizon values.

## Prespecified secondary clipped-action score

At clipped commands, replace the latent Gaussian score with the derivative of the corresponding boundary probability mass; interior scores are unchanged. This is a readout on the same trajectories, not a changed policy or training objective.

Measured-return score standard deviation: ordinary171.781→133.295 (39.79% lower observed sample variance), difficult675.709→638.546 (10.70% lower variance). Clipped-score measured-return intervals still span0: ordinary−6.786 [−22.017,+8.444], difficult−27.163 [−100.126,+45.799]. Bootstrap-difference intervals remain unresolved too.

No CAPG-based trainer or clipping-causes-collapse claim follows. Both estimators, unweighted readouts and every case remain in the evidence.

## Executed verification

Incoming410's66 original trajectories and original NPZ digest reproduce exactly. The new recorder then reproduces every array of all64 archived stochastic trajectories without changing them;178,306 native control transitions.

**32 tests pass:**12 independent mathematical controls and20 genuine-fixture/corruption reader tests. A separate PyTorch float64 autodiff check covers64 actor directional derivatives, maximum discrepancy1.39e−17. Tests include clipped-boundary mass finite differences/quadrature, baseline cancellation, terminal/return decomposition, wrong reward bits/noise/actions, altered values/cutoff observations/endings/keys/order, false selections, nonfinite data, partial panels and a real zero-direction control.

The NumPy/SciPy reader checks all1,024 physical paths, actor identities, innovations, command mapping, native float32 reward bits, observation bounds, physical endings, critic inference, scores and telescoping identities. Maximum independent command discrepancy2.57082862e−7 (<3e−6); maximum scaled-value discrepancy3.34697327e−7 (<2e−6).

A **fresh extraction of the final full ZIP** verifies115 payload hashes, the original historical NPZ digest, exact recorder controls and all32 tests; both numerical reports regenerate byte-for-byte. The reader uses no Torch/SB3/native/optimizer/pickle. It does not independently reintegrate dynamics or regenerate private native RNG draws.

## Retained failures and honest cost accounting

Initial import failed on the known SELECTION_SLIM.json alias; a byte-identical copy of original SELECTION.json repaired only that filename. Before executing the reader on measurements, source review corrected its unexecuted draft reset bounds to the actual archived .2/.4/.25/.5; the earlier draft is retained. An interactive-container request was rejected before execution.

A synchronous tool timeout completed difficult groups0–2 but left group03 with a zero-byte archive/no receipt. That directory/log is retained. Only group03 was rerun using the same frozen keys, policy, innovations and horizon, without observing an outcome to select the retry. The incomplete attempt is NOT claimed byte-compared or zero-cost. Its unrecorded transition count is at most524,288; successful recovery records378,641.

Known completed work:7,441,504 main +178,306 controls =7,619,810 native transitions, plus the unrecorded interrupted attempt (upper bound524,288). Exactly1,024 new independent cases, zero training and optimizer transactions. No efficiency claim.

## Code, evidence and disposition

The exact independently tested numerical core is committed in `credit410_20261014/credit_math.py` (SHA256009dd2ae707d900ec98a90e1a759d5673f5bdb5f0758b9e0fa2b2d1b4a560016). Complete executed collector, strict loader, independent reader,32 tests, autodiff control, policies, provenance, failure records and all numerical reports are in the attached packages. They are NOT falsely described as integrated into the production trainer or routine CI.

- Full `ppo_credit410_evidence_20261014.zip`:359,320,044bytes, SHA256 **204543f84d2473e33b2aa12da1fe90d949ff6949a4a64b4840fc45945bf241ee**.
- Compact `ppo_credit410_report_code_20261014.zip`:16,278,233bytes, SHA256 **372ff3e1e3231b4c9153dca5896a67d8cd9d0f71b02851776cd9a1e93dea50fb**.

Full read-only reproduction: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python REPLAY.py`. Recorded reader Python3.13.5 / NumPy2.3.5 / SciPy1.17.0. Compact replay explicitly omits full-trajectory reduction. Native/source subsets retain original manifests with explicit omission notices; third-party runtime installers are excluded. README.md documents separate exact-runtime native regeneration.

Next estimation target: same-state paired first-action continuations to separate sampled action credit from later action noise, rather than blindly enlarging this high-variance whole-episode score calculation. No such correction or next experiment is claimed executed here. Any production candidate still needs a declared from-scratch comparison and #33's sustained-balancing qualification.

Master rechecked at8514317e3b5cd57dc37bee2aa4c950af674751e1. No default, task/reward/reset/noise, protected-source, production PR, deployment or held-out-training-seed changes. #33/#35 remain open.

Method references: Schulman et al., Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 ; Fujita and Maeda, Clipped Action Policy Gradient, ICML2018, https://proceedings.mlr.press/v80/fujita18a.html . Current online documentation was not substituted for pinned SB3 source bytes.
