# PPO transaction60: full-rollout gradients do not rescue the retained failures

October14,2026. Pre-outcome protocol issue35/comment6180546352. Continues c753e639. Conditional experiment COMPLETE; no production correction, new trained cohort, PR merge, deployment or issue closure.

## Decision

Constructing the actor direction from ALL2048 recorded rollout rows, instead of the selected64-row minibatch, does NOT rescue the same ordinary and difficult recovery losses. This holds with original per-block normalization or whole-rollout normalization, with and without fixed adaptive scaling. All four alternatives improve BOTH full-rollout surrogate objectives, yet retain the same two failures and slightly reduce mean physical return relative to original Adam on both exposed panels.

This rejects these particular full-rollout direction replacements as a local rescue, not full-batch training in general, all minibatch-related effects, or a claim that the recorded rollout represents recovery adequately. No true/global gradient sign or production recipe is established. This is the historical stock-SB3 reference transaction60/global Adam step130940/update410, not a new production Rust optimizer measurement.

## Fixed intervention

Exact PRE59 actor and original observations/actions/old likelihoods/raw advantages/clip range. Four directions:

- PARTITION_RAW: negative mean of the gradients of ALL32 original epoch2 minibatches, each evaluated at the SAME PRE59 policy with its original64-row advantage mean/sample std.
- PARTITION_ADAPT: same mean gradient divided by the fixed original transaction60 Adam denominator.
- GLOBAL_RAW: negative gradient of one2048-row surrogate with whole-rollout advantage mean/sample std.
- GLOBAL_ADAPT: same global gradient divided by that fixed denominator.

No parameter step occurs between the32 gradients. No new labels, critic fitting, data rejection, supplied controller or first-moment mixture. All actor-network/log-sigma coordinates included. Post60 critic is retained but unused for control. The adaptive denominator is sqrt(v60/(1-.999^130940))+1e-5; it is a reference geometry, not a fresh optimizer history.

Each candidate is scaled with the inherited fixed32-doubling/80-bisection rule to match mean latent KL(pi_PRE59||candidate) on ALL2048 original rollout observations, with float32 stored candidates. Target8.611371737057112e-5; relative errors<=2.414e-7, below fixed.001. No evaluation observations/returns choose the scalar. This is a sampled latent-policy radius, not a clipped-force/global trust-region guarantee, equal Euclidean displacement or a qualified training learning rate.

## Complete physical results

All original1024 ordinary and512 difficult cases, previously EXPOSED development data. Policy410 controls ticks0–63, tested actor from64 onward. No reset/reseeding at switch. Same full post-reset native/RNG clones, paired innovations,64 inference lanes including inactive lanes, noisy nonlinear native plant,clipped commands,physical limits,1500-tick horizon,gamma.999 raw stopped return,no bootstrap.

|Actor after tick64|Ordinary complete /1024|Difficult complete /512|Gains/losses versus PRE59|
|---|---:|---:|---|
|PRE59|1001|400|Reference|
|ADAM|1000|399|0/1 in each panel|
|PARTITION_RAW|1000|399|Same lost cases|
|PARTITION_ADAPT|1000|399|Same lost cases|
|GLOBAL_RAW|1000|399|Same lost cases|
|GLOBAL_ADAPT|1000|399|Same lost cases|

Casewise identity is checked, not inferred from totals. All physical failures are cart-limit, not pole-angle. No alternative gains/loses a completion versus ADAM. Exactly9216 new trajectories; zero new independent reset/noise cases or independently trained agents.

Difficult witness3309102759 survives1500 ticks under PRE59, fails109 under ADAM and110 under all full-rollout alternatives. Ordinary lost key3209103024 survives under PRE59 and fails124–125 under the changed policies. These selected cases are not population effect estimates.

### Mean-return changes versus ADAM

Descriptive pointwise paired99% Student intervals, fixed-policy exposed cases:

|Alternative|Ordinary change [99% interval]|Difficult change [99% interval]|
|---|---:|---:|
|PARTITION_RAW|−.046403 [−.051027,−.041778]|−.097520 [−.112261,−.082780]|
|PARTITION_ADAPT|−.044382 [−.050110,−.038654]|−.093357 [−.111755,−.074958]|
|GLOBAL_RAW|−.036305 [−.041356,−.031254]|−.086596 [−.102429,−.070764]|
|GLOBAL_ADAPT|−.027676 [−.034356,−.020996]|−.075316 [−.096574,−.054058]|

Small differences relative to mean returns around723/526, but not improvements. All fixed intervals below0. These are not simultaneous/familywise, newly held-out, trained-agent-population or hardware guarantees. The original ADAM-minus-PRE59 global reset-mean signs remain unresolved; a lost case is not proof of negative global expected return. No equivalence margin was declared.

## Internal fitting improves, recovery does not

Original selected-minibatch gradient cosine with full partition=.177813; with whole-centered=.182164. The two full gradients have cosine.993724 with each other. This is a substantial direction change, not a no-op.

|Alternative|Actor displacement norm|Cosine with original Adam displacement|32-block mean loss change|Whole-centered loss change|
|---|---:|---:|---:|---:|
|PARTITION_RAW|.000832284|.044137|−.000845483|−.000776617|
|PARTITION_ADAPT|.003951766|.152229|−.001012974|−.000951166|
|GLOBAL_RAW|.000854007|.043835|−.000861442|−.000801512|
|GLOBAL_ADAPT|.004330893|.161405|−.001040808|−.000996216|

Original Adam displacement norm≈.004869398. Negative loss change=improvement. Every block loss and both gradient projections retained. This is an ACTUAL full-rollout-gradient intervention, not merely the earlier observation that a minibatch-derived direction could happen to improve the full-rollout surrogate.

The remaining distinction is optimizing this recorded surrogate versus preserving physical recovery. This experiment does not choose between inaccurate advantages/value estimates, finite rollout coverage, evaluation-state distribution, off-sampled-state step size, or a combination. It does not establish a wrong mathematical infinitesimal/global gradient. No variant is adopted.

## Exact inputs and executed verification

Geometry compact1ed133262145e238a850bbee8183a1a08559b97443a678d3cf9c561366fc66db:204 payload hashes,30 prior tests and2 reports replay before new experiment. Original confirmation717d6f72d39dca2e16afd922911b58f51ae914ced10c15282d034f272a63cefb outer hash and all hydrated group hashes checked. Runtime c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and11 payloads pass.

AMD EPYC9V74; Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0. Unchanged warm320-step replay exactly matches final policy/Adam/all RNG with zero simulator calls; original native/actual SB3 source gates pass. No initial-weight substitution, fresh Adam or relaxed identity.

32 NEW tests pass: aggregate/partition/normalization/candidate/critic/input/schema/radius/receipt and physical-record corruption controls, including one-representable-float reward changes. Eight finite-difference checks independently validate aggregate directional derivatives. Separate pinned Torch float64 autodiff checks33 gradient graphs (all32 blocks+global) and12 losses: max block-gradient error2.22044605e-15, global8.60422844e-16, partition mean1.94289029e-16, loss2.77555756e-17.

Independent vectorized NumPy reader forms a single2048-row derivative graph, separate from the producer's averaged block graphs. It verifies gradients/candidate equations/input hashes and KL. Physical reader checks all9216 paths,144 main arm archives/24 groups, selection/source/policy bindings, original64-step prefixes, innovations, forces,float32 reward bits,endings,padding,sequential returns and independent float64 inference. Max normalized-command error2.71306467e-7<3e-6. Reader does not independently reintegrate native dynamics or regenerate private native noise.

Six native controls are byte-exact:512928 transitions. All3072 PRE59/ADAM case records equal the prior study; both available64-case raw fixtures match every array byte. Missing prior raw groups are not claimed compared.

Fresh final FULL extraction verifies640 payload hashes,32 tests and FOUR numerical reports byte-for-byte. Fresh COMPACT extraction verifies299 hashes,32 tests and the candidate mathematical report; it explicitly skips full raw-panel reduction. The separate Torch receipt is not falsely called an offline rerun. All source is retained; no Torch/native/optimizer/pickle executes during offline REPLAY.py.

## Costs, test setup and publication

Main12676056+controls512928=**13188984 native transitions**. Optimizer work320 exact warm-replay steps; synthetic candidates0. New training-environment data0, independent cases0, trained agents0. Every main arm completes once in create-only directories: no simulation retry, interruption or unrecorded cost. No efficiency claim.

The first expanded32-test invocation had four receipt-test failures because nested retained utilities put the PRIOR reader earlier on sys.path. Corrected tests load the NEW reader by exact file path; offline replay pins paths too. Failed source/logs retained. No candidate, simulation, result or tolerance changed. Final32 tests pass.

Exact direction generator committed366bf25146f98a04f2031a5a342360af17dbdbc5 under fullrollout410_20261014/code/full_gradient.py; SHA2563cd7d9ef319f3a97c448697561dc039ff6353d86dbe06195ecfc144f1a95d6fe. It expects the packaged historical input layout. Complete eight-source-module collector/reader/test implementation, README and replay are delivered as both an applicable audit patch and evidence packages—not falsely described as integrated into production or routine CI.

Artifacts:
- Full ppo_fullrollout410_evidence_20261014.zip:869775381bytes, SHA256bff91ff6cb3ce68784a11973ef84b9cfcf5769ae324fe3cd513debb12cc2755e.
- Compact ppo_fullrollout410_report_code_20261014.zip:151295110bytes, SHA2565e57a283432d1fc03ab5d932eb6bf702f14738425b44c5e256861141e02efd65.
- Detailed ppo_fullrollout410_report_20261014.md, ppo_fullrollout410_audit_20261014.patch and detached verification JSON/logs accompany them.

Full offline command: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python REPLAY.py. Reader Python3.13.5/NumPy2.3.5/SciPy1.17.0. Parent manifests retained with SUBSETS.json explaining omissions/hydration; top-level manifest alone enumerates this package. Runtime installers/fonts excluded. README documents separate strict native regeneration; mismatch gates must not be waived.

A supported correction still needs a declared from-scratch ordinary-performance/recovery-retention comparison and #33 sustained-balancing qualification. No production/default/reward/noise/reset/physics/protected-source/held-out-training-seed/master/deployment change, production PR or merge; #33/#35 remain open.
