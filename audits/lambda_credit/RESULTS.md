# Lambda-one action credit: witnesses repaired, learning not qualified

September18,2026 America/Toronto; execution timestamps September19 UTC. Issues #35/#33. Corrected nonlinear production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**The two-stage experiment is complete. Lambda1 correctly ranks all14 frozen diagnostic cases, but the same change makes from-scratch control worse at the fixed4,194,304-transition budget. It is not promoted as a default or a reliable-controller correction.** Final deterministic return and outward-recovery completion regress on every exposed training seed. Three of four stochastic discounted returns also regress. No final candidate passes the existing deterministic robustness screen.

This is an important distinction: repairing selected old-policy action-credit witnesses is necessary evidence about those estimators, but is not sufficient evidence for the entire learning procedure. No new production default, controller, reward, plant change, merge or deployment was made. PR38 remains separate and unmerged. The earlier lambda-one negative experiment on the old linear plant remains valid; this study does not replace it.

## Protocol and actual execution

[Protocol posted before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5738910010). Audit branch `audit/ppo-lambda-one-credit-20260918`. Executed commit **a319fe6122f7415df64d4476fb69549d690ac9e3**, [run35418135240](https://github.com/yongkyuns/RustRobotics/actions/runs/35418135240). All nine jobs passed on their first attempt: preflight/credit gate followed by eight measurements. No trained seed was dropped or retried for a more favorable outcome. The workflow's green result means execution and invariants succeeded, not that the controller qualified.

The only learning setting changed is the existing `PpoTrainerConfig.ppo.gae_lambda`: baseline0.95 versus candidate1.0. Gamma0.99,512-step single-environment collection,128 minibatch,four ordinary epochs,epsilon1e-5,learning rate0.0003,existing2x64ReLU networks,exploration scale,reward,noise,reset/termination and corrected nonlinear plant remain unchanged. No extra critic passes, arbitrary-state training, supplied feedback law, diagnostic outcome, pretrained critic, normalization change, reward modification or optimizer replacement enters training.

A temporary audit ABI selects the parameter before constructing a fresh ordinary `PpoTrainerSession`. All actual updates call its unchanged `train_updates`. The original production trainer is modified only by appending a test-only module; production GAE and optimization bodies are not replaced. Constructors, episode/RNG state and both Adam histories persist continuously across fixed checkpoints. Evaluation snapshots never replace the live learner.

For a remaining trajectory length n, lambda1 telescopes to discounted sampled rewards plus gamma^n times the final-state value, minus the initial-state value. True terminals remove the final bootstrap; timeouts and live buffer cutoffs retain it. This removes intermediate critic predictions, but is NOT full-episode Monte Carlo, an exact return oracle, or removal of every critic dependency. In ordinary512-step rollouts, actions near the end have far fewer than512 subsequent samples. The initial baseline and finite-batch advantage normalization remain.

## Stage1: every fixed witness retained

Reuse all14 roots, all256 paired random seeds per root, both first-action branches and separate noiseless controls from the previous corrected-model rest-point study. Its original raw artifact10576391630 and SHA256 `9b29ad8d9fc34f6c6eecc6141bae66ca31acd9eaaf5863a2d5e0459414d6b295` are verified. All frozen actor/critic weights and physical states are unchanged.

The actual Rust probe executes the same stochastic continuations: first latent action plus/minus0.02, then the unchanged frozen policy, paired environmental/action innovations,2048-step return horizon or real terminal,512-step credit window. The sole measured probe change is lambda1. All7,196 resulting records have **byte-identical noncredit CSV fields** to the original experiment, including actions,observations,returns,durations and ending labels. This does not create new independent empirical evidence for the already-exposed return reference.

The predeclared gate requires all three previous wrong-sign cases to become confidently correct, with no new confidently wrong cases. Paired Student intervals use99% Bonferroni family coverage over14 roots x2 contrasts(actual return and candidate credit), with256 paired replications. That28-contrast family was fixed before execution; the previous study's42-contrast family also included TD and is not silently reused.

**Gate result:14 same-sign,0 opposite-sign,0 inconclusive.** The three original reversals are:

| Frozen witness | Actual discounted-return change | Old lambda0.95 credit | New lambda1 credit |
|---|---:|---:|---:|
|Constant201 at16M|-0.005190859|+0.001341052|-0.005195572|
|Constant202 at4M|-0.012093802|+0.001553199|-0.012111216|
|Constant203 at4M|-0.003178728|+0.001198534|-0.003179671|

For seed203, the new adjusted credit interval is[-0.003975483,-0.002383858], compared with the actual-return interval[-0.003974450,-0.002383007]. The old credit interval remains positive under the newly declared comparison family. All other cases and intervals are retained in gate.json and the independently generated summary.

This gate tests the first action with a full512-step remaining window, at selected old-policy resting points. It does not test every shorter remaining horizon, arbitrary recovery states, normalized minibatch updates, or the states later visited by a new lambda1 learner. Passing triggered the previously fixed learning matrix; no further parameter choice was made from the results.

## Stage2: equal-budget from-random learning

Two arms x exposed training seeds201–204; continuous checkpoints0/1,048,576/4,194,304. Each final run has8,192 policy updates,131,072 actor and131,072 critic Adam steps, and16,777,216 gradient-sample visits per network. There is no16M extension or new held-out cohort in this study.

At each checkpoint, use the unchanged32 noisy deterministic10s and64 noisy stochastic15s panels per seed. At1M and4M, use the unchanged native224-episode robustness screen per seed:32 nominal deterministic and32 stochastic300s,32 outward-recovery60s,six individual plus/minus10percent physical shifts with16 episodes each,16 doubled-noise episodes,and16 episodes with20ms command delay. Portable native policy inference and the original termination thresholds are retained. A first failure ends an episode; automatic resets cannot mask it.

The existing necessary final screen requires zero failures for EACH policy on nominal deterministic300s and every deterministic stress panel. A pass would still not establish hardware safety or independently replicated learning reliability.

### Fixed4M endpoint

| Measurement | Baseline lambda0.95 | Candidate lambda1 |
|---|---:|---:|
|Noisy deterministic10s completion /128|122|93|
|Deterministic mean return|883.099434|729.139693|
|Noisy stochastic15s completion /256|228|181|
|Stochastic discounted mean|82.712587|81.394112|
|Nominal deterministic300s completion /128|114|90|
|Nominal stochastic300s completion /128|114|89|
|Outward recovery60s completion /128|96|52|
|Final policies passing full necessary screen /4|0|0|

Short and native-long panels have different keys/inference paths. Compare arms within a row; do not treat a300-second trial as an extension of a particular10-second trial. These episode counts pool four trained agents, not128/256 independently trained agents.

### Every seed at4M

Arrows denote baseline to lambda1.

| Seed | Deterministic mean return | Stochastic discounted mean | Nominal300s /32 | Outward recovery /32 |
|---|---|---|---|---|
|201|892.392544 ->595.528117|88.057035 ->84.527609|30 ->16|24 ->4|
|202|841.514554 ->680.752782|81.685168 ->81.338990|29 ->25|24 ->16|
|203|863.977056 ->789.343960|78.648607 ->79.059731|26 ->25|24 ->16|
|204|934.513582 ->850.933914|82.459540 ->80.650117|29 ->24|24 ->16|

All four deterministic returns,all four nominal completion counts,and all four recovery counts regress. Only seed203 slightly improves the discounted stochastic score. Mean paired deterministic effect is-153.959741 with a pointwise99% Student training-seed interval[-454.232514,+146.313032]; stochastic effect-1.318476[-6.395424,+3.758473]. These n4 intervals are broad and cross zero. They do not prove equivalence or no effect; nor do they support promotion. Fixed endpoint failures already reject the declared robustness requirement.

At1M the candidate also trails: deterministic completions90->58/128,stochastic184->116/256,nominal300s deterministic97->62/128,and recovery72->12/128. There is no selected earlier winning checkpoint. All per-seed checkpoint outcomes are retained.

## Centring improves among survivors, but survival worsens

At4M all90 candidate nominal deterministic survivors satisfy the predefined centred/upright condition throughout their last10s:abs(position)<=0.5m andabs(angle)<=0.1rad. Baseline has85 centred survivors among114 survivors. Thus centred-and-surviving outcomes increase only85->90 out of128 initial trials while total survivors decrease114->90. Reporting only100percent centring CONDITIONAL ON SURVIVAL would hide38 candidate failures versus14 baseline failures.

The failures are not all the same kind. Nominal deterministic endings change14 position failures at baseline to21 position plus17 angle failures with lambda1. Stochastic nominal failures change14 position to23 position plus16 angle. Recovery changes32 position failures to43 position,32 angle,and1 combined angle/position failure. The candidate does not simply eliminate drift while preserving angular recovery. Every candidate still fails the full necessary screen.

No local equilibrium enumeration of the newly learned policies was performed here; centring claims above come from actual recorded trajectory statistics, not assumed basins or noiseless eigenvalues.

## Interpretation boundary

The frozen-action witnesses are repaired by longer reward traces, but that single parameter change is insufficient as a training correction. Higher sampling variance,remaining boundary bootstraps,current-value baselines,and finite correlated normalization remain possible contributors. This study does not export all training advantages/gradients or independently identify which of them causes the new failures. It would be incorrect to claim that the negative learning result proves variance is the cause, or that all of the candidate's action credits are wrong.

The next investigation needs to distinguish the full-rollout learner from the full512-step first-action diagnostic, rather than assuming the latter qualifies every training sample. No further knob or new candidate is promoted by this report.

## Verification and reproducibility

Preflight passed **77 native unit/ABI/lambda tests**,7 frozen-probe tests,7 ordinary balancing controls,9 ordinary learning/evaluator controls,strict Clippy,compilation,formatting and source restoration. Two separately ignored heavy historical integration endpoints were not run. Controls cover lambda-one terminal/cutoff telescoping,unchanged first sampled observations/actions/likelihoods,changed targets,ordinary-baseline equality,persistent grouping,and native environment/evaluation isolation.

Independent analysis verifies all **9 current ZIP digests and316 manifest-listed members**,plus the prior archives and members directly used by the gate and learning readers. All24 regular actor/critic checkpoint pairs are finite and correctly shaped; each baseline checkpoint reproduces the archived nonlinear weights exactly at0/1M/4M. Both arms have identical initial weights for each seed. All expected-unchanged native robustness records reproduce exactly. Baseline short records are exact for seeds201–203; seed204 has260 floating-field differences across its three checkpoints,maximum undiscounted1.91018e-4 and discounted3.73772e-5,with identical lengths and ending labels. No numerical rerun or seed replacement was used.

All65,536 measured training-update records satisfy exact chronology and transition budgets; all2,304 short and3,584 robustness episode identities,recipes,endings and costs are validated. These offline checks do not regenerate unexported optimizer moments,training gradients or every training transition. Snapshot weights are not exact-resume files.

For stage1,the predetermined first replication in each case/mode/branch retains56 full traces and111,183 transitions. Independent reconstruction verifies dynamics within1.192e-7,rewards within1.537e-7,actor latent inference within1.926e-7,critic inference within4.878e-5,discounted totals within4.690e-13,and lambda-one credit within2.153e-13. The telescoped-return identity agrees within2.135e-13. Other replications retain aggregate measurements,not full trajectories. All17 independent analyzer tests pass.

The first offline reader used `both` for simultaneous failures, but the actual native and short evaluators serialize `angle_position`. Both reader failures are retained; the reader was corrected to the actual source contract and a combined-ending regression test added. No measured outcome,threshold or trained policy changed. Main CI had no failed setup or measurement attempt.

Training/runtime versions: Rust1.98.1,Burn0.20.1 with the committed lockfile; Python3.11,Torch2.8.0+cpu,NumPy2.2.6,Gymnasium1.3.0,SB32.9.0. Gate statistics use SciPy1.17.0. Offline Python/NumPy/SciPy versions are recorded separately. SB3 is only an evaluation holder for these actual native training runs.

Main learning costs: **33,554,432 training transitions;1,591,447 short-evaluation transitions;32,235,407 robustness transitions;1,048,576 actor and1,048,576 critic Adam steps;134,217,728 gradient-sample visits per network**. The fixed-witness phase adds **14,034,199 evaluation transitions** and no training. Generic preflight/test interactions are additional and not fully aggregate-counted. No efficiency or all-inclusive cost claim.

## Evidence and disposition

Self-contained conversation bundle: `rustrobotics-ppo-lambda-credit-evidence.zip`. It contains all unchanged current raw archives,previous evidence nested within them,the exact audit sources/native binaries,all saved weights and episode outcomes,gate data,independent analyzer/tests,reader-failure logs,provenance,summary and replay instructions. Offline replay never loads the archived native library or starts training.

Keep production lambda0.95 unchanged. Retain lambda1 as a successful fixed-witness correction but a rejected standalone learning candidate under this tested budget. The old linear-task negative results,corrected-plant regression results and PR38 integration fix remain distinct. No new held-out learning,hardware qualification,merge or deployment occurred.
