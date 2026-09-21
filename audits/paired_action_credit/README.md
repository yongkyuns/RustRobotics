# Paired action-credit correction — native preflight passed; measurements in progress

September 21, 2026. RustRobotics #35. Protocol comment **5760268403** was recorded before new correction outcomes. Current executable source **d41c9ea708b9c5c74e4b98db0ab1f3559674eb4c**, workflow **35599815447**, branch `audit/ppo-paired-action-credit-20260921`.

**Repaired native preflight PASSED. Both selected-history measurement jobs are executing after immutable-input verification.** No completed paired-credit measurement, candidate-policy score or controller-quality conclusion has been independently verified. Documentation-only updates do not dispatch duplicate measurements.

## Fixed correction and witnesses

Use the exact incoming state of the independently confirmed harmful update **seed41006/update4140**, retaining **seed41004/update4137** as the preselected comparison. Reconstruct their original support1024/gamma0.995 histories from random initialization with persistent Adam, parameter identities, environment, unfinished episodes and RNG streams. Historical archives are comparisons only, never loaded as warm starts. Prefix records/checkpoints and every selected original target, support record and optimizer transaction must reproduce the immutable regression-window evidence.

The candidate replaces only the ACTOR advantages on the exact original1024 fitting observations, recorded latent actions and old likelihoods. For each fitting row:

1. Retain the actual physical simulator state and its original noisy observation.
2. Generate eight independent pairs of1024-step continuations, or stop at the first true failure.
3. The first branch takes the recorded action; the second samples its first action independently from the old stochastic policy.
4. Both then follow that same OLD policy, using paired environmental noise and time-indexed action innovations. They need not command identical actions after their states diverge.
5. Average recorded-action return minus independently sampled-policy return, then apply the existing global advantage normalization once.

Returns use training gamma0.995 and the old critic only at the final nonterminal cutoff. Physical failures never reset within a branch and never retain a value bootstrap. The initial learned value is not subtracted. The reference action is not a teacher's preferred action, the mean action, or an optimization over candidate actions. No correct-action labels are supplied.

The state is used only to initialize the cloneable simulator; actor/critic inference retains ordinary noisy observations. This is privileged simulator access for constructing targets and is not directly a hardware training method. The finite window, old final bootstrap, eight baseline draws, global normalization and PPO clipping mean no exact-value, unbiased-gradient or guaranteed variance-reduction claim follows.

## Controls

Incoming actor/critic parameters, both live Adam histories and shuffle state are restored for the original and candidate fixed-data updates. Critic targets remain the original recorded targets; every critic transaction must remain identical. Fitting batch1024, minibatch256,four epochs and16 Adam transactions per network stay fixed. No new states enter fitting.

An original-update restore/replay must reproduce the original actor/critic/mean/loss records exactly. A subsequent identical fixed-data optimization is also compared, with native tests additionally checking the next real collected update. The capture hooks only observe primary states, selected supplemental states and the final prepared batch; they do not replace collection or optimization bodies. Recording is scoped and disabled for other uses.

All8192 pair outcomes per history, all1024 corrected raw/normalized advantages and all optimizer transactions are retained. Full credit traces are fixed rows0/511/512/767/768/1023, draw0, both branches. Other pairs retain aggregates, not complete unexported paths. Physical sample identity, true-terminal semantics and common-random-number pairing must be checked before interpreting results.

## Evaluation and decision

First replay the prior256-case independent before/after panels on their original0x20000000 random domain, requiring exact historical fields. Then test incoming/original-update/paired8-update policies on512 new paired cases per panel, using the independent0x40000000 domain. Panels are20.48-second nominal mean-action, nominal stochastic and outward stochastic, ending at the first failure. The unchanged evaluator scores both configurations under common gamma0.99; survival and total return remain available. Replication0 full traces are retained for each policy/panel.

Twenty-four predefined two-sided99% Student Bonferroni intervals cover two selected histories,three panels,completion/discounted return,and candidate-minus-original/candidate-minus-incoming contrasts. These are conditional evaluations of fixed selected policies, not confidence statements about a new training population.

Local screening requires a positive adjusted lower bound for target41006 outward completion versus its original update; no pooled nominal-count loss versus the incoming policy on either history; no resolved harmful candidate-minus-incoming completion/return contrast; and positive corrected-surrogate improvement with nonzero actor movement. All contrary or inconclusive results are retained. Passing would justify a later fixed learning study, not adoption or guaranteed safe updates. No checkpoint selection, rollback, extra draw selection or threshold adjustment occurs.

## Cost and scope

Each history replays its original prefix through4137 or4140 and executes one candidate update. Additional credit estimation is bounded by **16,777,216 simulator transitions per history**. Per history,1536 historical-check and4608 new evaluation episodes are separately counted. Original/candidate/sham and two next-identical controls use five fixed-data optimizer executions in total at the target. Generic preflight work is additional. This is not an equal-sampling-compute comparison, a from-scratch candidate cohort, or an efficiency result.

## Native checks and retained setup failure

The first source9e6c008f/run35598650303 failed preflight before any measurement. Strict Clippy correctly rejected two audit declarations that exposed the private `RolloutBatch` through a crate-wide interface. The repair changes ONLY those two declarations from `pub(crate)` to `pub(super)`, restricting access to trainer descendants. No production type was made public, lint disabled, numerical rule changed, or experimental setting altered.

The failed artifact **10637259662**,504759bytes,SHA256 **eb27c75dcb28e33a6c22a376657c5878111b23a7da0571747612efb89f28c9d0**, is preserved unchanged; all118 payload hashes verify. It contains the original failure logs and prepared sources. Both measurement jobs were skipped in that run. This is not a failed controller trial or an excluded performance outcome.

The repaired source passes **116 native unit/audit tests**, **7 ordinary balancing controls**, **9 ordinary learning controls**, strict Clippy, formatting and protected-source restoration. Eleven heavy audit endpoints are ignored in the ordinary unit pass; the current endpoint is explicitly invoked by the two measurement jobs. Two historical heavy integration endpoints and the production platform/browser matrix were not rerun.

Passing build artifact **10638520932**,2171167bytes,SHA256 **2e13f431b1e3027fdb2d02c59cb7c896d2987b0bb3c1015a4a707761e8275f84**, and all125 payload hashes verify. Its compiled capture source differs from the failed build at exactly the two visibility declarations. The new native correction, preparer and workflow are byte-identical between those builds. These implementation checks are not evidence of improved control.

## Offline checks actually completed

The prepared Python suite passes **140 tests**:109 inherited controls and31 new paired-credit identity,terminal/cutoff,random-domain,denominator,missing-data,and statistical-family tests. These are not new learning results.

The new optimizer reader was exercised on the TWO OLD selected transactions. It reconstructs update4140's original surrogate gain0.008415139 and KL0.005862713, and update4137's gain0.002703510 and KL0.007071463, under unchanged numerical checks. Existing primary/supplemental target reconstruction remains bit-for-bit on those old records. This is compatibility checking, not evidence for the new advantages.

With measurement artifacts absent, the verifier returns INCOMPLETE/exit2 and produces no controller scores. It is prepared to verify published ZIP/member hashes,source/history bindings,all original and corrected targets,critic equality,sham/optimizer outputs,pair accounting and every exported action/physics/reward trace. It has not been exercised on completed new measurement artifacts. Unexported Adam moments,Gaussian random draws and remaining complete paths are not independently regenerated.

The conversation progress package retains the passing and failed build archives, actual prepared source, verifier/tests, compatibility results and incomplete-status checks. Offline commands never execute the bundled native runner,network,simulator or training.

## Disposition

Production gamma0.99/lambda0.95,rewards,normalization,learned policy,PR38,master,deployment and fallback behavior were not modified. No merge or adoption. The earlier rejected gamma/support/normalization candidates remain rejected. This step establishes a passing native implementation and running same-input correction comparison; it does not yet establish improved recovery.
