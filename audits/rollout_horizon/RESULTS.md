# Actual-rollout horizon and optimizer attribution

September 19, 2026. Issues #35 / #33. Corrected nonlinear production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Finding

**The short-window limitation is real: lambda1 can misrank the first-action change made by an actual PPO update when evaluated with that sample's remaining rollout horizon.** On the same continuations, replacing that short horizon with a full512-step window repairs all three confirmed reversals for the actual lambda1 update. This is not evidence that every late sample is wrong or that those three states explain the entire previous learning regression.

The actual optimizer presents a separate limitation. All four examined lambda1 updates improve their full-batch clipped surrogate, yet two have confidently negative discounted-return effects from fresh noisy resets. The third effect is negative but inconclusive; the fourth is positive but inconclusive. Correctly optimizing this finite batch is not sufficient evidence of policy improvement.

**No production correction or default is promoted.** This study replays four existing training histories, examines one predetermined actual update per seed, and makes an isolated same-input lambda0.95 counterfactual. It is not a new held-out learning comparison, a controller replacement, or a complete causal explanation of the earlier four-million-step results. PR38/master/deployed assets remain unchanged and unmerged.

## Protocol and execution

[Protocol recorded before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5739186854). Branch `audit/ppo-rollout-horizon-20260918`. Successful executed commit **b439cc1dd5c9ccb0075407ebbf6cbaaaeeedb13d**, [run35420390860](https://github.com/yongkyuns/RustRobotics/actions/runs/35420390860): all five jobs completed successfully, comprising preflight and seeds201–204.

Each ordinary lambda1 trainer replays from its original random initialization through update8191 with persistent Adam, environment, partial episode and RNG state, then captures the actual512-transition rollout for update8192, ending at4,194,304 transitions. The original collector, normalization, loss and optimizer execute unchanged, apart from test-only observation hooks. Physical states are captured only for diagnostic continuation initialization, never fed into training.

The actual lambda1 endpoint actor AND critic reproduce the previous lambda-one experiment's4M weights byte-for-byte for every seed. This binds the measured update to the historical training failure rather than creating a different convenient trajectory.

From the same incoming networks, both Adam histories, parameter identities, rollout and shuffle cursor, the original optimizer is executed with (1) actual lambda1 targets and (2) reconstructed lambda0.95 targets. A third execution restores the original anchor and repeats lambda1, requiring identical weights/metrics. Anchors clone the live modules and optimizer records internally; these are not weight-only warm starts. A separate native control verifies the next identical optimization also agrees, testing preservation of moment history beyond immediate weights.

All16 minibatch transactions per path retain indices, original pre-step actor/value losses, post-step full-batch action means, and actor/critic weights. Both paths use identical epoch permutations. The alternative lambda0.95 update is conditional on a lambda1-trained incoming policy; it is NOT a separately trained lambda0.95 policy or a proposed adaptive switch.

Settings remain one512-step environment,128 minibatch,four epochs,2x64ReLU networks,Adam epsilon1e-5,learning rate0.0003,gamma0.99,existing noise/rewards/exploration and the corrected nonlinear plant. There is no diagnostic outcome in an update, new reward, controller, extra critic fitting, artificial gradient, acceptance/rejection rule or checkpoint selection.

## Actual rollout coverage and continuation design

Indices were fixed before measurements:0,64,128,192,256,320,384,416,448,480,496,504,508,509,510,511. This deliberately oversamples the tail and must not be reported as the prevalence of errors throughout training.

All four captured rollouts happen to contain one live path with no true terminal or timeout before their final buffer cutoff. Thus the remaining horizons are512-index, from512 down to1. Termination/timeout correctness remains tested in controls, but this measured sample does not exercise those endings inside the captured batches.

At each of64 selected states,128 paired replications compare three first-action policies: old mean, actual lambda1-updated mean, and conditional lambda0.95-updated mean. The original noisy observation and physical state are held fixed. All branches share the first-action normal innovation and environmental randomness, and follow the SAME OLD stochastic policy after the first action. The perturbation is the actual learned change, not a forced minimum action difference. Every branch runs2048 steps or true terminal without reset.

Both lambda0.95 and lambda1 credit are computed from the same continuation at the original remaining horizon and at a full512-step horizon, using the old critic. Actual2048-step discounted rewards, without a critic bootstrap, supply the finite return reference. A surviving branch at an earlier comparison horizon retains a bootstrap; an earlier true failure removes it. Original external timeouts are not falsely treated as physical failure.

All24,576 fixed-state continuation branches reach2048 steps. Consequently, the confirmed discrepancies below concern return ranking, not first-action-induced differences in completion. Full traces cover the fixed first replication of every state/branch. Other repetitions retain aggregate returns and credit decompositions.

Classification uses paired Student intervals with99% Bonferroni family coverage over640 predeclared contrasts:64 states times two actual-return and eight credit contrasts. Both intervals must exclude zero with opposing signs for an opposite classification. The Student approximation and finite selected-state panel do not imply exact global coverage or a training-population failure rate.

## Credit rankings for the actual lambda1 actor movement

| Estimator/window | Opposite sign | Same sign | Inconclusive |
|---|---:|---:|---:|
|lambda0.95, original remaining horizon|2|29|33|
|lambda1, original remaining horizon|3|28|33|
|lambda0.95, full512-step horizon|2|28|34|
|lambda1, full512-step horizon|0|37|27|

All three actual-update lambda1 reversals become confidently same-sign with the longer window. The27 inconclusive full-window cases are retained; zero confirmed reversals does not mean all64 cases passed.

The reversals occur at64 and32 remaining steps, not universally at the final1–8 steps. Among the20 deliberately sampled1–8-step cases, nine are confidently same-sign and eleven inconclusive for lambda1; none is confidently opposite. Do not turn the measured boundary effect into a claim that the shortest sample must always be the worst.

### All three actual-update lambda1 reversals

Changes are updated first action minus old first action, with the old policy used thereafter.

| Seed / original index | Remaining rewards | Actual discounted-return change | Original-window lambda1 credit | Full512 lambda1 credit |
|---|---:|---:|---:|---:|
|203 /448|64|+0.000160149|-0.000185859|+0.000159203|
|203 /480|32|+0.000117161|-0.000097922|+0.000116350|
|204 /480|32|-0.000512101|+0.000537041|-0.000512621|

For204/index480, the actual return interval is[-0.000838654,-0.000185548], the short-credit interval[+0.000218045,+0.000856037], and the full-credit interval[-0.000839171,-0.000186070]. For203/index448, the respective intervals are[+0.000111245,+0.000209054],[-0.000231260,-0.000140457], and[+0.000110349,+0.000208057]. These small absolute effects are resolved in this paired diagnostic, not automatically large enough to explain every practical failure.

### Where the sign reversal enters

At204/index480, the32-step difference decomposes into:

- sampled discounted rewards: +0.000137544;
- discounted cutoff-value prediction: +0.000399497;
- identical initial-value baseline cancels;
- total estimated credit: +0.000537041.

The measured2048-step return change is-0.000512101, so the measured contribution after those32 rewards is approximately-0.000649645. The learned cutoff-value term instead predicts+0.000399497. Its error accounts for the discrepancy between short credit and measured finite return. With512 sampled rewards, the bootstrap difference shrinks to about1.05e-7 and the credit is correctly negative.

At203/index448, the64-step sampled reward contribution is-0.000051493, while the measured later contribution is+0.000211642. The bootstrap predicts-0.000134366, again assigning the wrong sign to the unresolved future. The longer window repairs the ranking. These are errors of the conditional cutoff-value differences on the tested continuations, not merely a common offset in initial value.

The conditional lambda0.95 actor movement gives a consistent but distinct control: original-window lambda1 has4 opposite/28 same/32 inconclusive rankings, versus0 opposite/38 same/26 inconclusive with512 steps. Original-window lambda0.95 gives3 opposite/24 same/37 inconclusive. Changing to lambda0.95 does not make every local credit correct.

## Actual optimizer improvement versus full-policy return

The separate reset test evaluates the WHOLE old, actual1 and conditional95 policies on128 paired fresh noisy reset episodes per seed, capped at2048 steps. It is not the first-action-only continuation test. Discounted-effect intervals use a separate eight-comparison99% family; no combined study-wide99% claim is made. Undiscounted returns and completions are also retained descriptively.

| Seed | Actual lambda1 full-batch surrogate gain | Actual lambda1 reset-return effect, adjusted interval | Conditional lambda0.95 reset-return effect, adjusted interval |
|---|---:|---|---|
|201|+0.005366764|-0.229643 [-0.355721,-0.103565]|+0.124475 [+0.009407,+0.239542]|
|202|+0.008808024|-0.250449 [-0.521460,+0.020561]|-0.236626 [-0.479869,+0.006617]|
|203|+0.000597122|+0.032948 [-0.007575,+0.073471]|+0.217125 [-0.097165,+0.531415]|
|204|+0.002070790|-0.175421 [-0.292044,-0.058799]|+0.043916 [-0.078011,+0.165843]|

All four actual lambda1 updates improve the scalar-reconstructed full-batch clipped surrogate. All four also have positive initial surrogate directional derivatives along their final parameter displacement. Nonetheless201 and204 have confidently negative independent reset effects. The other two actual effects are inconclusive. This is not a new arithmetic defect in Adam or proof that decreasing the empirical objective caused the harm.

Seed201 provides a particularly useful conditional witness: using the same initial weights, Adam histories, observations/actions and minibatch order, changing the return/advantage construction changes a confidently harmful actual update into a confidently beneficial conditional update. This establishes target-construction sensitivity for that one update, not a qualified policy for all states or a prescription to switch lambdas during training.

Reset completions old/actual1/conditional95 are201:77/76/80,202:100/95/95,203:98/99/100,and204:101/101/103 out of128. Discounted harm does not require a large completion-count change, and a few additional completions do not establish reliability.

The two kinds of evidence do not fully coincide. Seed201 has reset harm but no confirmed short-lambda1 sign reversal in the selected first-action panel. Seed203 has two confirmed short-window reversals, yet all16 selected actual first-action shifts are independently beneficial and its whole-policy reset effect is inconclusive. Different state distributions, finite correlated samples and the simultaneous network change remain important. Do not identify the cutoff bootstrap as the sole cause of every actual update or the whole prior lambda1 training regression.

## Whole-batch and normalization diagnostics

All2048 captured training rows and128 real optimizer transactions across the two paths/four seeds are retained. The original lambda1 and reconstructed lambda0.95 returns, raw advantages and global normalized advantages reproduce bit-for-bit with independent sequential float32 arithmetic. Every ordinary epoch permutation covers its512 rows exactly, and both conditional paths use identical indices.

Full-batch surrogate changes and initial parameter-direction projections are reported in four remaining-horizon bins:1–8,9–32,33–128,129–512. They demonstrate that tail rows are not the only contributors. For example, the actual201 surrogate gain decomposes into+0.000207,-0.001923,+0.003155,+0.003928 respectively. In204 it is-0.000131,-0.000946,-0.000278,+0.003425. The much larger population of long-window samples can dominate the net objective.

The exact stored normalized advantages define the primary derivatives. Raw-advantage and batch-mean subtraction projections are retained separately. These are local derivatives along actual recorded parameter changes, not the motion of an equilibrium, global policy improvement, or an intervention removing normalization. Their finite-difference checks agree within2.15e-10. No normalization modification or statistical-variance attribution was tested here.

## Verification, failures and scope

Repaired preflight passes **72 native unit tests**, seven ordinary balancing controls, nine ordinary learning/evaluator controls, strict Clippy, formatting, source restoration and compiled endpoint discovery. The diagnostic endpoint is invoked explicitly; two separately ignored historical heavy integration tests are not counted as executed.

An initial compile-time mistake in the diagnostic trace writer called the `truncated` field as a method. Run35420267748 failed before any main measurement. Its original67-member artifact is preserved. Preparation was corrected to use the actual field and to verify captured latents against the batch. No seed, policy, budget, threshold or experimental outcome was changed. Repaired measurements each execute once; no favorable rerun occurs.

Offline verification checks all four result ZIP digests and652 original members, four nested historical lambda1 archives and100 members, plus72 successful-build and67 failed-preflight members. It checks26,112 outcome records, all fixed panel keys and pairing, horizon/end semantics,2048 captured rows, both optimizer traversals and historical endpoint bytes. A complete set of204 preselected first-replication traces contains407,470 transitions. Reconstruction maxima: nonlinear dynamics2.73e-7,state-action inference1.63e-7,critic inference3.99e-5,reward1.66e-7,credit3.0e-13,and episode totals6.3e-13. Stored losses independently reconstruct within4.91e-8(actor) and2.02e-6(value); post-step mean error is below4.65e-8.

The first offline reader treated shortest-roundtrip float32 initial-value text as exact float64, causing a3.59e-6 decomposition discrepancy. Decoding that field to float32 before promotion restores the original strict tolerance. The failed reader log is retained and a regression test added. **23 independent analyzer tests pass.** No measured field or tolerance was relaxed.

Complete trajectories are retained only for the predetermined first replication; the remaining replications have aggregate records. Offline checks do not regenerate every unexported RNG draw, gradient or Adam moment. Moment preservation is checked by the actual native restore/replay and next-identical-update controls, not inferred from weight-only files. Portable continuation inference is independently checked but is not asserted identical to all Burn tensor reductions at arbitrary points.

Actual measurement uses Rust1.98.1/Burn0.20.1 with the pinned production lockfile. Offline Python3.13.5/NumPy2.3.5/SciPy1.17.0 is separately recorded and never loads the native executable or trains. The full platform/browser matrix is not rerun for this audit-only branch.

Costs: **16,777,216 prefix/target training transitions**, **50,331,648 fixed-state continuation transitions**, and **2,393,098 reset-evaluation transitions**. Per network there are524,224 prefix Adam steps plus192 target steps, including the conditional and sham replays. Generic preflight training/control interaction is additional and not all aggregate-instrumented. These replays are not four new independent training successes, and no efficiency claim is made.

## Disposition

Retain the short-horizon reversals and the actual201/204 harmful-update witnesses. A full-window first-action test cannot qualify every sample in a fixed-length rollout; conversely, the boundary effect does not uniquely explain the learning failure. The next correction must test usable reward support near collection boundaries and actual update generalization together, without treating a lower loss or repaired isolated witness as robust control.

Self-contained conversation evidence: `rustrobotics-ppo-rollout-horizon-evidence.zip`, containing unchanged successful/failed raw archives, historical inputs nested inside, exact prepared source/native runner, all captured batches/weights/outcomes, independent analyzer/tests, full tables, and offline replay instructions. Production lambda remains0.95; PR38/master/defaults/deployment are unchanged. No new held-out training or hardware qualification was performed.
