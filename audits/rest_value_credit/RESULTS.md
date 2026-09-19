# Rest-point value accuracy versus action credit

September 18, 2026, America/Toronto; execution timestamps are September 19 UTC. Issues #35/#33. Corrected production baseline: PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Finding and limits

**A critic can predict the level of return closely while giving the wrong local action ranking.** On the actual corrected nonlinear noisy environment, one-step TD ranks the tested first-action change incorrectly at12/14 selected stable resting points. The512-step GAE estimator used by the learner repairs many of those errors, but still has3 confirmed opposite-sign rankings,9 same-sign rankings and2 inconclusive rankings under the predeclared multiplicity-adjusted intervals.

A separate analysis of the actual saved update immediately after4M shows that the normalized batch surrogate opposes a precisely defined centring direction on3/4 seeds; the recorded actor movement has an opposing component on2/4. The seed2034M policy has both a confirmed wrong-sign local GAE witness and an opposing component in that next actual update. This is concrete mechanism evidence, not a unique reconstruction of the complete4M-to16M drift or proof that one change will solve training.

**No new learner modification, production default, merge, deployment or training qualification.** No supplied controller or critic-pretraining procedure enters training. No new full training runs were performed; ordinary unit-test training is additional preflight work.

## Protocol and provenance

[Noisy continuation protocol posted before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5738700472). A preliminary noiseless local Bellman-gradient calculation was explicitly post-hoc and came before this protocol; it is not described as blinded confirmation.

Audit branch `audit/ppo-rest-value-credit-20260918`; executed commit **66232d316de1dd6940e62aeda2efcd9b37afad7f**. [Run35416696724](https://github.com/yongkyuns/RustRobotics/actions/runs/35416696724) succeeded on its first attempt. The only executable addition is a separate read-only diagnostic using public production `PendulumEnv`, `PolicySnapshot` and dense-layer APIs. Protected production source and Cargo.lock match PR38 before and after preparation.

Weights and recorded update batches come from all eight unchanged late-actor-decay artifacts. All ZIP and member hashes are checked before use. The panel contains EVERY smooth locally stable upright rest inside the track for constant-rate4M and16M policies and decayed-rate16M policies:12 frozen actor/critic checkpoint pairs and14 roots. The two multiple-root policies contribute both roots. Root selection was not restricted to wrong-sign cases. Independent axis-region enumeration verifies coverage.

## Actual noisy counterfactuals

For each root, start both branches at the same physical upright rest and same noisy observation. Draw the same standard-normal first-action innovation in both branches. The first latent action is

`mu(observation) + 0.1*innovation +/-0.02`.

The action is the existing20*tanh(latent) bounded force. After that ONE action, both branches follow the same unmodified stochastic frozen policy, with paired environmental and action RNGs. Simulate2048 steps or true failure, with no reset inside a branch. Each root has256 independent paired replications. This is a local first-action intervention, not an entire translated-policy intervention or an actual Adam parameter step.

Record actual finite discounted return, one-step TD credit, and512-step GAE using the frozen learned critic. Gamma and lambda are the configured f32 values0.99 and0.95 promoted to f64 for diagnostic sums. GAE is decomposed into initial baseline, lambda-weighted direct rewards and future-value terms; its direct TD-residual sum must agree. True terminals remove bootstrap; the512-step GAE cutoff retains final-observation bootstrap. The2048-step measured return has NO critic bootstrap. These are finite-horizon results, not an assumed exact infinite-horizon value.

Actor/critic inference uses the portable f32 layer implementation; the deterministic zero-innovation actor path is checked against `PolicySnapshot.act`. No alternative Python plant is used. Portable inference is not claimed bit-for-bit identical to every Burn tensor reduction; the independent recorded-inference differences are reported below.

A separate deterministic/noiseless pair at every root is retained as a local countercheck, not pooled into stochastic intervals. The critic was trained on the noisy stochastic task, so the noiseless analytic result alone would not establish an error on its intended value target.

Classification uses paired Student intervals with nominal familywise99% Bonferroni coverage over14 roots x3 contrasts(return,TD,GAE). A wrong ranking requires both compared intervals to exclude zero with opposite signs. The finite-n Student approximation and the chosen fixed-state panel do not establish a population-wide critic failure rate.

## All noisy paired results

Changes are plus first latent action minus minus first latent action; all subsequent policy actions are unmodified. Positive values favor the plus branch. The value of12/14 TD failures must NOT be represented as12/14 GAE failures.

| Policy | Root x(m) | Measured discounted change | TD change | GAE change | GAE classification |
|---|---:|---:|---:|---:|---|
|Constant201,4M|-0.245076|-0.004351|-0.002454|-0.002721|same|
|Constant201,16M|-0.544222|-0.005191|+0.006049|+0.001341|OPPOSITE|
|Constant202,4M|-0.725945|-0.012094|+0.018059|+0.001553|OPPOSITE|
|Constant202,16M|-1.729481|-0.023485|+0.023905|-0.001807|same|
|Constant203,4M|-0.204559|-0.003179|+0.004587|+0.001199|OPPOSITE|
|Constant203,16M|-0.917878|-0.010257|+0.002350|-0.001122|same|
|Constant204,4M|-0.349995|-0.005347|+0.010168|+0.000190|inconclusive|
|Constant204,16M|-0.801103|-0.011490|+0.006005|-0.001544|same|
|Decay201,16M|+0.951568|+0.014140|+0.002185|+0.006337|same|
|Decay202,16M|-1.717986|-0.022143|+0.016951|-0.003652|same|
|Decay202,16M|+1.573498|+0.023154|-0.016928|+0.000480|inconclusive|
|Decay203,16M|+1.748667|+0.024690|-0.006573|+0.004425|same|
|Decay204,16M|-1.316953|-0.024958|+0.014322|-0.007965|same|
|Decay204,16M|+2.220085|+0.038554|-0.028428|+0.001989|same|

Comparing rows from different policies is not a same-state intervention on the training recipe. In particular, absence of a confirmed GAE reversal among the decayed policies' own selected rests does not rehabilitate their failed robustness results.

### Three confirmed GAE reversals

| Witness | Actual return difference and adjusted interval | GAE difference and adjusted interval |
|---|---|---|
|Constant201,16M|-0.00519086 [-0.00605834,-0.00432337]|+0.00134105 [+0.00045239,+0.00222971]|
|Constant202,4M|-0.01209380 [-0.01290788,-0.01127972]|+0.00155320 [+0.00059514,+0.00251126]|
|Constant203,4M|-0.00317873 [-0.00399844,-0.00235902]|+0.00119853 [+0.00040100,+0.00199607]|

All512 branches of each of these three witnesses reach the evaluation cap. The measured error is return ranking, not a difference in the completion count. None of these pairs demonstrates that a single perturbed action alone causes a rail failure.

## Why value MSE is an insufficient acceptance test

At the seed2034M rest, mean initial critic prediction over the noisy panel is **98.643018**, and the mean measured return across its symmetric first-action pair distribution is **98.706293**. The latter has the intentionally perturbed first-action distribution, so it is not asserted to be an exact nominal-policy value. Nevertheless, these similar aggregate levels coexist with a decisive reversal of the action comparison above. The policy does not need a large value-level error to receive misleading action credit.

For this witness, the GAE change decomposes as:

- direct lambda-weighted reward contribution: **+0.000177919**;
- future-critic contribution: **+0.001020615**;
- same initial observation/baseline cancels exactly;
- total GAE: **+0.001198534**, versus actual discounted return **-0.003178728**.

The short-weighted direct reward is already misleading here, and the future-value estimates reinforce it rather than correcting it. The other two confirmed reversals also have positive direct contributions despite negative actual return changes. Therefore this result must not be paraphrased as a future-value term reversing an already correct reward sign in every case.

The analytic noiseless check corroborates the distinction between value and value gradient. With held-input sampled dynamics J and B at a stable rest, and next-state reward derivative r_s, the deterministic policy-value gradient satisfies

`g = J^T (r_s + gamma*g)`.

The true local first-action derivative is `(r_s+gamma*g)^T B`. Substituting the learned value gradient produces the one-step TD derivative. Seed2034M gives true derivative **-0.00353862 per newton**, but learned-critic TD derivative **+0.00616372**. Its infinitesimal GAE derivative is also opposite, **+0.00211157**. Finite noiseless trajectories and the separate noisy test are retained, not conflated with this local infinite-horizon calculation. At seed2024M the noiseless infinitesimal GAE sign is correct but the noisy finite-action ranking is wrong; that contrary case demonstrates why the stochastic continuation test was necessary.

## The next actual saved actor update

The previous study retained complete observations, actions, old likelihoods, normalized advantages and pre/post latent means for update8193, immediately after the4M checkpoint. Its incoming actor/critic weights are also available.

Define a diagnostic network direction by translating only its position input: evaluate the old actor at `observation + epsilon*sign(rest_x)*e_x`. A positive epsilon translates that actor's zero-force rest towards the origin. This is an exact parameter direction, equivalently modifying first-layer biases; it was NOT installed or trained.

Using the recorded batch, independently differentiate its scalar clipped surrogate along this direction at the old policy. Validate the derivative with central finite differences, maximum error1.225e-10. Separately project the recorded post-minus-pre latent means onto the same position-input response direction.

| Seed | Normalized surrogate centring derivative | Recorded response component |
|---|---:|---|
|201|-0.0305492|against centring|
|202|-0.0011414|towards centring|
|203|-0.1102595|against centring|
|204|+0.0416433|towards centring|

The projection is NOT the actual displacement of the new network's equilibrium. It does not reconstruct all parameter changes, Adam moments or intervening4M-to16M updates. Seed202 is an explicit counterexample to equating the initial local surrogate derivative with the final minibatch/Adam movement.

An approximate target/current-baseline/normalization decomposition shows finite-batch sensitivity. For seed201 the raw centring derivative is+0.00095093; subtracting the batch mean advantage contributes-0.00330647 before division by its approximately0.07716 standard deviation, making the normalized direction negative. Reconstructing the raw components uses independent f64 value inference from stored f32 weights, not archived exact value tensors; normalized-gradient reconstruction error is at most2.072e-5. The exact stored normalized advantages supply the primary derivative.

This is not evidence that an action-independent baseline has nonzero expected score under the policy or that advantage normalization is universally a bug. A particular correlated finite batch can have substantial baseline and mean-subtraction projections. Removing either is not tested or promoted as a correction here.

## Verification and exact scope

Native preflight passed **7 new evaluator controls**, **68 existing trainer tests**, strict Clippy, release compilation and source-restoration checks. Controls cover malformed snapshots, portable actor parity, identical noisy action pairs, noiseless stationarity, lambda-one telescoping, true-terminal bootstrap removal, and an actually executed first-action perturbation. No deliberate learner defect or score-conditioned retry was introduced.

Current artifact **10576391630**, ZIP SHA256 **9b29ad8d9fc34f6c6eecc6141bae66ca31acd9eaaf5863a2d5e0459414d6b295**, size62,003,244bytes. All116 payload-member hashes verify; all8 nested source-policy archives and their360 original members also verify. The current ZIP retains those earlier archives unchanged, not replaced with summaries.

There are **7,196 outcome records**:7,168 noisy branch trajectories and28 separate noiseless trajectories. Full traces cover the predetermined first replication of every case/mode/branch:56 files and111,183 transitions. Other replications retain aggregate outcomes, not full step traces. Independent offline reconstruction of those exported traces has maximum errors: physical transition1.192e-7, reward1.537e-7, actor latent inference1.926e-7, critic inference4.878e-5, discounted total4.690e-13, GAE total1.746e-13. Every outcome's pairing, horizon/terminal fields and GAE decomposition are checked, but offline analysis does not independently regenerate every unexported RNG draw or full trajectory.

All **16 independent analysis controls** pass, including corrupted archive/member rejection, paired-statistic rules, sign classification, network derivatives, RK4/translation checks, and altered reward/credit rejection. Analysis uses local Python/NumPy/SciPy, while actual trajectories use the pinned Rust1.98.1 build. The offline replay never executes native binaries or performs training.

Main experiment cost: **zero training transitions**, **13,976,855 noisy continuation transitions** plus **57,344 noiseless transitions**. Native unit-test work is additional and not fully aggregate-instrumented. The14 roots are selected development states of previously exposed policies, not14 independent training runs or held-out hardware qualification.

## Disposition

Retain the wrong-sign witnesses and the actual-update centring projections as regression evidence for a future value/advantage correction. This study does not identify every harmful late update, qualify a new training recipe, or establish that changing lambda, removing normalization, enforcing symmetry, or further fitting would fix the controller. The integration corrections in PR38 remain separate and unmerged. Production defaults, master and deployed assets are unchanged.

Self-contained conversation bundle: `rustrobotics-ppo-rest-credit-evidence.zip`, containing the unchanged native result ZIP (with historical inputs nested inside), exact sources and binaries, analytical derivation/replay code, all outcomes and witnesses, tests, summaries, and provenance. Use its included offline replay instructions rather than assuming a previous container path.
