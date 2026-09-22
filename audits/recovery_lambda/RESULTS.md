# Recovery-data lambda factorial — lambda 0.95 rejected

September 22, 2026. Issue #35. Canonical run **35662335415**, source **f022f63d3c6541222650bbafab4ac47373cf43f3**.

## Decision

The clean recovery-data comparison rejects lambda0.95 as a replacement for the lambda1 development recipe. Lambda0.95 performs substantially worse on sustained nominal balancing and outward recovery, introduces a complete learned-policy collapse in seed41002, and fails survivor centring.

This does not invalidate the fixed-data diagnosis that lambda1 produced a trajectory-common harmful gradient at seed41006/update4140. It shows that changing lambda alone is not the solution to the broader finite-sample action-credit problem.

## Execution validity

All17 workflow jobs passed. Eight paired development seeds41001–41008 trained from random initialization for4096 updates.

Every lambda1 arm exactly reproduces the immutable fresh recovery-union predecessor:
- actor+critic bytes at checkpoints0/256/1024/4096;
- all4096 update records;
- all960 evaluation records.

For every paired seed, initial actor and critic bytes are identical. Config JSON differs only by arm label and lambda. At update1, fitting observations, latent actions and old log likelihoods are identical row-for-row; returns and raw advantages differ as intended.

The lambda.95 implementation recomputes both corrected-primary and supplemental GAE targets at.95. The collection domains, gamma.99, reward, plant, noise, sampled tails, eight supplemental streams,1024 fitting rows,minibatch256,four epochs and16 actor+critic Adam transactions per update remain unchanged.

All16 current measurement archives have valid internal manifests.

## Final checkpoint4096

| arm | five-minute deterministic | five-minute stochastic | sixty-second outward |
|---|---:|---:|---:|
| lambda1 control | **502/512** | **504/512** | **435/512** |
| lambda.95 candidate | **417/512** | **418/512** | **363/512** |
| candidate-control | **-85** | **-86** | **-72** |

The candidate-minus-control outward paired-history mean effect is **-14.0625 percentage points**. The ordinary two-sided99% Student interval is approximately **[-42.06,+13.93] percentage points**. Therefore the predeclared requirement for a positive adjusted99% lower bound fails even before multiplicity adjustment.

Mean outward discounted-return difference is approximately **-2.553**.

## Per-history sustained results

| seed | det lambda1->.95 | stoch lambda1->.95 | outward lambda1->.95 |
|---|---:|---:|---:|
|41001|64->64|62->64|62->64|
|41002|60->0|63->0|41->0|
|41003|63->61|64->62|59->48|
|41004|63->64|63->64|59->63|
|41005|63->50|63->50|47->38|
|41006|63->59|63->58|49->35|
|41007|62->62|62->62|60->61|
|41008|64->57|64->58|58->54|

Lambda.95 improves outward completion in41001,41004 and41007, but loses much more elsewhere.

### Seed41002 is genuine catastrophic forgetting

No NaN/Inf or failed optimizer transaction occurs. Actor/critic snapshots and every recorded policy/value loss remain finite.

On the fixed short panels, lambda.95 seed41002 is still learning at update1024:
- deterministic nominal **47/64**;
- stochastic nominal **50/64**.

By update4096 both are **0/64**.

At final:
- five-minute deterministic:62 position failures +2 angle failures;
- five-minute stochastic:63 position +1 angle;
- outward:64 position failures.

Thus this is a learned-policy collapse, not a numerical abort.

## Survivor quality

All lambda1 long-panel survivors satisfy the retained final10s centred/upright condition:
- deterministic502/502;
- stochastic504/504;
- outward435/435.

Lambda.95:
- deterministic **90/417** centred survivors;
- stochastic **95/418**;
- outward **90/363**.

Therefore the candidate fails not only completion reliability but also the centring property among many nominal survivors.

## Failure modes

Lambda1:
- deterministic:10 position failures;
- stochastic:8 position;
- outward:77 position.

Lambda.95:
- deterministic:93 position +2 angle;
- stochastic:93 position +1 angle;
- outward:149 position.

## Interpretation

The fixed update4140 arithmetic showed lambda1 can amplify trajectory-common GAE offsets and rotate one harmful gradient away from a conditional-action-credit reference. This from-scratch comparison demonstrates that simply shortening GAE to.95 can create different and worse learning failures.

The remaining hypothesis is broader: too few independent future trajectories can produce poor finite-batch action credit under either lambda. The repository already has a proper shared-policy multi-environment collector; its variance effect is being tested separately without training.

Production source/defaults, learned weights, PR38/master and deployment are unchanged.
