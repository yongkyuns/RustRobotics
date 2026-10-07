# Critic conditioning and matched LQR benchmark

RustRobotics · October 7, 2026 · Completed diagnostic, not a validated PPO correction.

## Result

The selected critic has a concrete scale/optimization problem: a bounded tanh representation sits against its current output-head ceiling while almost all fitting targets lie above it. Fixed value normalization prevents saturation and greatly improves isolated fitting from an unsaturated initialization. But accurately fitting the historical bootstrap targets still does not produce accurate continuation values. Retrofitting normalization to the already-saturated critic is not reliably better in this test.

A separately derived, untuned LQR completes **256/256 ten-second cases** and **508/512 fifteen-second cases with the incoming PPO policy's additional Gaussian action noise**. Four stress failures are retained. This supports feasibility on the tested nonlinear noisy plant without learning; it is not global-stability or hardware qualification.

**No new PPO actor training, production/default/reward/reset/noise changes, deployment or merge. #33/#35 remain open.**

## Provenance and protocol

Continue the completed seed204/update98 witness [6035533421](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6035533421). Original ZIP `ppo-beta0-update98-evidence-20261007.zip`: 206392423 bytes, SHA256 `22e3943f5e0f4ac0209e37003c0602807b661f357940492de770d974fae5af84`. This session verified all3443 publication payloads, both nested input archives, byte-identical numerical results and all34 original tests before new analysis.

Structural inspection was post-hoc. Before any fitting or LQR outcomes, the arms, budgets, split, scale, controller design and cases were frozen in [6036231942](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6036231942). The setup-only episode-boundary correction was recorded before fitting in [6036273138](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6036273138). No completed outcome was retried, replaced, extended or selected for success.

## Current output ceiling and state discrimination

For a tanh last hidden layer with output weights held fixed:

`V(o) = w^T tanh(h(o)) + b <= b + sum(abs(w))`.

This bounds the **current output head**, not the network's overall representational capacity; changing weights changes the bound.

| Incoming update98 measurement | Value |
|---|---:|
| Current output-head ceiling | 514.901702 |
| Prediction mean / SD on 2048 fitting observations | 514.726584 / 0.738063 |
| Mean headroom below ceiling | 0.175118 |
| Original bootstrap-target mean / SD | 535.313406 / 15.904891 |
| Targets exceeding current ceiling | **2045/2048** |
| Second-hidden-layer activations with absolute value >.99 | **99.7032%** |
| Mean second-layer tanh derivative | 0.000611 |

The first hidden layer is not similarly saturated. With the head fixed, even an arbitrary tanh hidden representation cannot reduce target RMSE below approximately25.877. The head can still grow; this is an optimization/discrimination issue, not an inability to represent large values mathematically.

On the64 archived continuation states, the critic averages514.693 versus independent old-policy stopped return655.061. Critic RMSE to the conditional row means is161.958; removing its mean error still leaves80.793 RMSE, versus return-row SD80.745. This is not just a missing constant offset. The historical bootstrap targets on that same panel average537.248, with RMSE144.233 and mean-centered RMSE83.206: those labels themselves do not encode accurate continuation values on this panel.

All129 saved critics were characterized on the same fixed update98 observation panel. Mean/SD/saturation progress from .025/.317/0% at initialization, to339.200/4.445/96.74% at64, 514.727/.738/99.70% at97 and521.484/.112/99.88% at98. At128 the SD recovers to86.970 and saturation decreases to93.84%; that contrary later behavior is retained. These are not contemporaneous-rollout statistics.

The MC reference has64 fixed physical states/observations,16 independent future draws each and8192 stopped ticks at gamma.999. It is not infinite-horizon truth, posterior integration over latent states given observation, or uncertainty across independent trained agents.

## Four-arm isolated fitting experiment

This is **critic-only regression against the original update98 bootstrapped targets**, not PPO continuation. No actor changes, new rollout data, MC training labels, architecture changes or input scaling.

Cross archived initial versus incoming critic weights with raw versus normalized value units. Normalized units use fixed `S=1/(1-gamma)=1000`, physical prediction `V=S*v`, and targets divided by S. Divide the initial output-head weights/bias by S so physical starting predictions match. Physical target meaning and reward are unchanged; optimizer conditioning intentionally changes.

Every arm has fresh Adam, lr3e-4, betas(.9,.999), eps1e-5, half-MSE, norm clipping.5, identical minibatches64, and exactly2560 optimizer steps; retain checkpoints0/160/640/2560. The incoming-weight arms are not historical optimizer resumes. New fitting runtime is Python3.13.5/Torch2.10.0+cpu/NumPy2.3.5, not a claim of exact historical Torch2.8 replay.

Actual episode starts are rows175,431,...,1967. Train new source episodes0/2/4/6 (1024 rows), validate1/3/5/7 (849 rows), exclude the preceding175-row partial episode. There are27 MC panel rows from validation episodes. The first preflight incorrectly assumed rollout-aligned episodes and stopped before fitting; its failed source/log and pre-outcome correction are preserved.

### Fixed endpoint, all arms

Errors are in physical return units; bootstrap-validation error and independent-return error answer different questions.

| Initial critic / units | Bootstrap train RMSE | Bootstrap validation RMSE | Hidden saturation | Independent return RMSE,27 validation rows |
|---|---:|---:|---:|---:|
| Fresh / raw |469.738|**483.897**|**100.00%**|594.279|
| Fresh / normalized |6.205|**19.619**|**0.00%**|**145.496**|
| Incoming / raw |4.843|**17.760**|97.46%|**136.993**|
| Incoming / normalized |9.442|**18.441**|98.48%|**144.894**|

Scaling prevents saturation and helps fresh fitting at this budget. Nevertheless, the normalized fresh critic still underpredicts the independent validation returns by110.782 on average, with mean-centered error94.321. Scaling the incoming critic is **worse**, not better, than its raw counterpart at the endpoint. Earlier and later checkpoints are all retained.

This distinguishes conditioning from historical target accuracy. It does **not** establish that normalization throughout online learning would fail: an online normalized learner could generate different, better bootstrap targets. That from-scratch PPO comparison has not been run here. Nor does it uniquely attribute every previous collapse to one mechanism.

## Matched LQR benchmark

One model-based law was derived without a cost/gain sweep from the archived upright plant linearization and RK4 step. Cart mass1kg, pole mass1kg, length2m, g=float32(9.81), dt=float32(.01). Original quadratic reward costs are `Q=diag(.2,.02,1,.05)`, `R=.001`, with gamma.999 and successor-state cost accounted for:

`Qd=A^T Q A; Rd=R+B^T Q B; N=A^T Q B`.

Discounted discrete Riccati feedback, using noisy observations only and force clip±20N:

`u=-K*observation`, `K=[-12.326826,-18.432033,148.100494,61.271130]`.

Riccati residual6.27e-12; local linear closed-loop spectral radius.985194. These do not certify the nonlinear saturated stochastic system globally.

Before LQR runs, four archived command traces replayed exactly through the preserved original native library over5000 transitions. The evaluation retains original reset/noise/failure laws and the same eight update98 confirmation domains. Ten-second mode has native plant/observation noise. Fifteen-second mode additionally uses the incoming PPO Gaussian sigma.791463 in normalized-force units—15.8293N before clipping—with original Gaussian streams. That is explicit stress matching, not an LQR-learned exploration distribution.

| Controller | Ten-second completion | Fifteen-second completion with policy noise |
|---|---:|---:|
| Archived PPO before98 |254/256|490/512|
| Archived PPO after98 |166/256|174/512|
| **LQR, no learning** |**256/256**|**508/512**|

PPO counts reuse original verified evidence; LQR counts are new, with matching initial states and keys. LQR finite gamma.999 mean reward sums are611.332 and608.822 for the two different horizons. All four stress failures are position limits: panel/episode2/50,3/8,6/23,6/36 after64,78,70,61ticks. Full traces remain. LQR uses model knowledge unavailable to PPO; this proves neither global feasibility nor a successful learning recipe.

## Verification, cost, recovery

The independent reader checks all129 archived critic measurements, four fitting arms/16 snapshots, actual episode splits, minibatches, physical initial equivalence, network forward predictions, saturation/errors, and768 LQR cases. It reconstructs case keys/initial states, noisy-observation commands, Gaussian draws, force bounds, rewards, physical endings and discounted scores. All18 new corruption/algebra/consistency tests pass. It does **not** independently replay Adam or simulate native dynamics.

New work: **10240 critic-only optimizer transactions**,655360 training-example presentations; **1018273 LQR native transitions +5000 exact-control transitions =1023273 native calls**. **Zero new PPO actor-training transitions/updates.**

Recoverable evidence: `ppo-critic-conditioning-evidence-20261007.zip`, **71593129 bytes**, SHA256 **ce82074556bcb5c89dd44ac28f9a70e52c64dfd2c8d911b816d978f784c7f341**. Fresh extraction verifies414 payloads and368 byte-identical selected historical inputs against the original manifest, reproduces VERIFIED.json byte-for-byte and passes18 tests. Offline verification executes no native simulation, optimizer, Torch or pickle loading. The separate earlier34-test audit needs the original full witness ZIP; its successful log is retained.

The package contains full REPORT.md, exact measurement source `code/critic_conditioning.py` (SHA256 **acc2a7ed9e69ff8730b694352ec08ceded000b5db534c1cf243e5aecc83c9ad0**), independent reader/tests, all four fits/checkpoints, all eight LQR panels/full traces, original native binary/source receipts, selected prior weights/batch/MC/evaluations, setup failure and reproduction instructions. Third-party runtime installers are excluded. Do not assume an earlier chat sandbox exists; recover the named conversation artifact and verify its digest.

## Next bounded question

Test value normalization **from random initialization throughout ordinary PPO**, retaining raw-unit-correct bootstrap/advantage semantics, frozen budgets/checkpoints and independent continuation-credit diagnostics. Healthier activations and lower fitting loss alone are insufficient. No supplied LQR actor, mandatory critic pretraining, oracle labels or selected rollback checkpoint substitutes for reliable learning. A full controlled from-scratch comparison and new held-out qualification remain required before changing defaults.
