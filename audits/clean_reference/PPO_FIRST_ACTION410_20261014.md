# PPO410 paired first-action test — complete, no production correction

October14,2026. Primary issue35 protocol6176323497; explicitly post-result cutoff addendum6176597334. Continues calibration75545b6c. Full executed code, source states, trajectories, readers, tests and reproduction are in the evidence packages below. Master/defaults/protected sources/deployments remain unchanged; no PR or merge, #33/#35 remain open.

## Result

Changing only one decision from historical policy410 to411 did not reproduce lost recoveries on these panels. One-step critic estimates are much less accurate than256-step lambda1 estimates of the measured small action effects. This does not establish a bootstrap-induced sign reversal in the actual historical PPO update. It is a stock-SB3 reference diagnostic, not a newly measured production Rust optimizer update.

### Fixed source/branch design

64 new independent reset episodes per panel; one source age assigned cyclically from0,16,32,64 ticks,16 per age. Ordinary reset keys3810142600–3810142663. Difficult panel takes first64 qualifying native resets from3910142600 under unchanged abs(theta)>=.15,abs(omega)>=.25,theta*omega>0. It considers607 keys, ending3910143206. Both reset panels freeze before prefix collection. Policy410 drives every source prefix; all128 assigned source states exist, without replacement. All prefixes and3,584 source transitions retained; both source panels freeze before branching.

Eight paired action/future replicas per source. Future key4010142600+panel_index*100000+source_index*16+replica; innovations default_rng(key XOR0xFA141041).standard_normal(8192).astype(float32). Original additive bridge restores exact physical/noisy-observation state, starts branch age0 and resamples native future randomness; clone it for the paired arm. This is NOT an alleged exact hidden-RNG resume from a source prefix.

Only tick0 differs: policy410 versus411, each with its own Gaussian sigma and a common standardized draw. Every later action in BOTH branches uses410. Original noisy nonlinear plant, clipped actions, reward and physical limits; fixed64 inference lanes including inactive lanes; first physical ending or8192 ticks, no short timeout/reset and no final critic bootstrap. Exactly2048 trajectories /14,893,031 native transitions, zero training/optimizer steps.

### Primary outcomes

Eight replicas are averaged within each source BEFORE pointwise two-sided99% Student intervals across64 independent source episodes. Not512 independent source states or independently trained agents. Both actual-effect signs remain unresolved.

|Panel|Old-first complete|New-first complete|Gains/losses|Mean measured gamma.999 return change [99% interval]|
|---|---:|---:|---:|---:|
|Ordinary-source|497/512|497/512|0/0|−.002456 [−.005692,+.000779]|
|Difficult-source|411/512|411/512|0/0|−.008949 [−.022674,+.004776]|

All physical failures are cart-position-limit failures. Same ending labels do not imply identical ending times or rewards, nor equivalence of the policies. These are DIFFERENT panels from the earlier whole-policy experiment; the contrast is not a same-case proof that multiple changed actions are necessary.

Identical clipped first commands occur in67 ordinary and126 difficult pairs. All193 such pairs have byte-identical full trajectories and remain included.

### Critic predictions versus actual returns

One-step prediction is r0+gamma*1000*V410(o1), zeroing bootstrap at physical termination; the SAME initial baseline cancels. Secondary256-step lambda1 target uses actual prefix rewards and the frozen critic only at a live cutoff. Gamma=.999,1000 converts the existing scaled critic to physical reward units.

|Panel|One-step predicted change [99% interval]|256-step predicted change [99% interval]|
|---|---:|---:|
|Ordinary|−.049454 [−.095986,−.002922]|−.002628 [−.006001,+.000745]|
|Difficult|−.129383 [−.285045,+.026278]|−.009516 [−.023405,+.004374]|

Paired one-step prediction-minus-measurement discrepancies: ordinary−.046997 [−.092939,−.001056]; difficult−.120434 [−.274142,+.033273]. Corresponding256-step discrepancies: ordinary−.000172 [−.000416,+.000072]; difficult−.000567 [−.001031,−.000103]. No equivalence margin was declared; unresolved effects are not equivalence.

Descriptive mean absolute source-level discrepancy is.048776→.000325 ordinary and.212937→.000598 difficult for one-step→256-step. One-step sample-mean signs disagree with measured sample means at34/64 ordinary and26/64 difficult sources;256-step signs disagree at0/64 and1/64. These are finite-sample descriptive counts, not exact conditional advantage labels or multiplicity-adjusted findings. All zeros and contrary cases retained. For branches ending by256,256-step targets equal measured returns by definition; this must not be promoted as independent cutoff calibration.

Every target discrepancy is checked against the discounted cutoff-value-minus-measured-tail identity, per trajectory and paired contrast. The measured horizon is finite8192 ticks, not an infinite-horizon oracle. These first-action interventions are not the actual minibatch-normalized training advantage or an infinitesimal global gradient.

## Explicitly post-result cutoff extension — no new simulation

After the primary result, addendum6176597334 fixed K=[1,16,32,64,128,256] on the SAME saved trajectories. Pinned64-lane Torch critic inference at actual noisy endpoints, zero at physical endings; independent float64 checks. K1/K256 values reproduce original recorded values byte-for-byte. Correlated exploratory readouts, no new independent evidence or selected training cutoff.

Mean absolute source-level prediction discrepancy:

|Reward steps|Ordinary|Difficult|
|---:|---:|---:|
|1|.048776|.212937|
|16|.012295|.066372|
|32|.014170|.064524|
|64|.015523|.075407|
|128|.001264|.002910|
|256|.000325|.000598|

Not monotonic at every cutoff. All cutoffs retained; no policy/target change recommended from this curve.

### Actual archived batch0410 support

Archive f5f37761202e09faeebd6dfd51e785ee5f007bc7dd388d42c6c24ec5351eefec and its member digest checked. Independent reader validates packet shapes, observations/actions, reset markers, ages and state continuity.

The2048 rows have eight episode timeouts, zero true terminals and one live rollout cutoff. All29 samples in the PRE-EXISTING instantaneous early-outward stratum have166–256 reward steps before the next actual boundary; none has128 or fewer. Thus the one-step/very-short-target explanation is not demonstrated for those29 samples in the harmful batch.

That stratum is NOT all recovery/braking states:800 rows have age<100;53 OTHER early rows have64 or fewer reward steps and100 have128 or fewer. Short-target effects elsewhere remain open. Support counts do not establish value accuracy or causal contribution, and the fresh-state cutoff curve does not automatically calibrate historical targets.

## Verification and source

Pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0, original native/SB3 source-byte gates. Runtime ZIP c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and all11 payloads verify; prior compact package and all83 payloads verify and32 prior tests pass.

66 original historical and64 recorder trajectories re-execute byte-exactly, plus128 same-first native control trajectories. **All39 new tests pass**:30 math/genuine-fixture reader tests and nine historical-batch controls. Independent command discrepancy2.8211414e−7<3e−6; scaled value discrepancy2.9628718e−7<2e−6. Reader executes no Torch/native/optimizer/pickle and does not independently reintegrate dynamics or regenerate private native RNG draws.

Fresh final ZIP extraction verifies147 payload hashes, original control bytes,39 tests and **all five numerical outputs byte-for-byte**. A combined read-only REPLAY invocation timed out in the frontend after controls/tests; unchanged primary and exploratory readers then completed separately on the same extraction. No simulation was repeated for this verification. Detached verification receipt binds output/log hashes.

The standalone tested support reader is committed in first_action410_20261014/batch_support.py, Git blob c9e179e6aa0d1fd0ec671fd528f0dfad24d8e056, exactly matching packaged bytes. Full collector, reader and test sources are in the packages; not falsely claimed integrated into production or routine CI.

## Honest cost and interruption record

Known native transitions:14,893,031 main +90,653 historical +87,653 recorder +35,748 paired control +3,584 source prefixes =15,110,669. Cutoff readout adds zero transitions; no optimizer/training transaction.

A multi-group frontend timeout left broad02 incomplete with no archive/receipt. A recovery attempt was stopped by the create-only directory guard; the empty directory/logs were preserved, then only that fixed-key group was repeated. Its interrupted cost is unknown, bounded by1,048,576 transitions, not claimed zero or an independent case. A streaming request was rejected before execution. No observed outcome selected a retry. Full notes/receipts retained; no efficiency claim.

## Recoverable evidence and disposition

- Full ppo_first_action410_evidence_20261014.zip:672,781,104bytes, SHA256 **50cac988d748ad383ef0efc54e8cf51e6bc214fdef653a5ca5568c8ace32b9fc**.
- Compact ppo_first_action410_report_code_20261014.zip:22,728,620bytes, SHA256 **8f06775b326fddee32d94ce9f344bba0070b68c1f2d6f059595bab91d78a7b87**.
- Detached ppo_first_action410_verification_20261014.json binds both archives and final checks. The detached detailed report adds the final verification note to the original packaged report.

Full REPLAY.py validates all payloads/controls/tests and re-reduces all five outputs; compact replay explicitly omits raw branch reduction. Recorded read-only runtime Python3.13.5/NumPy2.3.5/SciPy1.17.0. README documents strict native regeneration and retained original input subset notices. Third-party runtime installers/fonts excluded.

Next attribution target is repeated policy changes across the recovery maneuver and their relationship to the actual warm optimizer update, not claiming a timeout-bootstrap fix from a one-step mismatch. Not claimed executed. Any correction still needs a declared from-scratch comparison and #33 sustained-balancing qualification. No production/default/task/protected-file/held-out-training-seed/deployment changes, PR or merge.