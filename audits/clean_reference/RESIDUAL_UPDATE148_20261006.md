# Residual PPO update148: on-batch harm and a destructive optimizer tail

October 6, 2026, America/Toronto. Diagnostic evidence, not a corrected learning recipe.

## Main finding

The exactly replayed stock-SB3 physical-horizon candidate has a concrete residual witness: **seed204, update148**. Its batch contains ordinary reset/recovery experience, yet the update harms both independent reset evaluations and continuations from its own sampled physical states. The final actor also has a worse PPO surrogate on that batch, despite improved critic fitting loss.

An exact intra-update trace finds a particularly destructive tail. After Adam transaction313, the frozen actor completes **483/512 stochastic** and **252/256 deterministic** cases. The remaining seven ORIGINAL transactions leave only **201/512** and **143/256**. Those seven steps lose282 stochastic and109 deterministic completions, with ZERO gained completions on the same cases.

This is not an identified Adam arithmetic defect, proof that every collapse shares one cause, or a recommendation to deploy transaction313. No early stopping, rollback, momentum reset, critic correction, reward scaling or new training recipe was applied.

## Exact historical identity and seed203 exclusion

Original run37550853465, source0db6cae555cabb7bc7e2676d535ebf14365e0504. Candidate: stock SB32.9.0, one environment,2048 samples/update, minibatch64, ten epochs, gamma.999, lambda1,256-step external training truncations with normal value bootstrapping. Plant, ordinary resets, reward, noise and physical failures are unchanged.

The preserved runtime is Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/Gymnasium1.3.0. All four original archives and1312 original outer-manifest payloads verify; the complete previous four-seed numerical report reproduces byte-for-byte after JSON key normalization.

Seed204 replays from random initialization through1048576 interactions. Every original checkpoint network and evaluation value, all512 coverage rows, complete episode ledger and final policy/Adam/Python-NumPy-Torch RNG/live observations/episode state match exactly. The transparent native packet recorder matches plain training in a separate short control. Replay source is committed as7568c5ae283f93e1432cafdd23688bffbc549339.

Seed203 fails initial-tensor identity on this host. Two saved-policy endpoint re-evaluations also differ numerically in all96 records each. They remain UNQUALIFIED; no203 training replay or weight-import bypass was attempted. Numerical-environment cause is not isolated. The new causal localization concerns204 only.

Protocols: issue35/comments6028527470 (replay/localization),6028601694 (conditional panels),6028800690 (optimizer trace after sampled-state results),6028849261 (six fixed intra-update snapshots before evaluating them).

## Harmful update despite recovery coverage

The fixed every16-update grid from128 to512 selects the largest stochastic-drop block144–160; evaluating every policy inside that block selects update148, earliest ties. This is outcome-selected DEVELOPMENT localization, with all panels retained.

Eight new-from-selection confirmation domains retain the original32-deterministic/64-stochastic inference shapes and10s/15s horizons:

| Policy | Deterministic /256 | Stochastic /512 |
|---|---:|---:|
| Before148 |224|414|
| After148 |143|201|
| Lost / gained |81 /0|213 /0|

Mean duration falls9.651->8.351s deterministic and13.572->10.678s stochastic. Stochastic gamma.99 discounted return falls74.211206->73.636649.

The actual batch has8 episode starts,8 timeouts,0 true terminals. Physical angles span[-.219,.224]rad,487/2048 rows exceed|angle|.05 and217 exceed.1. Position spans[-.842,1.014]m; cart velocity[-2.179,1.992]m/s. This is not the earlier reset-free, near-upright batch. It does not establish complete state coverage, but the earlier coverage-only explanation cannot simply be assumed here.

Update148 is at303104 interactions. Later training includes recoveries; this witness is not claimed to cause every endpoint failure at update512.

## Independent continuations show harm inside fitting states

The native ABI supplies exact physical-state and noisy-observation packets. The recovered additive from-state bridge uses the same native physics; no approximate reconstruction or hidden-state policy input. New controls match4096 original/additive transitions byte-for-byte, with state-transplant/reseed/physical-ending controls. Training replay uses the original library.

Fixed64 rows16+32*k,16 independent future draws each, four common-noise branches: old throughout; new first action then old; recorded action first then old; new throughout. Old/new policy inference uses equal complete array shapes, including inactive lanes. No measurement becomes a training label.

Effects are new-minus-old, unbootstrapped stopped gamma.999 returns. Intervals are conditional pointwise approximate99% Student intervals over16 replica averages of64 fixed states, not uncertainty over independent trained policies.

| Horizon | New first action then old | New throughout |
|---|---:|---:|
|1024ticks|-.142938[-.160806,-.125069]|-96.227244[-100.393068,-92.061421]|
|4096ticks|-.392336[-.500892,-.283780]|-274.388803[-283.203768,-265.573839]|
|8192ticks|**-.397452[-.511675,-.283228]**|**-285.886521[-294.977123,-276.795919]**|

At8192 ticks, old mean return781.692754 versus new-full495.806233. Full-policy effects are negative at62/64 row means; first-action effects at50/64. Contrary rows remain included. These condition on captured state AND observation, not an exact infinite-horizon oracle or a latent-state-posterior integration given observation alone.

On the fixed64-row panel, the original raw-GAE likelihood direction for the actual parameter displacement is already **-.177119**. Independent centered stopped-return credit is **-.211854**,99%[-.299370,-.124337]; resampled centered GAE is-.009011[-.011439,-.006584]. Analytical likelihood derivatives, including log-std, pass independent finite-difference checks. These raw directions are not substitutes for normalized/clipped PPO loss. Their signs show that a wrong-sign-credit estimate is not required for this final direction to be harmful.

## Exact Adam trace and physical timing

The warm update, all320 actual transactions/permutations, and outgoing policy/next Adam state reproduce exactly with ZERO new environment transitions.

Whole-batch actor LOSS (negative clipped surrogate, lower is better): approximately0 incoming;-.003219 after epoch9; best-.003641 at transaction300; **+.015192 final**. Averaging the surrogate on all320 actual minibatch-normalization groups also worsens, approximately0->-.014990. Meanwhile critic MSE against fixed targets improves128.064880->51.756363.

81/320 actor displacements locally ascend their current minibatch loss;85 actually increase it. Many other transactions descend locally, and the total gradient-dot-displacement sum is negative. This is not a claim that every Adam step is wrong.156 steps increase the full-batch actor loss.

The exact Adam displacement was decomposed into carried first-moment and current-gradient components using its actual second moment/bias corrections. Reconstruction agrees within float32 rounding. **Five of the final seven steps locally ascend actor loss because the carried component exceeds the descending fresh-gradient component.** The actor/critic share global norm clipping. Preclip combined gradient norm changes from about1.83 at313 to21.84 at314; actor step norms are.02782/.02881. Varying critic gradients therefore change the actor-gradient scale entering Adam; causal importance of that interaction remains unisolated.

Six snapshots0/288/300/313/314/320 were fixed after the trace but BEFORE their control outcomes. Each uses all eight already exposed confirmation panels; incoming/outgoing results reproduce exactly:

| Original Adam transactions | Deterministic /256 | Stochastic /512 |
|---:|---:|---:|
|0|224|414|
|288|213|372|
|300|207|351|
|313|252|483|
|314|229|422|
|320|143|201|

The tail is physically destructive. The table ALSO contradicts a simple stopping rule: the best surrogate at300 does not have the best control. Performance declines earlier, recovers sharply by313, then collapses. These exploratory snapshots are timing localization, not policy selection, qualification or a successful nine-epoch recipe.

## Saturated critic and actual continuation discrepancies

On the fitting inputs,99.8787% of second-hidden-layer tanh activations exceed|.99| and98.4871% exceed|.9999|. These are activations across inputs/units, not exactly zero derivatives. The first layer is not similarly saturated. Incoming value mean721.4656,SD.1701.

At ALL8 actual timeouts plus the live final cutoff, the critic predicts about721.4–721.5.64 independent old-policy continuations per exact endpoint give8192-step mean returns from **593.9 to861.0**. Example predicted-minus-return errors: row241 +113.645; row1521 +127.613; row2033 -139.499. All endpoints, uncertainty and4096/8192 horizon dependence are retained.

The mean terminal-value sensitivity is.8832. Actual conditional value discrepancies are therefore measured, not inferred merely from a large bootstrap coefficient. But critic error, shared clipping and carried actor momentum are NOT independently isolated as causes, and no correction is qualified.

## Verification, failures and costs

All384 late batches/786432rows pass physical/noisy-observation/reward/ending checks, bitwise backward GAE and bitwise targets. The separate float64 physical-return decomposition has maximum absolute discrepancy.008709 against cancellation-prone float32 GAE, not claimed bitwise equality.

**32 distinct reader/numerical tests pass**:14 packet/GAE/directional controls and18 conditional/optimizer controls. They reject altered anchors/actions/values/cutoffs/keys, missing arms, nonfinite outcomes, false completions, post-terminal rewards and changed optimizer order/summaries. The independent reader checks4096 sampled and576 bootstrap continuations,4608 intra-update records and all320 optimizer transaction reductions. It does not regenerate every unexported action/random draw.

Preserved failures:203 numerical gates; and control-only early-terminal serialization, where the initial writer omitted a requested512-step stopped-return key after all branches ended at step1. The serializer was corrected before EVERY main MC measurement; identity and physical-terminal controls then pass. No main measurement was restarted or chosen for a favorable result.

This continuation used1056768 training transitions (1048576 exact historical replay+8192 recorder controls),39508098 diagnostic/evaluation transitions, and960 additional zero-environment Adam transactions across three warm replays. Original four-seed experiment costs are separate. Detailed ledger is COSTS.json in the evidence package. No sample-efficiency claim.

## Decision and evidence

Coverage-only is insufficient for this witness. The next bounded causal question is **critic/value conditioning, joint gradient clipping and carried actor momentum**, not another gamma/lambda sweep. No such corrective intervention or successful from-scratch qualification is claimed here.

Conversation attachments: `ppo-residual-update148-results-20261006.md` (full report), `ppo-residual-update148-evidence-20261006.zip` (all original archives, replay networks/batches/warm states, measurements, failures, executable sources, NumPy-only readers/tests) and detached verification receipt. Runtime installers are excluded. ZIP SHA256 **f48ccce10782a1c6dcdfb0b751f6d37b3784f4932b8bde82d412bc968bd365bc**,233433423bytes,1483 manifest-listed payloads. Use `python code/REPLAY.py .` after extraction for offline checks; exact native retraining is a distinct CPU/runtime-gated operation.

**Master, defaults, reward/noise/reset laws, protected MuJoCo/control files, deployments and production PRs remain unchanged. Issues33/35 remain unresolved.**
