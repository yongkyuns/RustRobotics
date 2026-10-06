# PPO replay localization — October 6, 2026

## Result and scope

The saved stock-SB3 policies show two distinguishable changes: loss of recovery behavior despite command-force headroom, and loss of a previously acquired locally stabilizing actor in seed 203. This identifies concrete controller changes, not yet the PPO update or credit-estimation error that produced them. No learning update, production recipe change, production merge or deployment was performed.

**The full historical exact-replay gate FAILED.** Seeds 201, 202 and 204 match their original outcomes exactly at all four checkpoints. Seed 203 matches all episode lengths/endings but not every numerical value. Its traces are retained as `UNQUALIFIED` and excluded from the accepted physical-trace analysis. The independent local/noiseless calculation on saved weights is a separate diagnostic.

Protocol: [issue #35/comment6020510359](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6020510359). Original experiment: run36129074098, source`da6623ae4b5878ae5079bf05c18c3d327570bc16`, production base`95c670b9f4618a11dc5439b026caf385a39718c8`. Replay retains all stock-SB3 seeds201–204 and checkpoints0/65536/262144/1048576, each with32 deterministic10s and64 stochastic15s cases. These are exposed development cases, not new held-out learning trials.

## 1. Lost recovery with substantial command-force headroom

Accepted exact traces cover seeds201/202/204: nine nonzero-checkpoint panels and864 complete trajectories. Comparing the same cases at65536 interactions and the endpoint gives101 early-success/final-failure cases, with no gained completions. Seed203's other63 lost cases are not included in these trace statistics.

|Training seed|Lost deterministic|Lost stochastic|
|---|---:|---:|
|201|10|16|
|202|18|28|
|204|13|16|
|Total|41|60|

Of the101 failures,49 end at the cart-position boundary and52 at the pole-angle boundary. Failure duration ranges0.67–6.35s, median**1.59s**; **72/101 fail within2s**.

**None approaches the command-force limit.** The largest absolute command anywhere in these complete failing trajectories is**12.540979N**, versus a20N limit. None reaches95% of that limit. These are commanded forces, not realized forces after actuation noise/disturbances: the archived ABI's force field is pre-noise.

Both saved mean actors were also evaluated on the exact same noisy observations along each final-policy failure. This read-only float64 reconstruction of saved float32 weights consumes no simulator/RNG state and replaces no actions. The early actor's per-case mean absolute command is larger in**all101 cases**. The median of the per-case means is**7.184875N early versus3.824299N final**. The late deterministic reconstruction agrees with recorded deterministic commands within the diagnostic0.0002N check; bitwise Torch equivalence is not claimed. Stochastic cases compare mean actors, not individual action draws or learned standard deviations.

Interpretation: policies lost recoveries and command smaller forces at the same observations while command range remains available. This excludes exhausting the command limit as the direct explanation for these cases. It does not prove that indiscriminately increasing force, changing one particular action, or applying the early actor from an already-diverged state would save an episode. Action signs, timing and coupled state feedback still matter. These are101 conditional evaluation cases from3 training seeds, not101 independent training trials.

## 2. A saved actor loses local stabilizing feedback

A separate analysis uses all16 saved actor weights, including seed203, without relying on its unqualified historical traces. It finds sign-bracketed zero-force upright equilibria `[x,0,0,0]`, computes the actor Jacobian, and linearizes the noiseless sampled-data nonlinear plant with force held during each RK4 step.

For seed203, the equilibrium nearest the origin changes as follows:

|Interactions|Equilibrium x|Position component of force Jacobian|Closed-loop spectral radius|
|---|---:|---:|---:|
|65536|−0.1414m|+4.601N/m|**0.9932858**|
|262144|−0.0485m|−0.498N/m|**1.0019430**|
|1048576|+0.1798m|−0.453N/m|**1.0029773**|

A spectral radius below1 indicates local asymptotic stability for this smooth real-valued map; above1 indicates instability. The early saved actor stabilizes its near-origin upright equilibrium, while the intermediate and final actors destabilize theirs. These are each policy's own equilibria, not all evaluated incorrectly at a non-equilibrium origin.

The force Jacobian `[x,velocity,angle,angular velocity]` changes approximately from`[4.601,6.092,−79.541,−26.727]` to`[−0.453,0.878,−36.488,−13.295]`. All components change; the position-gain sign change is not an isolated causal intervention. A stabilizing local property has been lost in the saved actor itself, independently of sampled exploration noise. This is not yet an explanation of why PPO learned that change.

### Contrary evidence and limitations

The other three final actors still have at least one locally stable upright equilibrium, at approximately−0.567m(seed201),−0.727m(seed202), and+0.565m(seed204). They can fail finite/noisy recovery without sharing seed203's final local-instability result. Seed202 is locally unstable at its intermediate checkpoint and stable again at its final checkpoint. No universal or monotonic collapse mechanism is claimed.

The calculation is float64 evaluation of exact saved float32 weights, local and noiseless. It is not a global region-of-attraction, noisy robustness, hardware safety or exact finite-precision certification. Root search scans[−2.4,2.4] with0.005m spacing, retaining sign brackets and exact grid roots; it could miss a tangent root.

The correct discrete feedback Jacobian is`Ad + Bd K`, using the fourth-order RK4 linearized state/input maps and a command held across substeps. It does not recompute feedback inside RK4 substeps. Analytic actor derivatives are independently checked by central differences; a separate finite difference of the held-command nonlinear RK4 map agrees with maximum discrepancy**2.33e-9**, below the fixed1e-6 diagnostic tolerance. The saved source is`audits/clean_reference/local_linearization.py` at`fc71157433e51ae433aee12795a44737fc4dc0fd`.

## Replay acceptance and failures

The original native binary and original Python evaluator are reused unchanged, including batch shapes/inactive rows, case keys, action randomness, clipping, reward accumulation and episode boundaries. A transparent ABI wrapper records trajectories without feeding privileged state into the policy or consuming randomness. The pinned run uses Python3.11.16, Torch2.8.0+cpu, NumPy2.2.6, SB3 2.9.0 and Gymnasium1.3.0; the checked SB3 source modules match archived source bytes. Historical CPU and Python patch-level identity are not established.

|Seed|Exact historical checkpoint panels|Lengths and endings|
|---|---:|---|
|201|4/4|All match|
|202|4/4|All match|
|203|0/4|All match; numerical values differ|
|204|4/4|All match|

Seed203 has11 differing records at checkpoint0 and96 at each nonzero checkpoint. Its maximum absolute undiscounted-return discrepancy is0.003867, discounted-return discrepancy0.000283, and state-component discrepancy0.000199. These quantify the failed check; they are not replacement tolerances. The discrepancy's cause remains unresolved. No CPU path, library bug or PPO arithmetic defect is established by it.

All1536 recorded cases match their old lengths/endings, and instrumented execution matches the current unchanged evaluator exactly throughout. Nevertheless, the historical all-values-exact criterion remains false, so seed203's traces are unqualified and excluded from the101-case physical analysis. Its saved-weight Jacobian result does not depend on accepting those traces.

The initial local inference-only reconstruction used newer Python/Torch/NumPy and failed exact matching; it is not accepted historical evidence. Obtaining the older local environment failed at DNS resolution. Execution moved to Actions. The first pinned run stopped at seed203/0 after exactly reproducing201/202; restoring the original seeded SB3 model-construction path in the second run did not fix that discrepancy. The final runner retained every panel and then exited with failure on the unchanged exact gate, rather than stopping before later panels or hiding the mismatch.

## Reproducibility receipts

|Run|Executed source|Result|Artifact|
|---|---|---|---:|
|37494693446|`7fc2d2c0e2153b768ce01640ce37f2846d4c25c3`|Stopped at seed203/0|11427625754|
|37495282668|`92d93b64012226a93883e11b9bc9dcf2cf260f28`|Same stop with original setup|11426818144|
|37495693725|`ed18b85c08230c7518e442d043b823b571b726a0`|Complete ledger; historical gate failed|11427133521|

All three replay ZIP hashes and their260/262/275 manifest-listed payloads verify. The first run's576 stored trajectories equal the final ledger's corresponding arrays. The full ledger performs**2,384,860 simulation steps, zero learning updates**; earlier failed attempts add replay work outside that total. Six trace-corruption controls pass on the pinned runner. The local read-only suite has**19 passing tests**. A fresh archive extraction reproduces both analysis JSON files byte-for-byte and passes all19 tests; these are not additional training trials.

An initial local analyzer used Python3.13's sum rather than the original left-to-right floating accumulation; operation order was corrected without relaxing equality. Root scanning was corrected to include exact grid roots and regression-tested. No training equations, seeds, weights or learning thresholds changed.

The conversation evidence package includes all final-run archive contents and its unchanged manifest, five original comparison ZIPs, full trajectories including UNQUALIFIED ones, policy weights, runnable analysis/tests, original and instrumented evaluator sources, mismatch ledgers, detailed REPORT.md, and selected failed-attempt logs/source/receipts. ARTIFACTS.json records all original replay ZIP IDs, sizes and hashes; the two earlier full replay ZIPs are not duplicated inside. Artifact retention is finite; do not assume another chat has the same sandbox path.

## Next bounded learning question

Target an observable property loss: determine when seed203's stabilizing feedback disappears between65536 and262144 interactions, then distinguish wrong action credit from actor changes that damage recovery away from fitting states. Retain contrary improving updates and the other seeds' different outcomes. Attribution requires a continuing or reproducibly replayed learner with actor, critic, Adam, environment and RNG state intact; weights alone are not exact-resume checkpoints. Historical replay limitations must remain explicit.

No causal update audit or new training comparison was launched in this continuation. No reliable production learning fix or held-out sustained-balancing acceptance is claimed. These findings do not justify adding a curriculum, critic-repair phase, rollback, or another broad tuning sweep. Changes remain confined to diagnostic source, its isolated replay workflow and this report on the audit branch. Master, production PR heads and protected MuJoCo/control files were not modified.
