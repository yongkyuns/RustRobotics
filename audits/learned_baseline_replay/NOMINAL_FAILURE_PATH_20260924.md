# Nominal failure paths: a fragile early transient survives average-KL matching

24 September 2026. Completed targeted independent-simulator diagnostic; no new training algorithm, native run, or population qualification. Protocol: issue35/comment5815632990. Prior report f8b4e481 and exact movement-matched bundle SHA25650303c7710a5f00c2f177462bdbe80438bd137a3d4ec6674a4e71066da3f8376. All existing actors and task definitions are unchanged.

## Exhaustive selection and exact controls

Select ALL nominal cases in both existing movement-matched domains where incoming succeeds and any corrected actor fails. This yields one exposed nominal-deterministic case:41004/update4195, replication324, root2026092442. No such case exists in the other three nominal panels in these particular domains. This is outcome-selected diagnostic evidence, not a population safety statement.

All12288 original nominal evaluation records were replayed through both the UNCHANGED evaluator and the instrumented evaluator. Endings, lengths, total rewards and discounted returns match the original exactly. Original512-wide arithmetic is retained for those controls. The8-wide policy-switch endpoint state/action paths also match their original512-wide paths exactly.

| Actor on case324 | Outcome | Duration | Discounted return |
|---|---|---:|---:|
|Incoming|Completes|20.48s|8.415923|
|Original update|Position failure|2.56s|-6.572327|
|Full8 baseline correction|Position failure|2.58s|-4.222097|
|Paired8 direct credit|Angle failure|1.77s|-10.485565|
|Full8 matched|Position failure|2.62s|-2.495538|
|Paired8 matched|Angle failure|2.71s|-1.347770|

Uncontracted paired8 keeps the aggregate510/512 nominal count by gaining replication131 while losing324. Its previous aggregate-screen pass did not imply preservation of incoming success cases. Previous screen decisions are unchanged.

The initial physical state is x+.172686m,v-.041756m/s,theta+.225763rad,omega+.482645rad/s: a legitimate nominal reset, not an outward-start evaluation state. Incoming commands-19.728338N initially; matched full8/paired8 command-19.729126/-19.723967N. By.5s, incoming reaches x-1.521263m,v-4.226638m/s; both matched trajectories reach approximately-4.2646m/s. Small early differences precede larger later outcome differences; no unique scalar causal state variable is identified.

## Fixed prefix/suffix interventions

Execute all32 schedules: two matched candidates, both switch directions, cuts[0,25,50,100,200,400,800,2048] ticks, using the original case's time-indexed noise. True termination is absorbing. No schedule is adopted as a controller.

| Schedule | Full8 matched | Paired8 matched |
|---|---|---|
|Candidate first.25s, then incoming|Completes|Completes|
|Candidate first.50s, then incoming|Position failure2.57s|Position failure2.63s|
|Incoming first.25s, then candidate|Position failure2.62s|Angle failure2.71s|
|Incoming first.50s, then candidate|Completes|Completes|
|Incoming first1s, then candidate|Completes|Completes|

All later incoming-prefix cuts complete. Later switchbacks on candidate trajectories do not rescue this case. This finite grid isolates a consequential early transient, not an exact threshold or proof of physical irrecoverability.

## Fresh future-noise comparisons

Probe ticks0,100,400 along incoming and each failed matched trajectory while alive. Compare incoming versus candidate FIRST command, then use the SAME continuation policy for both branches, separately incoming and candidate.64 paired fresh-noise draws per comparison, root2026092451, explicit case/source/tick/draw identities. Remaining horizon2048-t, first-terminal stopping, reward-only labels, NO critic bootstrap. All selected cases are deterministic-policy cases; these are NOT estimates of the stochastic PPO training gradient. Environmental/observation noise remains.

Twenty completed first-action comparisons produce2560 episode records; two400-tick candidate-source probes are skipped after actual termination. Some contexts/controls are shared, not2560 independent cases. The same outputs give whole-policy comparisons by choosing each policy's own first action AND continuation:

| Fixed state/context | Incoming completed | Matched candidate completed |
|---|---:|---:|
|Initial, shared incoming-source draws, full8 comparison|23/64|0/64|
|Initial, same shared draws, paired8 comparison|23/64|0/64|
|Initial, separate full8-source draws|18/64|0/64|
|Initial, separate paired8-source draws|20/64|0/64|
|Incoming trajectory at1s, either comparison|64/64|64/64|
|Candidate trajectory at1s, either comparison|0/64|0/64|
|Incoming trajectory at4s, either comparison|64/64|64/64|

The shared initial context is the same64 noise draws across both matched comparisons and must not be counted twice. Incoming itself is fragile at this selected initial state. Failure from candidate states under the two tested policies is not proof that another controller could not recover.

Whole-policy discounted-return differences at initial shared context: full8-matched minus incoming-6.587580, descriptive99%[-7.368763,-5.806396]; paired8-matched-5.442922,[-6.240366,-4.645477]. At incoming's1s state, both matched policies instead improve return and all policies complete. These are conditional-state findings, not global rankings.

One-action contrasts at initial state, always followed by INCOMING policy:
- Full8-matched first action: mean discounted improvement+.001678,99%[+.001099,+.002257].
- Paired8-matched first action: mean loss-.009519,[-.011879,-.007159].

The corresponding.995-discount contrasts have the same signs,+.012513 and-.070311. Thus full8-matched's locally beneficial first command does not make its entire feedback policy beneficial. Both candidates' one-action contrasts are negative at the incoming1s state and positive at4s. All remaining combinations are retained. Ordinary pointwise mean±2.586SE intervals are not multiplicity-adjusted, trained-seed bounds or formal action-quality guarantees. No inference that the policy-gradient theorem fails or that exact stochastic training credit was measured here.

## Actor-batch coverage

Compare against the original1024 actor-fitting observations at4195, NOT every historical training sample. Standardize observation coordinates by fitting population SD and use nearest Euclidean distance. Also measure incoming-actor hidden-feature departures outside fitting ranges, scaled by feature SD (<=1e-8 uses1). No threshold is fitted.

Initial observation: nearest-neighbor distance2.2544 standardized units. Although every coordinate is inside its marginal fitting range,28/64 final hidden features lie outside fitting feature ranges; maximum departure8.7065 fitting SD. This is a joint-combination/feature-coverage diagnostic, not a calibrated out-of-distribution probability.

At.5s incoming has velocity-4.2266m/s and angular velocity-1.3728rad/s, beyond that batch's observed minima-2.0087m/s and-.3454rad/s; nearest distance5.4726. Both matched trajectories similarly leave this batch's coverage. No claim is made that these states were absent from ALL previous training rollouts or that additional coverage alone necessarily fixes the learner.

On identical incoming observations over the first50 ticks, matched full8/paired8 mean absolute command changes are.17022/.11953N, maxima.43991/.34233N. Tiny initial-command differences and matched batch-average KL do not determine accumulated feedback effects.

## Verification and work

All537 original package payload hashes verified on ingestion. The new compact package retains21 exact required members with complete original manifest/origin receipt, not the whole old ZIP. Actor hashes checked on every load. All30 retained control paths plus32 switch paths reconstruct rewards, both discounts, first boundaries and state/observation continuity:62 paths. All20 probe comparisons and case swaps reconstruct from raw records.

23 actual-data/mutation tests pass with warnings as errors. They cover source corruption, exact instrumentation, terminal absorption, equal-policy switching, single-first-action identity, shape/finite checks, noise separation, paired statistics, independent nearest-neighbor calculation, feature inference, skipped terminal probes and corrupted traces. All32 switches and2560 probe records repeat exactly,28 output files byte-for-byte. Repeated analysis reproduces report.json exactly. Full nominal panels were run once unchanged and once instrumented; a further full nominal repeat is not claimed.

Main active transitions:50011326 for original+instrumented nominal controls,37000 switches,2538929 first-action comparisons,52587255 total. Switch/probe repeat adds2575929. Tests perform additional toy/control simulations outside those main totals; vector work/inference/hashing adds computation. No new actor/critic optimizer steps, learned baselines, training histories, native workflow or production change. A streaming-session tool request failed before executing; synchronous execution succeeded. No numerical threshold or actor changed.

## Interpretation and next boundary

The earlier baseline-causality results remain valid. This identifies an additional failure mode: a corrected actor update can alter a fragile nominal-reset transient outside its fitting batch's well-represented states, even after average movement matching. One-action credit and whole-feedback behavior must not be conflated.

The practical next comparison needs independently fixed reset-transient/high-speed-reversal coverage and casewise preservation checks, not merely an outward-state mask or scalar mean KL. This exposed case and successful switch cut must not be treated as held-out validation or a tuned runtime gate. No such learner is claimed implemented. Only this report is added outside native workflow triggers; production/defaults/master/deployment and all actor weights remain unchanged. Full source/tests/retained evidence are in rustrobotics-nominal-failure-path-20260924.zip.