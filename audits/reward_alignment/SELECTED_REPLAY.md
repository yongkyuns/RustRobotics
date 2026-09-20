# Selected reward-survival disagreement: exact replay completed

September20,2026. Issue #35. This completes the selected-case follow-up to the retained-data reward audit, not a new learned controller or robustness qualification.

## Finding

**The reflected controller survives300seconds but incurs substantially larger early recovery costs than the original1.2second rail failure. At gamma0.99 those costs outweigh its discounted survival benefits.** Re-scoring the unchanged trajectories at the predeclared gamma0.995 reverses this particular ordering. The symmetric survivor already scores above the failure at gamma0.99.

This is one retrospectively selected initialization/random sequence, not an expected-value comparison or evidence that learning at gamma0.995 succeeds. The earlier complete audit remains unchanged: all15 primary original-versus-symmetric comparisons with different survival outcomes favor the survivor under the current score. Neither a blanket discount explanation nor a production gamma change follows.

## Exact replay

The queued run **35508137816** completed successfully on its first attempt at2026-09-20 11:59:48UTC. Executed source **114fe450771f4fb9d2e3853d4a2c5757abf94891**, branch `audit/ppo-reward-alignment-20260920`. Analysis registration5749512184 and retrospective replay addendum5749522580 preceded their respective new outcomes.

Seed41003, long-stoch, episode45, key6219739887089942572. All19 fields of each of the three outcome records reproduce the prior immutable reflection archive exactly. Original mapping is mu(o), reflected is -mu(-o), symmetric/odd is[mu(o)-mu(-o)]/2; the existing Gaussian innovation and single20*tanh transform follow. All weights remain frozen and the corrected nonlinear RK4 plant is unchanged. Initial states/noisy observations match, and exported policy innovations match through every shared tick, including all30000 ticks between survivors.

| Mapping | Ending | Duration | Gamma0.99 return | Undiscounted return | Maximum cart speed |
|---|---|---:|---:|---:|---:|
|Original|Rail failure|1.20s|27.464364|31.662562|3.019996m/s|
|Reflected|Survives, finishes centred|300s|15.554230|29402.219491|4.849191m/s|
|Symmetric|Survives, finishes centred|300s|38.657315|29681.381316|3.455416m/s|

The initial physical state is approximately[-.004868m,-.384609m/s,+.213695rad,+.328532rad/s]. Time labels use nominal100Hz; actual dt is float32(.01). New work is60120 simulator transitions, zero policy-training transitions. Generic preflight/regression training is additional.

## Recovered reward accounting

The unchanged reward uses post-transition state and clipped commanded force:

`1 - .2*x*x - .02*v*v - theta*theta - .05*omega*omega - .001*u*u`.

A true terminal replaces that step by-10; a timeout is not a physical failure. Applied disturbances are not substituted for the commanded-force cost. The original sequentialfloat32 reward implementation reproduces BIT-FOR-BIT on all60120 exported steps.

All contributions below are discounted at the configured float32(.99):

| Contribution | Original failure | Reflected survivor | Symmetric survivor |
|---|---:|---:|---:|
|Positive nonterminal bonuses|+69.759588|+100.000095|+100.000095|
|Position cost|−24.579223|−40.388030|−41.128939|
|Velocity cost|−6.986644|−16.936483|−8.602732|
|Pole-angle cost|−1.271960|−6.431321|−2.686188|
|Angular-velocity cost|−0.679260|−3.405263|−1.141592|
|Commanded-force cost|−5.754089|−17.284769|−7.783329|
|Terminal penalty|−3.024048|0|0|
|Total, including retained float32 rounding residual|27.464364|15.554230|38.657315|

Relative to the failure, the reflected survivor gains30.240507 in positive survival bonuses and avoids3.024048 of discounted terminal penalty. But running costs increase by45.174689, yielding net−11.910134. This is actual return accounting, not an inaccurate critic prediction or a claim the controller intentionally chooses failure.

Between0.50and1.00seconds all50 reflected-policy rewards are negative, contributing−10.564880 discounted reward. At1.00s its cumulative score is0.378977, versus31.289376 for the original. At1.20s the original has terminated with27.464364; the reflected survivor has1.019773 so far. Its remaining298.8seconds contribute14.534457, leaving15.554230 total. After5seconds the remaining295seconds contribute only0.633588; after10seconds the remaining290 contribute0.004285.

The symmetric survivor takes a less costly transient than the reflected survivor. Its velocity/force costs are8.602732/7.783329 instead of16.936483/17.284769. It both survives and scores higher under the existing gamma. A survival/reward trade-off is therefore not unavoidable for this initialization.

## Fixed-discount sensitivity, not new training

| Offline scoring discount | Original failure | Reflected survivor | Symmetric survivor |
|---|---:|---:|---:|
|0.99|27.464364|15.554230|38.657315|
|0.995|30.040024|52.176858|101.286393|
|0.999|31.469616|703.900898|842.846552|
|1|31.662562|29402.219491|29681.381316|

These four values were specified before the replay. Discounts are decoded to float32 then accumulated in float64. At0.995 the reflected survivor leads by22.136834, whereas it trails by11.910134 at0.99. No path changes, gamma optimization, new noise replications, or training at alternative discounts occurred. Higher gamma changes reward scale and estimation properties; this single-case ranking does not qualify a learning change. Neither survivor is shown optimal, and no expected-return or hardware claim follows.

## Verification and limitations

CI passes74 native unit/audit tests,7 ordinary balancing controls,9 ordinary learning/evaluator controls, strict Clippy, formatting and protected-source restoration. Two audit endpoints are ignored during the regular unit pass; the selected endpoint is then explicitly executed. Two historical heavy integration endpoints and the full production browser/platform matrix were not run.

The new artifact10604534387 is5455706bytes, SHA256 **80e7f877e64bd4351a8452a02d7b73f66ed1f3ac13cab76e0298275d6955ce0d**. Parent10597837264 is bound to SHA256 **b915c76d4fe28be4dddcc686214a37e1901bba2c47ee2511195c0d2774883039**, reflection run35486221478. Both unchanged ZIPs are included in the incremental evidence.

Offline verification checks77 current and19 parent payload hashes, all576 parent identities, all three exact selected records, all60120 recovered transitions, actor/action reconstruction, reward/end semantics, initial observations, paired innovations, force RMS and centring. Independent maxima: actor4.775e-7, force1.640e-6, dynamics2.219e-7, aggregate2.843e-13, inside unchanged inherited tolerances. Rewards separately match bit-for-bit. Unexported RNG draws are not independently regenerated.

**64 offline tests pass** (51 inherited+13 selection/pairing/interval checks). The new time accounting was additionally checked on three already-recorded parent replication0 traces,90000 archived steps; these are compatibility checks, not additional simulator experience or episode45. No measurement failed or was repeated; no data, replay assertion, numerical tolerance, reward or decision rule was relaxed.

The evidence includes both raw archives, complete executed source/logs in the native artifact, independent scripts/tests, six numerical output files, cumulative plot and offline replay instructions. Python3.13.5/NumPy2.3.5/SciPy1.17.0 verification runs no native code, network, simulator or training. The prior full4608-outcome study remains separately archived; this increment does not pretend to repeat it.

## Disposition

This closes the missing time-local explanation of the sole secondary reward/survival reversal. It follows the configured discounted costs and terminal reward, not a new return-calculation or termination bug. The15 primary comparisons favoring successful recovery still require the learner to discover and retain that behavior.

No production gamma, reward, normalization, trained weights, PR38, master, deployment or fallback was changed or merged. Do not adopt a new discount from this retrospective example alone.