# Reset-transient generalization beyond the exposed failure case

24 September 2026. **The weakness recurs across independently sampled reset states and stochastic policy actions. Baseline corrections largely avoid update4195's additional stress-region deterioration, but do not preserve every incoming success or fix the incoming policies' existing fragility.** This is a completed fixed-policy external-simulator study, not native policy qualification or a new training algorithm.

Protocol issue35/comment5816231669 preceded all new initial states/outcomes. Prior report679c2233; source package SHA2568844dc4f15dc62477cab95b9a67f8556cf15bbdc624d8ffd1a315f33bbe647f2. Exact incoming/original/full8-matched/paired8-matched actors are retained at41006/update4140 and41004/update4195. Matched endpoints remain post-update contractions, not new lower-LR PPO updates or validated continuing Adam states. No actor weights were changed.

## Design and statistical unit

Ordinary physical starts are uniform in x±0.2m,v±0.4m/s,theta±0.25rad,omega±0.5rad/s. The predeclared stress group keeps x/v unchanged and conditions on |theta| in[0.125,0.25], |omega| in[0.25,0.5], matching angular signs with uniformly sampled sign. This subset occupies1/8 of the ideal ordinary reset law, not a broader physical reset box. The definition was informed by the earlier failure diagnosis, so this is not untouched final qualification.

256 independent physical starts/group,8 independent initial-observation/future-noise draws each. Both deterministic and stochastic policy modes, shared physical states/environment noise across actors, modes and witnesses, separate Gaussian action innovations. Root2026092461; physical key[root,stratum,state_id,0]; environment/action keys[root,stratum,state_id,draw,1 or2]. The old exposed case is not reused. Horizon2048 ticks,dt.01, inherited float32 task/policy/noise/terminal arithmetic. All65536 new episode records retained, with total/.99/.995 returns.

There are512 distinct physical starts TOTAL, not65536 independent starts or training seeds. Descriptive99% intervals first average8 draws WITHIN a state, then use mean±2.586SE across256 starts. Intervals are unadjusted conditional fixed-policy approximations, not trained-seed reliability bounds or formal noninferiority. A zero empirical interval is not proof of equivalence. Stress is deliberate oversampling and is NOT pooled with ordinary resets as an ordinary reset success rate.

## Ordinary-reset completion, each /2048

|Witness/mode|Incoming|Original|Baseline-only matched|Paired-credit matched|
|---|---:|---:|---:|---:|
|4140 deterministic|2016|2016|2016|2016|
|4140 stochastic|2016|2016|2016|2016|
|4195 deterministic|2023|2008|2016|2016|
|4195 stochastic|2021|2007|2018|2018|

At4195 each matched correction loses7 deterministic successes with no gains, all at ONE new physical state185. Incoming completes7/8 draws there; each correction0/8. State x+.074255,v-.320434,theta+.225545,omega+.380616 is distinct from old rep324. Under stochastic actions each correction gains1/loses4, net-3, across one better and two worse physical starts. This is not only deterministic-action or old-noise fragility.

Clustered deterministic completion-rate change is-.003418,99%[-.012257,+.005421]; stochastic-.001465,[-.005654,+.002725]. These few adverse starts do not precisely establish population harm. Original4195 discounted-return changes are-.590302[-.851351,-.329253] and-.594012[-.844653,-.343371]. Both corrections' ordinary discounted-return intervals include zero at4195. No state/policy is selected from those intervals.

## Stress-reset completion, each /2048

|Witness/mode|Incoming|Original|Baseline-only matched|Paired-credit matched|
|---|---:|---:|---:|---:|
|4140 deterministic|1858|1872|1853|1856|
|4140 stochastic|1856|1866|1851|1853|
|4195 deterministic|1944|1865|1942|1944|
|4195 stochastic|1933|1857|1930|1938|

Original4195 loses79 deterministic successes across11 physical starts and76 stochastic successes across21 starts, no gains. Clustered completion-rate differences are-3.86 percentage points,99%[-6.92,-0.79], and-3.71 points,[-6.08,-1.35]. Discounted-return losses are-3.06065[-3.61912,-2.50219] and-3.10799[-3.64589,-2.57009]. The deterioration thus reproduces over new stress starts, not only the old selected episode.

Matched baseline-only reduces those completion losses to2/3 net. Matched paired credit preserves the incoming deterministic success set here and has5 net additional stochastic successes, but the latter comprises9 gained/4 lost across3 better/4 worse starts. Its clustered completion-rate interval includes zero. Its stress discounted-return changes versus incoming are+.430073[+.206353,+.653794] deterministic and+.424398[+.181860,+.666936] stochastic. These do not erase its ordinary losses.

Contrary4140 finding: the ORIGINAL update increases stress completions by14/10, while matched corrections lose5/5 and2/3 net. Paired stochastic accounting is1 gained/4 lost. All those4140 completion-rate intervals include zero. Earlier original outward-recovery harm and corrected outward benefits remain valid; this new experiment is reset-only, not a fresh outward-recovery evaluation. Region-specific gains are not global improvement.

## Concentration, pre-existing fragility and coverage

35/256 ordinary starts (280/2048 episodes) lie in the predeclared stress subset. EVERY observed ordinary-reset failure across all tested actors/modes/witnesses starts in that subset. This is finite-sample concentration, not impossibility of failure elsewhere.

Incoming policies already fail190/192 stress episodes at4140 and104/115 at4195 (deterministic/stochastic), touching24/27 and13/18 physical starts. All4140 stress failures occur within1s. At4195,96/104 deterministic and98/115 stochastic incoming stress failures occur within1s; all finish by5s. Baseline correction has not repaired these pre-existing difficult transients.

On4195 incoming stress trajectories, first-second peak speed averages5.36m/s among failures versus4.00m/s among successes. Its actor-fitting velocity range was[-2.0087,+3.4394]m/s. Success and failure speeds overlap; no velocity cutoff or physical irrecoverability claim follows.

Coverage refers ONLY to the original1024 actor-fitting observations, not all historical training. At0.5s on4195 ordinary deterministic incoming trajectories, mean fitting-standardized nearest distance is.698 among successes versus5.128 among the7 cases lost by either correction. Initial actor-feature departure on those lost cases averages8.77 fitting SD. Outcome subgroups are descriptive, not used to train a guard.

A simple range flag is much too broad:1296/2023 successful4195 ordinary deterministic trajectories leave at least one fitting-coordinate range during the first100 observations. Among successful deterministic stress trajectories it is1935/1944 (99.5%). Distances/features are not calibrated failure probabilities and ranges overlap.

On identical incoming stress observations at0.5s,4195 baseline-only/paired matched mean KL is.005515/.014881 despite equal old-fitting-batch mean KL.004772. Corresponding4140 stress values are.009774/.017916. These scores cause no new contraction, threshold or policy selection. Matching old-batch movement does not control movement throughout the later transient.

## Verification and preservation

All108 prior package payloads verified before retaining13 exact input members:8 actors,2 fitting batches, actor identities, prior outcomes and unchanged evaluator. Complete prior manifest and origin map retained; not a fresh all-historical-archive audit.

Before new outcomes,4096 OLD deterministic control records (2 witnesses×4 actors×512) matched original endings/lengths/returns EXACTLY through both unchanged and observer-instrumented evaluators. Every new run checks immutable random-array hashes. Fixed full traces use draw0 of state IDs0 and255, not selected examples. First-terminal stopping is absorbing.

**21 actual-data/mutation tests pass**, also from a fresh package extraction. They verify source/actor hashes, all65536 case identities,256×8 clustering, sampling bounds, random/mode/block separation, readonly simulation, terminal absorption, corruption/nonfinite/false-timeout rejection, independent scalar nearest-neighbor/inference checks, and all64 retained full traces' rewards/both discounts/boundaries/continuity. Both stochastic stress panels repeat ALL16384 actor-episode records exactly (32 result NPZ files plus manifests/initial states). Other6 witness/group/mode panels are NOT fully repeated. Independent analysis repeat matches11 files byte-for-byte; fresh-package analysis also matches the original report. Repeats are not new statistical samples.

Main simulation128236288 active transitions; unchanged/instrumented old controls16687500; required stochastic-stress repeat31011386. Tests add small control simulations separately. Inference/distance/range calculations and inactive vector work add cost. Zero new optimizer transactions, critic fits, MC estimates, native CI jobs or continuing-learning histories. Inherited actor/MC costs remain.

A streaming-exec request failed before a command ran; synchronous calls succeeded. Initial summary-script syntax error was fixed before that analysis ran; its log is retained. No actor, simulation output or statistical criterion changed.

The result supports a bounded ACTOR-fitting reset-transient/high-speed-reversal coverage experiment, with ordinary/outward coverage preserved, casewise comparisons and from-scratch controls. It does NOT demonstrate that such extra data alone fixes the learner, validate a range/switch/KL guard, or qualify the oracle-corrected actors for deployment. That new learner experiment was not performed here.

Full report, scripts/tests, exact inputs, all outcomes/snapshots/64 traces, random hashes, repeats and logs are retained in `rustrobotics-reset-transient-generalization-20260924.zip`. Only this evidence report is added outside existing workflow triggers; production/defaults/master/deployment and all actor weights remain unchanged.
