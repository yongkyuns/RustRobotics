# Reset-coverage first pair: seed 41001 improves, full cohort still incomplete

24 September 2026. **Both seed-41001 native from-scratch runs completed. Extra difficult-reset trajectories improve final outward and stress-reset completion, with no lost control-success case on any final panel. This is ONE paired training seed, not the required four-seed/eight-run cohort result.**

Run36021389833, executable c634e052e19dd48b1c72627f54e4a9a14ad36cf1, protocol issue35/comment5816968405. This is the first pair returned by the fixed execution schedule, not an outcome-selected seed. No candidate, checkpoint, budget, reward, gamma, normalization or acceptance condition was changed.

## Same-budget from-scratch comparison

Both arms begin with exactly identical random actor/critic weights and train for4608 prescribed updates. They retain the existing1024 fitting rows and append512 on-policy rows from four streams. Extra-ordinary starts them ordinarily; extra-transient uses the predefined within-reset-box large-angle/large-angular-velocity same-sign subset. Each stream selects128 rows with1024-step future support unless a true terminal occurs; subsequent resets are ordinary in BOTH arms.

Both actor and critic fit1536 rows, four epochs, minibatch256,24 transactions/network/update, with continuing Adam. The original raw rows are preserved, but global normalization is recomputed. This is not a pure actor-only intervention, an oracle baseline, or equal cost to unaugmented training. No earlier fixed-policy correction is imported.

## Final short panels, each /256

| Panel | Extra ordinary | Extra transient | Gained/lost versus control |
|---|---:|---:|---:|
|Ordinary deterministic|256|256|0/0|
|Ordinary stochastic|255|255|0/0|
|Outward recovery|232|243|11/0|
|Difficult reset deterministic|237|247|10/0|
|Difficult reset stochastic|242|248|6/0|

Each episode is capped at2048 ticks,dt.01,about20.48s. Deterministic refers to policy actions, not noiseless dynamics. Every observed control-success case is retained on these final panels. The treatment still fails13 outward,9 deterministic-stress,8 stochastic-stress and1 ordinary-stochastic cases. Finite case preservation is not a safety guarantee.

Transient-minus-ordinary discounted returns, with unchanged ordinary descriptive99% paired intervals mean+/-2.586SE:

|Panel|Mean difference|Interval|
|---|---:|---|
|Ordinary deterministic|+.518469|[+.314194,+.722743]|
|Ordinary stochastic|+.533765|[+.365903,+.701627]|
|Outward recovery|+2.415044|[+1.621467,+3.208622]|
|Difficult reset deterministic|+3.471691|[+2.690059,+4.253324]|
|Difficult reset stochastic|+3.071276|[+2.399429,+3.743123]|

These are pointwise unadjusted episode contrasts between fixed policies, not confidence bounds across training seeds. The unchanged cohort analyzer rejects these two inputs with 'all eight runs are required'. No cohort screen or four-seed interval is emitted.

## Contrary results and learning progression

Ordinary UNDICOUNTED episode-total reward is slightly lower after transient training: differences-1.082771 deterministic and-1.249854 stochastic, intervals[-2.141239,-.024302] and[-1.944763,-.554945]. Discounted reward and whole-episode reward still rank differently. These small adverse effects relative to roughly2000 total reward are retained, not relabeled as improvement under every metric.

Final five-minute deterministic/stochastic counts are32/32 and31/32 in BOTH arms;60-second outward counts are31/32 in both. No success/failure cases are exchanged on these long panels. Discounted returns increase, while undiscounted totals decrease by36.892568,36.170601 and6.661065 respectively. No criterion was adjusted from these outcomes.

The treatment is worse at the retained early checkpoint1024: stress deterministic/stochastic4/64 and1/64 versus17/64 and15/64 for extra-ordinary; ordinary and outward completion are also lower then. At4096, stress counts are nearly equal:61/64 and62/64 versus62/64 and62/64. Final actors are the fixed4608 endpoints, not selected best checkpoints. The detailed report also compares the SAME first64 cases across checkpoints, never earlier64 counts directly against final256 counts.

## Actual initial-episode exposure

The previously prepared read-only exposure checker now passes end-to-end on real completed output. Initial-episode selected rows out of512:

|Detailed update|Extra ordinary|Extra transient|
|---|---:|---:|
|1|196|177|
|4097|512|512|
|4608|512|512|

At update1, transient streams contribute45+36+39+57=177 initial-episode rows; the other335 selected rows follow ordinary resets. Terminal rows belong to their ending episode. At the two later exported checkpoints all512 rows belong to initial episodes. This is intended collector behavior, not a training bug; origin does not imply every row remains physically stressed. Only these three updates export complete added streams, so these counts are not extrapolated over all4608 updates.

## Verification

Both original result ZIP outer hashes and all702 result payloads verify. The exact build outer hash and122 members verify; both runs carry that build manifest and source commit. The prior checker ZIP's20 members also verify. The registered analyzer body exactly matches the verified native build source, and the exposure checker is unchanged.

Both published single-run summaries reproduce exactly. Verified4672 evaluation records and exact random-domain keys, all9216 update rows/costs, identical initial actor/critic bytes, identical first original union and all initial evaluation fields except arm names. On six detailed updates the original raw prefixes,1536-row normalization,24 added streams/27648 source rows of float32 GAE/returns, reset/terminal semantics, selected support and observation continuity verify. All144 detailed minibatch records have complete four-epoch permutations and288 saved actor/critic checkpoints. Final actor/critic bytes equal each final24th optimizer checkpoint. Unexported moments and all unexported fitting batches are NOT independently reconstructed.

All16 retained final rep0 traces exactly reconstruct recorded total/discounted returns, boundaries, state/observation continuity, maxima, applied-force RMS and centering. This is not an independent dynamics/noise replay or reconstruction of unexported full episode traces.

**14 new actual-data/mutation tests and15 unchanged exposure tests pass**, including a fresh self-contained package directory. New controls cover casewise accounting, independent statistics, actual post-failure resets, altered outcomes/traces/keys, corruption, costs and incomplete-cohort rejection. An initial test-helper unclosed-file warning was repaired and rerun without changing results. Complete repeated verification, including the fresh package, reproduces pair.json byte-for-byte, SHA25600654755944b082a133cdeec13535e200407da97cb454398196451e33ddb91c8.

## Cost, source and disposition

Ordinary/transient training transitions are81410282/80975612, including primary, original supplemental, added and tail collection. Each arm includes21233664 added transitions and110592 Adam transactions/network. Evaluation uses5579185/5538610 transitions. Matched added/fitting budgets are not identical total realized simulation cost. This continuation performs offline verification only; no new simulation, optimizer step or workflow was executed locally.

Original artifacts:
- Ordinary10819007711: fb643f479ccc69e75b1fc2de7c578d0bcbb14402c44d6c5f91653803ee0efdc3.
- Transient10819547603: 79f79d2e09685dbfd9ff07fb752919b06862011bb51c46efb2c776037f99f5a3.
- Build10817355765: d76b73cb79b9127fc9bb48c3fe654a9baf157b62ccacd1345e052becbfb3caf0.
- Prior exposure-check package:5743ca332a6e4acc1c3f76f4efcd1dd2e110e1cdf4dfa3fdf4136d44472541ff.

The first pair is consistent with the intended benefit, but does not establish it across the remaining three seeds or authorize production adoption. At the recorded final status check, both41002 arms were training and the four41004/41006 arms were queued. Only this report is added outside workflow trigger paths. Experiment code, active run, master, production defaults and deployed policies remain unchanged. Complete original ZIPs, standard-library verifier/tests, reports and logs are retained in the conversation evidence package.
