# Reset-coverage cohort and evaluation-only failure replay

24 September 2026. **The full four-seed/eight-run coverage experiment fails its registered component screen. Do not adopt it as a training fix.** The positive first pair is retained, not generalized to the cohort.

Original protocol: issue35/comment5816968405. Original execution: run36021389833, source c634e052e19dd48b1c72627f54e4a9a14ad36cf1. Full-cohort result and fixed replay selection: issue35/comment5821257570.

## Full-cohort result

Every short-panel cell below is extra-ordinary -> extra-transient, out of256. Both arms fit actors AND critics, preserve the original1024 raw rows and append512 rows; they match added/fitting work, not the unaugmented learner's budget. Global advantage normalization is recomputed. The seeds are exposed development seeds.

|Seed|Ordinary deterministic|Ordinary stochastic|Outward stochastic|Difficult-reset deterministic|Difficult-reset stochastic|
|---|---:|---:|---:|---:|---:|
|41001|256 -> 256|255 -> 255|232 -> 243|237 -> 247|242 -> 248|
|41002|255 -> 251|251 -> 249|240 -> 236|235 -> 221|245 -> 227|
|41004|254 -> 256|251 -> 253|242 -> 250|237 -> 247|233 -> 250|
|41006|255 -> 255|256 -> 256|242 -> 242|246 -> 246|241 -> 239|

Pooled difficult-reset successes increase only9:43 gained and34 lost. Training-seed mean completion effects are +0.586 percentage points deterministic and +0.293 stochastic; unchanged descriptive99% df3 intervals are[-12.371,+13.543] and[-16.509,+17.095] percentage points. These are not thousands of independent training seeds. The registered screen is false, not merely statistically inconclusive about safety.

The final five-minute ordinary deterministic/stochastic counts, each /32, are32/31 ->32/31 for41001,31/32 ->30/32 for41002,32/32 ->32/32 for41004 and41006. Sixty-second outward counts are31 ->31,28 ->25,29 ->30,30 ->30 respectively. Seed41002 loses46 final panel-cases and gains none;41006 loses two stochastic-stress cases. Across these48 losses,44 terminate at the position boundary and4 at the angle boundary. Panel-cases are not independent training runs.

## Late loss, not simply failure to learn

For41002 extra-transient, the SAME first64 cases at4096 and4608 give difficult-reset deterministic60 ->57 and stochastic62 ->56. Across all five short panels,12 previously successful cases are lost and none gained;11 position endings and1 angle ending. Across both arms/all four seeds, there are28 late loss pairs.

Seed41002 extra-transient is better than extra-ordinary on ALL five early1024 panels but worse on ALL five final panels. The half-outward supplement begins4097, but association does not establish that transition as the cause. Do not infer actor/critic attribution from endpoint counts alone. Saved actor/critic weights omit Adam moments and environment/RNG state and cannot implement an exact continuation.

## Verification completed before the new native replay

All2,930 build/run manifest-listed payloads verify. The original cohort analyzer reproduces its published JSON byte-for-byte. The separate standard-library analyzer reproduces per-case contrasts and four-training-seed intervals. Its18 actual-data/synthetic mutation tests pass. The original downloaded native executable passes114 ordinary tests with10 ignored. This does NOT claim the new replay executable has already qualified.

## Fixed evaluation-only diagnostic

Select both final actors for EVERY control-success/treatment-failure case; select4096/4608 actors for EVERY late loss on the same first64 in EITHER arm; add all64 existing final rep0 full-trace controls. Deduplicate policy/case identities. This gives202 requests and a cap budget of972,160 simulation steps, with ZERO training updates. Every request corresponds to an already-recorded case. No unmeasured4096 replicate is invented, no endpoint is selected as a training result, and no new held-out claim is made.

`prepare.py` reuses the original preparation chain and appends one test-only child. `native.rs` validates and decodes saved actor bytes using native shape/metadata, then calls the unchanged evaluator with the original reset, noise/action keys, simulator, reward, horizons and evaluation discount. It never trains or imports optimizer state. Three new native controls check complete snapshot/inference/trace roundtrip, wrong length and nonfinite rejection.

`analysis.py prepare INPUT OUTPUT` validates manifests, initialization,18688 evaluation rows, exact matching case keys, full cohort and independent contrasts, then writes an exhaustive request list and expected outcomes. `analysis.py verify INPUT OUTPUT` requires all202 native outcomes to match exactly, all64 old full traces to be byte-identical, trace continuity and reward/maximum/applied-force reconstruction, and a zero-training completion receipt. Only then should physical lost-case sequences be interpreted.

The dedicated workflow downloads all original artifacts from run36021389833, runs the request/analysis and native controls, exports the exact qualified executable, executes only the explicit replay endpoint, verifies results, restores protected sources and retains source/build/result manifests. It does not trigger or repeat the eight training arms. No master, production crate/default or deployed-policy change.

## Original artifact recovery

|Artifact|ID|SHA256 ZIP|
|---|---:|---|
|build|10817355765|d76b73cb79b9127fc9bb48c3fe654a9baf157b62ccacd1345e052becbfb3caf0|
|comparison|10824228845|ca760b0528aed6834e422eee764c12ce8bb69cfc404047f23d3b7431782c691b|
|41001 ordinary|10819007711|fb643f479ccc69e75b1fc2de7c578d0bcbb14402c44d6c5f91653803ee0efdc3|
|41001 transient|10819547603|79f79d2e09685dbfd9ff07fb752919b06862011bb51c46efb2c776037f99f5a3|
|41002 ordinary|10820653100|cb4a297a9a3b1f6218bd0c433763aa20a34f3c629e9e686ce902225fd275c18a|
|41002 transient|10821375469|a82f5cee4970b6e7c14505716cabfcc0b0a8b813ca8119df69f3f248b3901902|
|41004 ordinary|10822926954|d98fddff8992095e45a5a58ad3a91cd56e5fafb5c8d7c73dcf447e09c49788dc|
|41004 transient|10823393781|347f6a90c941f55a216daa98ff54005f2b47a4d7bbead738ecb243082694c245|
|41006 ordinary|10824566356|fd3ce76a6461f5cc044e35621345dfb544adf29b5607beada904f8e16521807c|
|41006 transient|10825383553|bd4cc597586d54d2a116af4c5c3e830204c638cb7d6e92989648caeb458756e8|

Artifacts expire. Preserve the original ZIPs, not merely this chat's temporary paths. The new replay is an exposed-case diagnostic, not a production correction, automatic acceptance gate or solution to the sustained-balancing issue.
