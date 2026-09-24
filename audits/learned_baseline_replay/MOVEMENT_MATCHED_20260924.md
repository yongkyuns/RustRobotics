# Movement matching: paired credit retains a recovery benefit at one witness, not both

24 September 2026. **At equal original-batch average policy KL, paired credit gives12 more recovery completions than baseline-only correction at4195, but9 fewer at4140. Contracting the larger paired step does not eliminate nominal losses.** This completed experiment is a post-update parameter-displacement diagnostic, not a native or online training implementation.

Protocol issue35/comment5811891721 was recorded before contraction scales or new outcomes. Previous endpoint source: continuation-precision report497d44df and exact conversation ZIP SHA25612e49bef833cb5aca508bd165f91e85666d09a870c131d999ce4bf43f982ba59. No intermediate checkpoint, continuation-draw subset or evaluation domain was selected after outcomes.

## Controlled comparison

Reuse each witness's exact archived incoming actor and completed16-transaction original/full8/paired8 numerical endpoints. Full8 uses original returns minus the eight-draw baseline; paired8 uses eight-draw paired Q-minus-V. Their incoming Adam histories, original fitting data and minibatch order remain as previously verified.

For each corrected endpoint form theta(alpha)=float32(theta0+alpha*(theta_endpoint-theta0)), using float64 interpolation then float32 rounding. Target the ORIGINAL updated actor's mean Gaussian KL relative to incoming over the original1024 fitting observations: mean((mu_new-mu0)^2)/(2*sigma^2), with sigma=float32(.1) and independent float64 network inference on float32 parameters.

Forty deterministic bisection iterations on[0,1] are performed, choosing the evaluated point closest to the target, lower alpha on exact tie. Require positive target, endpoint KL at least target, and relative matching discrepancy<=1e-4. Preserve every probe and final actor hash before any evaluation. The scalar solve receives no rewards, evaluation states, independent credit labels or completion counts. Global monotonicity or uniqueness of the neural-parameter-ray KL root is not asserted.

|Witness|Target KL|Full8 displacement fraction|Paired8 displacement fraction|
|---|---:|---:|---:|
|41006/update4140|.005862666891|.743150864|.359990833|
|41004/update4195|.004771813186|.757620930|.251896910|

All four relative discrepancies are<=2.6212e-8. The original target differs from its archived float32 KL by at most4.433e-9 absolute. Independent scalar and Torch float64 checks agree.

These matched actors are POST-UPDATE contractions, not16 new PPO transactions at another learning rate. No optimizer state is installed alongside a contracted actor, and no continuing-learning compatibility is claimed. This isolates final displacement length conditionally on an existing complete optimizer path; it does not implement a line-search trainer or practical baseline estimator.

## New external evaluation

Unchanged NumPy/PCG64 evaluator, fixed new roots2026092441/2026092442. Six actors per witness,512 paired cases per panel,2048 ticks at dt.01 (about20.48seconds). Deterministic refers to policy actions, not absence of environmental noise. All18432 episode records, totals, discounted rewards and paired case changes are retained. These are not native Rust/StdRng results or independently trained policies.

|Witness|Actor|Nominal deterministic|Nominal stochastic|Outward recovery|
|---|---|---:|---:|---:|
|4140|Incoming|509|506|416|
||Original|509|506|380|
||Full8|509|506|443|
||Paired8|509|506|452|
||Full8-matched|509|506|439|
||Paired8-matched|509|506|430|
|4195|Incoming|510|510|411|
||Original|509|510|411|
||Full8|509|510|441|
||Paired8|510|510|490|
||Full8-matched|509|510|436|
||Paired8-matched|509|510|448|

At4140, paired-matched versus full8-matched gains0/loses9 recovery cases. At4195 it gains12/loses0. Thus larger WHOLE-BATCH AVERAGE KL is not the entire explanation for the paired direction's4195 benefit, but that benefit is not uniform across witnesses.

Matched corrections versus incoming, recovery gained/lost:4140/full8 25/2 (net23),4140/paired16/2 (net14);4195/full8 25/0,4195/paired37/0. Observed absence of lost recovery cases is not a general safety guarantee.

### Paired-matched minus full8-matched discounted returns

Ordinary descriptive99% paired intervals retain mean +/-2.586*SE. They are pointwise and unadjusted, not independent-training-seed confidence bounds or formal noninferiority guarantees.

|Witness|Nominal deterministic|Nominal stochastic|Outward|Outward completion change|
|---|---|---|---|---:|
|4140|+.041478 [.029382,.053575]|+.036398 [.027297,.045499]|+.028340 [.001765,.054915]|-9|
|4195|+.100056 [.077382,.122731]|+.108095 [.083141,.133048]|+.166423 [.147348,.185499]|+12|

At4140 the paired-matched actor has higher discounted return yet completes fewer recoveries; its undiscounted outward episode-total difference is-27.0022. This is another direct completion-versus-discounted-return ranking discrepancy at matched average movement.

Matched outward return versus incoming:4140/full8+.330531 [.173947,.487115],paired+.358871 [.204210,.513531];4195/full8+.191528 [.158294,.224762],paired+.357951 [.307770,.408133].

At4195 both matched actors lose the same single incoming nominal deterministic success. Full8-matched nominal-det return is-.032356 [-.098689,+.033977]; paired-matched is+.067700 [+.000927,+.134474]. Improved mean discounted reward does not erase the lost completion.

## Smaller steps are not monotonically safer

Contracting paired8 reduces recovery from452 to430 at4140 (0 gained/22 lost) and490 to448 at4195 (4 gained/46 lost). At4195 it also loses one nominal deterministic success that the UNCONTRACTED actor completes,510 to509. Other nominal completion counts stay unchanged. An excessively large average KL is therefore not a sufficient explanation for all nominal losses, nor does contraction automatically make a step safe.

Both matched corrections pass the unchanged local screen on4140 and fail on4195. The uncontracted paired8 actor passes both screens in THESE new domains. Earlier domains where it lost nominal successes remain valid contrary evidence; the favorable new cases do not establish safety or authorize choosing this domain. The previous native causal results and failed online/cheap-head studies remain valid for their own cases.

## Mean KL does not control its regional allocation

At4195 the two matched corrected actors have identical mean KL.004771813, but different regional movement:

|Fitting provenance|Full8-matched mean KL|Paired8-matched mean KL|
|---|---:|---:|
|Primary continuing rollout,512 rows|.002566557|.000011899|
|Nominal-start supplement,256 rows|.006000373|.003908159|
|Outward-start supplement,256 rows|.007953767|.015155295|

Paired-matched changes the primary distribution roughly216 times less while changing the outward-start supplement about1.9 times more. Provenance is not a physical recovery label; separate outward-moving-proxy, maximum and95th-percentile KL statistics are retained too.

On separately drawn panel INITIAL observations, scored only after alpha was fixed,4195 nominal-det KL is.0038717/full8-matched versus.0013653/paired-matched; outward-start KL is.0037998 versus.0073415. This is not equality over future visited states or a worst-case trust region.

All8-credit-gradient full-displacement projections:4140 original-.062012,full8-matched+.067801,paired-matched+.077411;4195 original+.006317,full8-matched+.058778,paired-matched+.072155. The4140 projection ranks the two corrections differently from recovery completion. These are sampled local derivatives, not expected reward. Monte Carlo estimates still use actual pre-action simulator states, limited draws and a far-cutoff critic bootstrap.

The bounded conclusion is that the direction and regional allocation of movement matter in addition to average length. Scalar KL matching does not isolate estimator accuracy from every effect of nonlinear parameterization, state weighting, visitation and finite-data uncertainty. This validates no automatic scalar-KL gate, production critic or continuing learner.

## Verification and cost

The full prior741-member ZIP manifest was verified before extraction. The new package retains483 exact original-manifest-bound members, including its348-member support set, inherited source, original/full8/paired8 transactions and endpoints. This is not a fresh complete audit of every old historical archive.

All96 inherited numerical ACTOR Adam recurrences verify bit-for-bit from exported gradients/moments/counters, and all final endpoints equal the16th saved checkpoint. Existing original-native actor/critic controls also pass through the retained input loader. No new gradients or optimizer steps are claimed.

**22 actual-data/mutation tests pass**, and pass again from a fresh extraction. Tests check scalar/Torch KL, constraint roots, exact interpolation/endpoints, malformed/impossible/nonfinite inputs, corruption rejection, all outcome identities/paired statistics/screens and both outward repeats. The complete constraint/identity repeat reproduces14 files, including all12 actors, byte-for-byte. Both outward panels repeat all6144 episode records and metadata exactly. The four nominal panels were NOT rerun in full. Repetition is reproducibility, not a new statistical sample.

Main evaluation executes36,069,377 active transitions; outward repeats11,063,714. No new critic fit, MC baseline estimation, actor Adam transaction, native CI run, online continuation or from-scratch cohort. Inference, constraint solving, recurrence reconstruction and hashing are real computation; inherited endpoint and baseline costs remain.

Logs retain unrelated nonfatal artifact-tool environment-startup messages; every numerical command exits successfully and all fixed study gates pass. No candidate, target, root, tolerance or behavioral criterion changed after outcomes.

Conversation bundle `rustrobotics-movement-matched-20260924.zip`:23,179,085bytes, SHA25650303c7710a5f00c2f177462bdbe80438bd137a3d4ec6674a4e71066da3f8376,537 manifest-bound payloads. It contains full local report, source/tests, retained provenance, all matching probes/actors, outcomes, repeat evidence and logs. Readable report: `rustrobotics-movement-matched-report.md`.

Only this summary report is added to the repository outside workflow triggers. Production/defaults/master/deployment and previous experiment code remain unchanged.
