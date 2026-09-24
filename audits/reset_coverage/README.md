# Reset-transient actor-fitting coverage

Protocol: RustRobotics issue35/comment5816968405. Base056428249f11e9019d33cdca648f17d02d244f2a. This is an audit-only from-scratch development comparison; no completed learning result or deployable correction is claimed by this source submission.

## Question

Does adding difficult reset transients to the ordinary PPO fitting batch improve their handling without sacrificing ordinary or outward recovery, compared with an equal number of additional ordinary-reset samples?

The two arms are `extra-ordinary` and `extra-transient`. Both preserve the original1024 fitting rows (512 primary +512 supplemental), including raw advantages, and append512 rows from four independent current-policy streams. Select the first128 observations per1152-step stream, with1024-step future support unless a true terminal occurs. This includes the first1.28seconds and its high-speed response; it is not merely four extra initial observations. After true failure, both collectors use ordinary resets. Targets use the existing gamma.995/lambda1 implementation and incoming-critic far bootstrap, not an oracle.

Transient starts use the previously fixed within-reset-box subset: upper-half absolute angle/angular velocity and matching signs; x/v retain their ordinary ranges. All training/evaluation random domains are disjoint. Stress evaluation offsets the base0x78000000 seed domain by0x00800000 before using the existing evaluator's keys. The same reset formula is shared, not the sampled states.

Both arms use1536 rows, minibatch256, four epochs,24 actor and24 critic Adam transactions/update. The original1024 normalized advantages can change because the FINAL whole-batch normalization is recomputed; raw-row preservation is not frozen gradient influence. Actor and critic both receive the new data. This changes the fitting-state distribution and does not claim unbiased reweighting or separate actor effects completely from critic effects.

## Fixed cohort and budgets

Seeds41001,41002,41004,41006, two arms each,4608 updates from random initialization. Original supplemental curriculum remains: ordinary through4096, half outward from4097. These seeds have been exposed in development; they are not held-out final validation. No checkpoint import, pretraining, oracle baseline, additional critic-only optimization, rollback, reward/gamma/lambda/LR/clipping change, or outcome-based stopping.

Per arm/seed:2,359,296 primary transitions;40,108,032 original supplemental;21,233,664 added; at most18,874,368 tail transitions;110,592 Adam transactions and28,311,552 sample visits PER network. Tail terminations can change actual collection counts across diverged policies. Matching the added-data budget and fitting work does not imply identical total realized simulation cost. Neither arm matches the unaugmented learner's compute.

Keep checkpoints0/1024/4096/4608 and detailed updates1/4097/4608. At each checkpoint evaluate five short panels: ordinary deterministic/stochastic, outward stochastic, stress deterministic/stochastic. Earlier panels64 cases, final256, plus32 each five-minute ordinary deterministic/stochastic and60-second outward. That is2336 records/run,18688/cohort. Use final prescribed actors, never best checkpoints. Retain total and discounted returns, every ending and paired gained/lost cases. The eight runs use a maximum of two measurement runners simultaneously.

The component screen requires no per-seed aggregate ordinary/outward completion loss, a positive pooled stress gain and no confidently harmful ordinary/outward discounted-return contrast. Report all casewise losses, even when an aggregate screen passes. Four-seed paired t intervals use df3; episode comparisons remain conditional fixed-policy descriptions. These unadjusted approximations are not safety, equivalence or population reliability guarantees. Failed/incomplete jobs are not replaced by zeros or better seeds.

## Implementation boundary and controls

`hooks.rs` only appends precollected immutable batch data inside a scoped test hook. It never receives a trainer or optimizer. `native.rs` collects new data, calls the existing update_support and evaluator, and checks preservation. `prepare.py` appends one test module and makes three reversible edits to the inherited audit driver: append batch rows; account for variable batch size/transactions; optionally supply stress evaluation starts. The production optimize body is unchanged. Normal inactive calls are exact identity.

Native tests cover disabled update/next-optimizer identity, collector reproducibility/live-state purity, exact original data prefix, manual24-step optimizer replay and following state, conditional reset bounds, stress-evaluator scoping, and terminal targets. Python tests cover source restoration on an actual prepared fixture, input/outcome corruption, complete cohort accounting, exact float32 normalization/returns, sampling keys and paired statistics. Source qualification is not a policy-quality result.

Run locally in a pinned checkout:

```sh
export HORIZON_BUILD=/tmp/coverage-build
export COVERAGE_REPEAT_FIXTURE=$HORIZON_BUILD/repeated-before-coverage.rs
python audits/reset_coverage/prepare.py
python -W error -m unittest discover -s audits/reset_coverage -p test_evidence.py -v
cargo clippy --locked -p rust_robotics_train --all-targets -- -D warnings
cargo test --locked --release -p rust_robotics_train --lib -- --test-threads=1
```

The workflow separately invokes the ignored heavy endpoint only after preflight. Detailed result verification binds selected added rows to full source streams and checks support/GAE and each four-epoch permutation. Unexported optimizer moments and all unexported ordinary batches are not independently reconstructed by the analyzer. Source/defaults/master/deployment stay unchanged.
