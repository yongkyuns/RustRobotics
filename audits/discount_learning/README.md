# Discount-learning comparison — submitted, native execution pending

September 20, 2026. RustRobotics #35. Registration comment5749864155 precedes all new training outcomes. This is an actual learning experiment, not a re-score of fixed trajectories. At this report's latest check, workflow **35511524659** at source **6516572add5509ccb338e4ecb55d572733f99bd2** still has native preflight106080137542 QUEUED, with no executed steps. No new native-test pass, trained policy, survival score or robustness result is claimed.

## Fixed comparison

Two arms, training gamma0.99 and gamma0.995. Eight paired previously exposed development seeds41001–41008;16 from-scratch training runs. Gamma is set before the first target is computed. Both actor and critic initialize randomly with the same weights/random streams per seed. No warm-start weights, changing a trained critic's objective, separate critic pretraining, teacher or added controller.

Both follow exactly the existing recovery recipe's two phases:4096 ordinary-reset recovery-union updates, then512 updates with four ordinary-reset and four outward-start supplemental streams. The nonlinear plant, noise, rewards, network architecture, action distribution, global advantage normalization,lambda1,Adam settings,minibatches,epochs and prescribed sample-support lengths are unchanged. Only the training discount differs.

The discount reaches primary GAE, sampled-tail sums, primary cutoff substitution and supplemental returns. Five new native tests exercise propagation/physical-data isolation, terminal/timeout bootstrap semantics, repeated optimizer history and common evaluation scoring. These are required preflight checks, not claimed passes while queued. The already measured recipe bodies are reused without replacement; preparation only attaches a child driver to the existing audit tree. The earlier Sept8 gamma experiment predates this corrected plant/recipe and is not substituted for this test.

## Controls and evaluation

The gamma0.99 actor AND critic must reproduce the archived comparator at updates4096/4128/4224/4608 exactly. The comparison files are never loaded as a warm start. Any mismatch stops and is retained. Both gammas preserve their live optimizer/environment/episode/random state through all4608 updates.

Checkpoints0/1024/4096/4128/4224/4608 are fixed; only4608 decides advancement. At each checkpoint,64 episodes on the existing20.48-second mean-action nominal,stochastic nominal and outward panels. Final evaluation adds64 five-minute mean-action nominal,64 five-minute stochastic nominal,and64 sixty-second outward trials per seed/arm. Evaluation uses the new fixed offset0x08000000 and is isolated from learning. Environment noise remains active; a true failure ends a trial immediately.

BOTH arms' recorded discounted evaluation scores use the SAME gamma0.99. This prevents comparing numbers on different return scales. Common undiscounted return and survival are retained. Gamma0.995 rescoring of the exported full traces is secondary accounting, not invented aggregate rewards or a learning result. Replication0 full traces are retained at4096 and4608; other cases retain outcomes.

Development advancement requires a resolved positive outward-completion gain under paired-history99% Student intervals adjusted across the three long completion comparisons, pooled nominal counts not below the matched control in either panel, and at least61/64 nominal successes for every candidate history. The original absolute reliability screen is separately reported:each long panel at least507/512 pooled,every history at least61/64,family-adjusted one-sided history lower bound at least95%,and at least99% final-ten-second centring among survivors. No best-checkpoint selection, excluded weak seed, changed threshold or favorable retry. These are exposed histories, not fresh held-out qualification, even if numerical criteria pass.

## Budget and limitations

Per arm/history:2,359,296 primary +21,233,664 supplemental +up to9,437,184 sampled-tail interactions;73,728 Adam steps and18,874,368 gradient-sample visits per network. Both follow the same prescribed budgets; actual tail counts may differ because failure ends a tail. Preflight/evaluation experience is additional. This is not sample-efficiency evidence versus ordinary PPO.

Keeping the same support windows means gamma0.995 gives the final bootstrap greater weight. Therefore this tests the practical gamma-only change, not an isolated objective horizon with value-estimation difficulty artificially held constant. No gamma change is adopted based on the earlier single recovered reward-ordering example.

## Prepared offline verification

The independent Python verifier passes48 synthetic/property tests, including both discounts' target semantics, strict endings, fixed keys/denominators, common-score scale, archive corruption, missing evidence, and rejection of nominal regressions or hidden weak histories. Fresh-unpack verification rechecked9 payload hashes and reran all48 tests. An empty raw directory explicitly yields INCOMPLETE for17 required archives and exit code2; it emits no survival scores.

The verifier is prepared to check all current build/result hashes, source bindings,historical weights, first-update data identity, selected target/loss calculations and exported physical trajectories. It has NOT been tested on new native artifacts because those do not yet exist. It does not independently regenerate unexported Adam states or every trajectory and never executes a native binary or training.

Conversation package **rustrobotics-discount-learning-prepared.zip**,19,570bytes,SHA256 **260569c29999fe16f0b54cb185b1a54004208631dad425381ad81b1b28e8ed74** contains the standalone verifier,48 tests,pending status,local test logs/versions and replay instructions. Native driver/preparer/workflow are pinned in the repository. No current measurement output is misrepresented as present.

No production gamma,lambda,reward,global normalization,learned weights,PR38,master or deployed asset was modified or merged by this study. Do not submit a duplicate to bypass a queue; a successful preflight would still not be a passing controller qualification.
