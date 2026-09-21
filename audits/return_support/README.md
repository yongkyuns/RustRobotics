# Gamma0.995 with longer sampled return support

Registered before new outcomes in issue35 comment5753874883. Executable source **0a4445f6c5ab83650133a696faaeb764fd5b6c26**, workflow **35548063439**, branch **audit/ppo-return-support-20260920**. At this status check the native preflight is running; no native completion, training score or controller improvement is claimed. Documentation-only commits do not launch another measurement.

## Fixed question

The preceding gamma-only experiment improved nominal survival but regressed outward recovery. Does more sampled future reward support at the SAME training discount address that weakness? A nonterminal cutoff bootstrap has coefficient0.995^512≈0.07681 at512 steps and0.995^1024≈0.00590 at1024 steps. The latter is close to the preceding gamma0.99/512 coefficient0.00582. These are arithmetic weights, not measurements that the critic caused the regression.

Two arms only, support512 and support1024. Both use gamma0.995 from random actor/critic initialization. Sixteen total runs pair seeds41001–41008; the seeds are already exposed development histories, not a fresh confirmation cohort. Each performs4096 ordinary-reset recovery-union updates then512 half-outward updates with persistent Adam/environment/episode/random history. No warm-start weights or changed objective halfway through training.

| Quantity | support512 control | support1024 candidate |
|---|---:|---:|
| Primary fitted samples/update |512|512|
| Supplemental streams |8|8|
| Fitted samples per supplemental stream |64|64|
| Collected transitions per supplemental stream |576|1088|
| Maximum steps per primary-cutoff future |512|1024|
| Primary-cutoff futures |4|4|
| Total fitting samples/update |1024|1024|
| Actor/critic optimizer transactions per update |16 each|16 each|

Extra observations/actions after the selected64 supplemental samples only provide return support; they are not added to actor/critic fitting. True terminals stop credit, so no rewards cross a physical failure/reset boundary. The main stream stays512 transitions/update. The existing scoped outward-start mixture,lambda1,global normalization,rewards,noise,nonlinear plant,architecture,exploration,Adam settings,minibatch256 andfour epochs are unchanged. No teacher,imitation,gradient projection,symmetry wrapper or inference fallback is introduced.

The test-only support scope defaults to512 and restores on exit/unwind. Existing tail and supplemental collector bodies are reused with reversible local length hooks. All old controls still use historical lengths. Seven new native controls cover scope/thread isolation,unchanged short dispatch and following optimizer state,first physical/fitted sample identity,tail-prefix identity,terminal/cutoff math and reproducible long-support updates. Those are required tests, not completed results while preflight runs.

## Reproduction and evaluation

The short arm must match BOTH actor and critic from the prior gamma995 run at every saved checkpoint0/1024/4096/4128/4224/4608. Every one of4608 update records (after the arm label) and final costs must match too. Historical artifacts are checked by immutable ZIP/member hashes and used only as comparisons, never to initialize training. Any mismatch stops and is retained. Both arms' initial main paths and first selected supplemental physical samples must pair; future rewards and subsequent training trajectories may differ.

All checkpoints are retained, only4608 decides advancement. Detailed updates1/4097/4608 export incoming weights,main batches,all return-support rows,targets and16 optimizer transactions. Evaluation uses fixed new offset0x10000000,64 replications per seed/arm/panel,unchanged noisy plant and COMMON score-gamma0.99. Short20.48-second nominal-mean,nominal-stochastic andoutward panels run at every checkpoint; final also runsfive-minute nominal-mean,five-minute nominal-stochastic andsixty-second outward panels. Failure ends the trial immediately. Replication0 full traces at4096/final; others retain outcomes. No outcome enters fitting or selection.

Development advancement requires a positive lower bound on outward-completion gain from the eight paired history effects using two-sided99% Student intervals adjusted overthree long-completion contrasts. Both candidate nominal pooled counts must be>=control and every candidate nominal history>=61/64. Original absolute screen is separate:each long panel>=507/512 pooled,every history>=61/64,three-panel-family one-sided99% history lower bound>=95%,and>=99% centring among survivors in their final10seconds. Missing/failed histories cannot pass. No threshold relaxation or best-checkpoint substitution. Passing an exposed-history screen would not constitute hardware qualification.

## Work and limitations

Per history/arm, primary interactions2359296; supplemental21233664(control) or40108032(candidate); cutoff futures at most9437184 or18874368. Each network performs73728 Adam steps and18874368 gradient-sample visits in BOTH arms. Evaluation and generic preflight work are separate. This is matched fitting/optimizer work, not matched total sampling or a sample-efficiency claim.

Increasing support changes realized sampling variance and the target's reliance on a later critic value, not just a scalar coefficient. A successful result would not uniquely identify bootstrap error as the cause of prior harm. Conversely, a failed result would reject this recipe change, not prove all longer-horizon estimators ineffective. No alternative horizon or preferred seed is selected after outcomes.

## Offline verification prepared

The standalone verifier passes63 synthetic/property tests:48 adapted identity/numerical/statistical controls plus15 support/scope-preparation checks. It rejects missing/corrupt evidence,wrong windows,nominal regressions,hidden weak histories and credit crossing true terminals. With all17 required archives absent it emits INCOMPLETE/exit2 and no controller scores. These are verifier tests, not new native or learning results.

The generalized batch reconstruction was also run on the already-archived gamma995 seeds41001 and41004 at512 support. Its complete results matched the inherited verifier exactly, without resimulation. This validates compatibility with old artifacts; it is not a native comparison or evidence about1024-support training. Local source transformations were checked for exact reversibility on the actual archived collectors.

Prepared verification binds all future run/build hashes and historical files; reconstructs both window lengths' GAE,tail sums,cutoff substitution,returns,global normalization,selected optimizer outputs and full exported evaluation paths; and checks first-update sample identity. Unexported Adam states,gradients and remaining trajectories are not independently regenerated. Offline replay uses no native executable,network,simulator or training.

No production gamma0.99/lambda0.95,reward,global normalization,learned weights,PR38,master or deployed asset changed. No merge or controller adoption. The experiment is submitted with native execution still pending completion; do not treat a green verifier or arithmetic weight reduction as robust control.