# PPO update410 continuation: exact replay restored and destructive step decomposed

October14,2026. Continues issue35/comment6178554565 and the four component arms already registered in6177921064. Diagnostic complete; no production correction, PR merge, deployment or issue closure.

## Chronology and disposition

The previous chat attachment reported an Intel-host historical-replay block, but GitHub already contained a later exact trace and transaction60 report. This continuation explicitly treats that result as exposed prior evidence, not a newly independent discovery. On the current AMD EPYC9V74 host, the unchanged strict reconstruction now passes, and the captured warm update replays exactly without instrumentation. The prior Intel failures are not relabeled as valid historical evidence.

The replay blocker is closed. At transaction60, the mean-network change is sufficient for the retained recovery failure; exploration-scale change alone is not. Both the current-gradient-only and carried-first-moment-only components of the actual Adam increment also fail that recovery. Consequently this witness is not explained solely by carried first-moment momentum, exploration noise, critic-dominated clipping or an Adam arithmetic error. No practical learning correction is established by these synthetic single-transaction components.

These are historical stock-SB3 reference policies in the RustRobotics investigation, not newly measured production Rust optimizer steps. All cases are exposed development data. No new independent training seed, reset/noise case, evaluation-based actor selection or default change.

## Historical and warm identity

Pinned Python3.11.14 / Torch2.8.0+cpu / NumPy2.2.6 / SB3 2.9.0 / Gymnasium1.3.0. Actual original native and SB3 source gates pass. The strict reconstruction does NOT restore initial weights, relax numerical comparisons or replace ordinary learning with a fresh-Adam approximation.

All512 original fitting/physical batches,512 diagnostics, the complete episode ledger and six original policy/evaluation checkpoints pass. Checkpoints0,131072,262144,524288,786432,1048576 interactions. Canonical digest `76e003ff39029b70faabe40c9f85aafba8880fb2c86500576637a2b0e063d0f9`; available original incoming model arrays also match. This is comparison to evidence actually retained, not nonexistent all-update Adam snapshots.

Actual update410 runs global Adam steps130880→131200,320 transactions and ten original2048-row permutations. Restoring the captured policy/Adam/RNG/rollout and invoking unchanged SB3 PPO.train with the recorder removed reproduces final policy, Adam and all RNG states exactly, with zero simulator calls.

Independent NumPy loss/clipping/moment/update verification passes all320 transactions: maximum parameter-update error5.9510940e-8, moment error2.3283064e-9. A hand-derived reverse-mode graph checks every raw gradient, max error1.9442413e-6 within fixed tolerances. No tolerance was changed after observing results.

Passing on this AMD host resolves the practical block but does not fully isolate every earlier Intel kernel mismatch or guarantee cross-platform bitwise arithmetic.

## Transaction60 and fixed components

Transaction60 is global Adam step130940, epoch2. Its exact current-minibatch actor loss IMPROVES from−.0243281927 to−.0260670837. Clipped actor gradient dot actual step is−.00057830157. Critic squared-gradient share is5.6698718e-7. The full update contains51 first-order actor-ascent steps and54 actual current-minibatch actor-loss increases, but transaction60 is NOT one of them.

Every arm: the same64 difficult group00 cases, OLD410 first64 actions, chosen actor from tick64 onward, original full post-reset native/RNG clones and standardized action innovations,64 lanes including inactive lanes,1500-tick horizon, noisy nonlinear plant, bounded commands, gamma.999 raw stopped returns and no bootstrap.

|Actor after tick64|Complete /64|Witness3309102759 duration|Witness discounted return|
|---|---:|---:|---:|
|Actual pre59|51|1500 ticks|618.649299|
|Actual post60|50|109 ticks|20.821598|
|Post60 mean / pre59 sigma|50|109 ticks|20.821589|
|Pre59 mean / post60 sigma|51|1500 ticks|618.649957|
|Current-gradient component only|50|109 ticks|20.877959|
|Inherited-first-moment component only|50|109 ticks|20.863014|

All harmful arms lose the same one cart-limit case and gain none. Their64-case mean-return intervals span zero; the selected witness is not a resolved population effect.

The last two arms split m60=.9*m59+.1*g60 while holding the ACTUAL updated second moment/denominator, global bias corrections,lr and epsilon fixed. They reconstruct the actual proposal within2.9760525e-8. This is not a fresh optimizer or momentum-free training run. The current-gradient component still uses historical second-moment scaling.

Both components improve the same sampled minibatch loss: actor losses−.026651 current-gradient and−.026515 carried-moment versus−.024328 before. Thus inherited FIRST-moment momentum is not necessary for this witness; this does not show that all optimizer history is irrelevant or that changing beta1 is a qualified remedy.

The retained per-row fitting readout has2 pre-existing early-outward rows,22 other early rows and40 later rows. Its raw actor-gradient projection is−.004928815, versus the differently scaled clipped-gradient projection above. Neither is a ground-truth action-credit label or proof of coverage sufficiency.

## Full fixed epoch panel

All11 declared epoch-boundary actors were evaluated on ALL512 original difficult cases, OLD410 first64 ticks then the actor. These are diagnostic hybrids, not standalone deployment policies.

|Transactions applied|Complete /512|
|---:|---:|
|0|402|
|32|402|
|64|398|
|96|400|
|128|394|
|160|399|
|192|399|
|224|400|
|256|400|
|288|399|
|320|396|

Recovery degrades and partially returns within the update, not a monotonic loss. Final versus initial:6 losses,0 gains, gamma.999 mean return−7.303382 with descriptive pointwise99% paired interval[−14.408638,−.198126], matching the prior temporal endpoint. No best checkpoint is selected. These exposed fixed-policy case intervals are not new held-out or trained-agent-population qualification.

## Reusable optimizer-only checkpoint

The new313224-byte standalone package contains JSON/NPZ policy/Adam/rollout/RNG state and source/version-gated replay code. Replay uses allow_pickle=False and never calls the native simulator or loads pickle. A shape-only environment rejects reset/step calls. The original PPO.train executes once and all52 policy/Adam arrays, all RNG states and the final clock match exactly.

It passes both the first export replay and a fresh-extraction replay. Final packaging changed ZIP timestamps only; all9 final payloads and manifest are byte-identical to that successfully replayed extraction. This is not a second independent training trial. No .pt file, native library or million-interaction history is needed for portable replay. Export consumes only locally self-created trusted .pt files; do not supply untrusted pickle to export.

Command under pinned runtime:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python portable_warm.py replay checkpoint VERIFIED_NEW.json

This is an optimizer-only checkpoint, NOT a full native-simulator resume or universal cross-host guarantee. Mismatch gates remain mandatory.

Exact executed component_math.py committed b89b3d6a; portable_warm.py committed818b214e. Portable source Git blobf22a768ffdcf7419b73607c10989a1f0826007bf matches the executed file, SHA25615c9fd75c14ee4645d1f8ad706eecab8a5e06076f0ffedfbd24aa0ce2ad93f21. Full collectors/readers/tests are in the evidence package, not falsely claimed production integration.

## Verification, failures and costs

23 NEW tests pass:15 component/corruption,4 manual-gradient/finite-difference,4 real pinned-entry-point checkpoint rejection tests.18 existing trace controls pass against the ACTUAL historical trace;30 existing temporal controls also pass. Total71 distinct tests, not71 new tests.

Independent read-only validation covers all6144 newly executed behavior trajectories:5632 epoch,384 actual/component,128 corrected identity controls. Maximum independent normalized-command error2.7623324e-7<3e-6. Full group/key/policy/switching/prefix/noise/reward-bit/physical-ending/padding/score checks pass. It does not independently reintegrate native dynamics or regenerate private native noise.

A fresh final FULL archive extraction verified1444 payload hashes, exact portable-array binding to the captured trace,67 read-only tests, all320 transaction equations and all6144 trajectories. FIVE numerical reports regenerate byte-for-byte. Four additional checkpoint-rejection tests require the pinned runtime and passed separately. No Torch/native/optimizer/pickle executes during offline reduction.

Native transitions:8192 preflight +1048576 historical training +630280 historical evaluation +7337300 new behavior +77763 initial failed-fixture control = **9102111**. Optimizer transactions including all warm and fresh-package replays: **166400**. Zero new independent agents/cases, no efficiency claim.

The retained temporal fixture is ordinary group06, not difficult group06. The initial control completed OLD difficult00 exactly, then rejected the incorrect fixture-panel binding before a second simulation. Corrected using actual keys, no tolerance change. The initial exporter rejected a Gym Box object before optimization; explicit JSON space metadata plus NPZ bounds fixed serialization. Failed source/logs retained. One streaming request and a network retrieval failed before numerical execution. No main component/epoch simulation retry, selected replacement outcome or unrecorded partial simulator cost.

## Artifacts

- `ppo_optimizer410_verified_warm_checkpoint_20261014.zip`:313224bytes, SHA256 **e2558823223e70309fa3a884329e788e51d4c3bfb43ad8799c427f42c7a414ce**.
- `ppo_optimizer410_resume_evidence_20261014.zip`:490290153bytes, SHA256 **e69ea9a8e684cbcf386289e095b4d89b75d4952c8fe7ff20af53ffe5137da896**.
- Detailed `ppo_optimizer410_resume_report_20261014.md` and detached `ppo_optimizer410_resume_verification_20261014.json` are delivered alongside them.

Full offline REPLAY.py verifies hashes/tests and reproduces all5 outputs. Recorded reader Python3.13.5/NumPy2.3.5/SciPy1.17.0. Complete source/trace/warm states/new trajectories and required original subsets are included. Original subset manifests and omission notices are preserved; large parent archives, runtime installers and fonts are excluded. Outer hashes and final logs/outputs are bound by the detached receipt.

The replay blocker is solved, not PPO reliability. A practical correction must alter learning in a way that retains recovery while improving ordinary from-scratch performance and then pass #33's independent sustained-balancing qualification. No production/default/reward/noise/reset/physics/protected-source/held-out-training-seed/master/deployment change, production PR merge or issue closure.
