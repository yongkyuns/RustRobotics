# Timeout256: original hosted result, not the separate local execution

October 6, 2026 (America/Toronto). Issue35/comments6027852640 and6027940114.

**Reject 256-step bootstrapped training resets as a complete solution for this recipe.** Original run37544083790, source5fe425b921646e51aca4dd24cc719556a57600c7, completed all four seed pairs on attempt1. This continuation recovered and independently verified its original artifacts without rerunning measurement training.

## Endpoint on the shared hosted evaluation panel

One environment,2048 samples/update, stock SB3 PPO, gamma.99/lambda.95. The only arm change is training episode cap5000 versus256. Each arm/seed received1048576 training interactions,512 updates and163840 Adam steps. Both hosted arms evaluate on domain930000+seed; deterministic panels contain32 ten-second cases, stochastic panels64 fifteen-second cases. Training truncations do not shorten evaluation.

| Seed | Control deterministic /32 | Timeout256 deterministic /32 | Control stochastic /64 | Timeout256 stochastic /64 |
|---:|---:|---:|---:|---:|
|201|21|21|33|36|
|202|16|1|35|0|
|203|11|7|15|8|
|204|23|8|38|8|
|Pooled descriptive counts|71/128|37/128|121/256|52/256|

Candidate stochastic completion progresses **237/256 at65536 ->219/256 at262144 ->52/256 at1048576 interactions**. These are four trained development policies, not256 independent training runs.

Every candidate update has at least8 episode starts:2048 updates across four seeds, zero reset-free batches. Yet all final candidate failures are position/rail failures:91 deterministic and204 stochastic, with zero angle endings. Stochastic discounted return improves on EACH training seed, mean75.506536 ->81.072499, paired mean effect+5.565963. The frozen short reliability screen fails.

Coverage changed as intended, but removing missing-reset batches did not establish sustained centering. Discounting is a plausible contributor, not an isolated cause. At dt=.01, gamma.99 weights a reward ten seconds away by about4.317e-5. Bootstrapped256-step segments still depend on estimated continuation values; they are not a literal256-step task horizon. Critic error and off-support generalization remain possible contributors.

## Provenance: do not substitute the earlier local counts

The separate local report6027135157 gave candidate62/128 deterministic and107/256 stochastic, including a strong seed203 result. Those are not the hosted measurements above. Historical pool evaluation used domain920000; hosted control and candidate both use930000.

The old workflow installed Python3.11.16 and generic Torch2.8.0 rather than the preserved Python3.11.14/Torch2.8.0+cpu runtime. Every hosted pair starts from exactly equal tensors. Comparing original pool-control to hosted-control checkpoints, seeds201/202/204 match at all four saved checkpoints, but203 differs even at initialization. CPU/build dependence is not causally isolated, and local candidate203 policy bytes were not recovered here. Do not claim cross-runtime exact replay.

The original hosted artifacts also omit native/sources/.github/workflows/ppo-clean-reference.yml because hidden-file upload was disabled. The reader first rejected that missing entry, then explicitly recovered it from the independent hash-verified original native ZIP and checked every present hosted member against the original. No original artifact was modified.

## Verification and evidence

Independent readers check4096 update/coverage records,3072 evaluation records,32 saved networks, final policy/Adam identities and counters, complete boundary ledgers and568 original-native manifest payload hashes across the four hosted artifacts. A separate reader checks original historical archive/member hashes before tensor comparison. Eight actual-record tests cover missing/duplicate cases, nonfinite scores, fake completions, wrong checkpoints, shortened evaluation and mislabelled endings. Readers perform zero training and zero native calls; they do not reconstruct unretained gradients or every historical random draw.

Original hosted IDs201/202/203/204:11451032992/11451551895/11451545349/11451297552. Original hosted measurement cost:8388608 training interactions and2480354 evaluation transitions.

The conversation evidence attachment **ppo-time-limit-hosted-evidence-20261006.zip** is34656542 bytes, SHA256 **2d864f833b2b29befe070b1115eb7dae0a1153e9ac5af2e3374f081ef5c18bfd**. A fresh extraction verifies331 package payloads, reproduces both numerical JSON reports byte-for-byte and passes all8 reader tests. It includes unchanged original archives, readers, tests, the full report, native input, preflight receipts and the initial next-experiment failure witness. Runtime installers are not redistributed. The detached verification receipt is ppo-time-limit-hosted-verification-20261006.json.

## Next fixed experiment, not a completed learning result

Protocol6027141679 keeps timeout256 and compares gamma.99/lambda.95 against gamma.999/lambda1, without changing PPO mechanics, reward, resets, plant or noise. This is a joint temporal-parameter test, not a separate identification of gamma versus lambda. Execution binding6027852640 uses a new same-host paired reference on every seed, rather than historical cross-runtime controls. Evaluation uses950000+seed and the unchanged fixed acceptance thresholds: every candidate seed must complete at least31/32 deterministic and61/64 stochastic cases, lose neither completion count against its paired reference, and improve mean stochastic discounted return. Long qualification remains conditional, followed by fresh held-out training qualification.

Initial run37550323733 passed every hosted preflight but failed before main training on my checkpoint-zero snapshot bug: SB3 had not yet populated its last observation. No main learning outcome was produced. The original seed201 failure archive and all305 verified payloads are retained. Its four preflights consumed81936 training-environment transitions outside the main budget.

Fix **0db6cae555cabb7bc7e2676d535ebf14365e0504** preserves uninitialized None values without resetting or mutating training state. Nine local preflight groups pass. A two-arm regression reproduces the old failure and passes with the fix, adding zero training transitions and15494 evaluation transitions. Local old+fixed preflights used40968 training-environment transitions plus additional evaluation work.

Corrected **run37550853465** is active with the same registered four pairs and budgets. It restores immutable hashed runtime inputs, strictly checks Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB32.9.0/Gymnasium1.3.0, and retains runtime/CPU/source receipts, the original native ZIP and hidden files. Its main budget remains8388608 interactions, with81936 additional preflight training-environment transitions per completed four-job attempt.

**No completed physical-horizon learning result, held-out qualification, production/default change, master merge, protected-source change or issue resolution is claimed.**
