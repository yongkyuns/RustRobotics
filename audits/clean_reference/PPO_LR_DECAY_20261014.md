# PPO learning-rate decay: stronger late recovery, strict retention still fails

October14,2026 (America/Toronto). Issue35 protocol6183008716, fixed before main outcomes. Continues59edfadf. Eight new from-scratch runs complete. No production correction, deployment, PR merge or issue closure.

## Main finding

Nonzero learning-rate decay substantially improves late difficult recovery on these four paired development seeds. At the final1310720-interaction checkpoint, difficult stochastic completions are **177/256 constant versus246/256 decay** (69.14% versus96.09%). All four seed-level difficult completion counts improve. However, the preregistered development screen **FAILS**: the decayed learner still loses recoveries during further training, and one seed has slightly lower ordinary raw return. No criterion is relaxed and no earlier checkpoint is substituted.

At1048576, every comparative condition passes: no difficult-count reduction or ordinary-return reduction in any seed/mode, and difficult completion-with50mm gain+11.71875percentage points. At1310720, the difficult-count condition and margin gain still pass, but the ordinary-return condition fails for203. The independently required continued-learning retention conditions fail for204.

These are new paired trainings on AMD EPYC9V74, not exact reproductions of the previous Intel cohort or new measurements of the production Rust trainer. Both arms use the normalized SB3 reference recipe. The original evaluation panels and seeds201–204 are exposed development data, not a held-out cohort.

## Complete learning curve

Difficult stochastic evaluation,64 cases per seed,256 total. Complete+50mm means a completed episode with minimum signed cart clearance at least50mm; this is a diagnostic category, not a changed physical boundary or hardware safety requirement.

|Training interactions|Constant complete|Decay complete|Constant complete+50mm|Decay complete+50mm|
|---:|---:|---:|---:|---:|
|0|0|0|0|0|
|65536|204|192|198|189|
|262144|234|250|232|244|
|524288|238|221|234|218|
|786432|211|243|210|239|
|1048576|218|248|214|244|
|1310720|177|246|175|245|

All denominators256. Decay is not better at every checkpoint: the65536 and524288 results are retained. The measured curve remains non-monotonic. No best-checkpoint selection or evaluation-driven stopping.

## All final difficult-case results

Each cell is completions/64. The extra262144 training interactions start at1048576; the candidate rate remains positive throughout them.

|Seed|Mode|Constant at1048576|Decay at1048576|Constant at1310720|Decay at1310720|Decay50mm at1310720|
|---:|---|---:|---:|---:|---:|---:|
|201|deterministic|56|62|57|64|63|
|201|stochastic|57|62|55|63|63|
|202|deterministic|54|58|45|59|58|
|202|stochastic|54|58|44|58|58|
|203|deterministic|56|64|50|64|64|
|203|stochastic|56|64|50|64|64|
|204|deterministic|52|63|28|62|60|
|204|stochastic|51|64|28|61|60|

During the positive-floor tail, decay204 loses1 deterministic and3 stochastic difficult completions, with zero gains in those panels. Its completion-with50mm count falls62→60 in both modes. Decay201 gains2 deterministic/1 stochastic;202 gains1 deterministic with stochastic unchanged;203 preserves all64 in both modes. All individual gained/lost case keys are retained in RETENTION.json. Equal counts elsewhere are not used to infer casewise retention.

At the final endpoint, decay203 ordinary raw return is lower than CONSTANT by **1.023863 deterministic and0.891097 stochastic** units. Both arms still complete64/64 ordinary cases in both modes; reference means1472.133847/1469.163114 versus decay1471.109983/1468.272017. These small negative effects still fail the declared non-reduction condition; they are not waived.

The mean difficult50mm gain across four seeds/two modes is+27.1484375percentage points at the final endpoint. This positive aggregate does not erase the per-seed tail regression.

## Uncertainty and scope

Pointwise two-sided99% Student intervals use FOUR paired training-seed differences, separately by mode. Modes and trajectories are not additional independent trained agents. At1048576, difficult stochastic completion gain averages7.5 cases/64, interval[-4.303,+19.303]; at1310720 it averages17.25, interval[-14.508,+49.008]. The corresponding ordinary raw-return mean intervals also span0. Four exposed seeds do not certify a population-wide expected benefit despite every difficult seed improving on these measured endpoints.

A pass would only have made this a development candidate. This failed screen does not qualify deployment, continued-learning stability, operational safety or issue33 sustained balancing. It does show a substantive improvement over the paired constant-rate reference, not another no-op optimizer intervention.

## Single-factor recipe

CONSTANT lr=.0003. DECAY lr(t)=.0003*(1-.9*min(t/1048576,1)), with explicit floor.00003. t is absolute COMPLETED training interactions, evaluated after each collected rollout. The schedule is independent of SB3 progress_remaining, which resets between checkpoint learn calls. The final262144 interactions keep the positive floor, never zero. Both actor/critic remain on the original optimizer; no optimizer history is cleared.

Both arms retain stock VecNormalize(norm_obs=True,norm_reward=False,clip_obs10,epsilon1e-8), training-only statistics and frozen evaluation. Original PPO.train, collector and GAE methods execute unchanged. Same original noisy nonlinear native plant, physical reward/failure/reset laws,256-step external training timeout, inherited fixed reward-unit scale1/1000 and divided initial critic head, gamma.999,lambda1,2048 rollout rows,batch64,ten epochs,Adam epsilon1e-5,clip.2,global norm.5,entropy0,two64-unit tanh layers and learned Gaussian scale. No teacher, curriculum, safety override, reward shaping, extra data, selected trajectories, rollback or additional optimizer passes.

Four fixed seeds201–204; equal fresh actor/critic arrays and empty Adam in each pair. Both initial evaluation panels match exactly. Four single-threaded process slots, constant first for odd seeds and decay first for even seeds, sequential within a seed. Each run1310720 interactions/640 PPO updates. Checkpoints0,65536,262144,524288,786432,1048576,1310720. Every run completes once, all process exits0.

All original64 ordinary+64 difficult native-reset cases per seed, deterministic/stochastic, same original reset/innovation keys,64 inference lanes including inactive lanes,1500-tick cap, first original physical failure, raw/gamma.999 stopped return,no bootstrap. Physical state is used only by the native environment/evaluator, not extra actor inputs. Evaluation cannot update statistics, policy, Adam, training observation or RNG.

## Actual continued updates

Every main run applies204800 Adam transactions. Each floor-tail includes **40960** transactions; all actual learning rates are checked before every optimizer step, and all640 update-rate records are retained. An independent rational calculation verifies every recorded schedule value. No restart at checkpoints or zero-rate freeze.

All4546 actor parameters change during the decay tail in every seed. Actor displacement L2 norms:201 .525098065;202 .412552213;203 .486314701;204 .325052994. Model, named Adam arrays and NORMALIZER.json are saved at all56 checkpoints. They are not complete native-simulator-resume checkpoints; statistics are essential inference state.

## Verification and provenance

Pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0 on AMD EPYC9V74. Original native archive58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0 and142 payload hashes pass. Original actual PPO/buffer/policy/collector AND normalizer/RMS source-byte gates pass. Runtime c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1 and11 payloads pass. Parent compact61b564896e00fcf794d5a5fd0ceb8dc8db31198a73316792636af24414e40311 passes267 hashes,26 prior tests and57 prior control arrays before setup.

Seven new live checks pass before main training: original normalized learner versus constant callback/observers byte-exact; actual optimizer LR; grouped/split decay policy/Adam/RNG/normalizer/live-state identity; absolute clock; all four initial pairs; frozen evaluator isolation; independent physical reward/action/clearance fixture. Two control-state pairs export56 numerical arrays each plus complete JSON metadata and match exactly.

**30 new tests pass** (19 initial schedule/decision tests and11 expanded genuine-record/validation tests). They reject missing/duplicate seed/mode records, per-seed/extra-tail failures, nonfinite or fractional counts, restarted/zero rates, altered keys/precision/policy/statistics/initial observations, padding and one-representable-float reward changes. The26 parent tests are retained, not30 additional new tests.

The independent reader validates **14336 evaluation trajectories /17528456 physical transitions**,224 groups,56 checkpoint/source bindings, complete640-update LR records per run, named Adam clocks, training-only RMS counts, complete training episode ledgers, initial pair equality, actions/force/raw float32 reward bits, observation bounds, physical endings/padding, sequential returns and continuous clearance. Maximum independent normalized-command discrepancy3.8132828583670175e-7<unchanged3e-6.

A FRESH final full archive extraction checks **830 payload hashes**, all30 tests, both56-array exact controls, every checkpoint/schedule and regenerates all FOUR reports byte-for-byte (25.72s). Fresh compact extraction checks607 hashes,30 tests,controls/checkpoints and the frozen decision; it explicitly omits full main raw-path reduction. No Torch/native/optimizer/pickle executes offline. The reader does not independently reintegrate native dynamics/private noise or reconstruct unretained training trajectories. No tolerance changed.

## Costs, source and artifacts

Main training10485760 native interactions/1638400 Adam transactions. Main evaluation17528456 transitions. Live controls add36864 training+3947 evaluation and5760 Adam transactions. **Total28055027 native transitions and1644160 optimizer transactions.** Eight new development runs, no new held-out training seeds. No efficiency claim.

A streaming-tool launch request was rejected BEFORE numerical execution; the ordinary supervisor then completed every main run once without interruption/retry. All exit codes and logs retained. The collector and decision hashes were frozen before launch and remain unchanged. Expanded read-only validation does not alter learning or criteria. The complete source patch applies in a fresh Git directory with the existing schedule at its base; all30 tests pass with the supplied genuine fixture. One inherited trailing-blank-line warning is nonblocking and retained.

Exact schedule.py committed **59cb5ac65cf842ca7484f3744cd24e25d8dcb40a**, blob7030eceecd4a3518c03a62a387bab9f28844e1b1, matching executed bytes. Complete driver, independent readers/tests, retained source/input layout and replay are delivered, not falsely claimed production/routine-CI integration.

Artifacts:
- Full ppo_lr_decay_evidence_20261014.zip:742748035bytes, SHA25620a3068e5180cc29ce7974e4aeed95a29c97e08f79c1f9a9f7f79a8503a0c5b3.
- Compact ppo_lr_decay_report_code_20261014.zip:17224586bytes, SHA2565cfa6ec278eaf8f2812b2c84952309b18b14190fdd6d0018147b725919b2f5b6.
- Source patch ppo_lr_decay_audit_20261014.patch:140946bytes, SHA2560d6136102c549399668a800ec5da9dc218586a6f3ddc4863b687813bc949d9b4. Patch base59cb5ac6 already contains schedule.py; do not add it twice. Binary inputs/fixtures are in the evidence package.
- Detailed report and detached verification/logs accompany them.

Full REPLAY.py verifies hashes/tests/controls and regenerates RESULT.json,CHECKPOINTS.json,CASES.json,RETENTION.json. Reader Python3.13.5/NumPy2.3.5/SciPy1.17.0. Compact retains all56 model/Adam/statistic checkpoints, all case summaries and control fixtures, not main raw traces. Parent manifests and omissions are documented; runtime installers/fonts and duplicated extracted native libraries are excluded. README documents separate exact-version execution; these checkpoints are not full simulator resumes.

Learning-rate schedules are described by official SB3 examples and the original PPO implementation literature; those sources motivate the experiment, not its success. Actual execution remains bound to the pinned source, not current online versions.

No production trainer/default, physical task/reward/noise/reset, protected MuJoCo/control source, master, deployment or held-out-training-seed change. No production PR merge or issue closure. Decay is beneficial in these measured runs, but is not promoted as a qualified reliability fix.