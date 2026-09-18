# C4 — epsilon-only from-scratch PPO comparison

Date: 2026-09-18. Learning cases: #35 / #33. Separate integration fix: #36 / PR #37.

## Conclusion

**All 16 full-budget training runs completed. Adam epsilon materially changes these results, but epsilon alone is not a reliable correction for the balancing failure. Production epsilon remains unchanged.**

In Rust, changing epsilon from 1e-5 to 1e-8 improves final deterministic return on three exposed seeds and worsens one. Stochastic discounted return improves on two and worsens two. In SB3, the smaller epsilon increases pooled completions but makes final stochastic discounted return worse on all four seeds. No arm reliably completes the evaluation horizons.

The earlier C3 claim that recipe-matched SB3 beats Rust on every seed does not survive genuine epsilon matching: at 1e-5, SB3 has better final deterministic return on two of four seeds; at 1e-8, it has better return on only one of four. This does not prove implementation equivalence or establish a unique cause of long-run failure.

## Protocol and provenance

The [protocol was recorded before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5734427307).

Production baseline: `d3f59f9bf38d4038ba7c5008e3a3b41c15e99835`. Actual experiment: `b1100474d512f45856967c9c3b78be0fc6ee9e1b`, [run 35381533937](https://github.com/yongkyuns/RustRobotics/actions/runs/35381533937). All 17 jobs succeeded: the validated native/reference preflight plus the 16 measurements. No measurement was retried to improve a score, and no trained seed or checkpoint was dropped.

Four arms, each on exposed development seeds 201–204: Rust/Burn epsilon1e-5; Rust/Burn epsilon1e-8; SB3 epsilon1e-5; SB3 epsilon1e-8. Every run starts from random initialization and uses exactly 1,048,576 actual environment transitions. Checkpoints are 0, 65,536, 262,144 and 1,048,576, not selected best checkpoints.

Within each implementation, epsilon is the only changed recipe setting. All arms use one environment, rollout512, batch128, four epochs, 2x64 ReLU, identical initial actor/critic weights per seed, fixed latent std0.1 and force20*tanh(z), gamma0.99, lambda0.95, clip0.2, learning rate0.0003, value coefficient0.5 and entropy coefficient0. No gradient clipping, learning-rate annealing, target-KL stopping, reward changes, observation normalization, supplied controller or critic pretraining is added. Adam beta1/beta2 are 0.9/0.999; runtime optimizer settings are exported, not inferred from another library.

The Rust audit constructor selects epsilon only before the first update. Both optimizers are fresh at construction and persist thereafter. Existing training code remains unchanged, with temporary ABI/constructor hooks in a disposable checkout. The explicit default constructor is checked against the ordinary session, including subsequent rollouts and grouped/split updates.

SB3 uses its actual unmodified collector, GAE and PPO.train, with the previously documented C3 whole-rollout normalization and latent-action adapter. It calls the actual Rust plant through the checked native ABI, not Python dynamics. There is no fixed-tensor playback or artificial ReLU gate intervention in these learning runs.

Pinned execution: Rust1.98.1, Burn0.20.1 and committed Cargo.lock; Python3.11, Torch2.8.0+cpu, NumPy2.2.6, Gymnasium1.3.0, SB3 2.9.0. Across implementations or CI hosts, online RNG draws and floating reductions are not required to agree bit-for-bit. Timings are retained but are not a controlled speed benchmark.

## Evaluation

The unchanged C3 evaluator uses the same explicitly seeded, independent-from-training reset/noise/action-innovation panels for all arms at every checkpoint. Each training seed receives 32 deterministic-policy episodes capped at 1000 steps (10 seconds), and 64 stochastic-policy episodes capped at 1500 steps (15 seconds). The evaluator preserves training random generators, models and state; no score selects, rejects or modifies an update.

These evaluation panels and four training seeds have already been exposed. They are not held-out acceptance. A pooled count of 128 or 256 episodes is not 128 or 256 independent trained agents. The predeclared co-primary descriptive metrics are final deterministic mean return and stochastic mean gamma-discounted return. Their uncertainty uses paired training-seed differences, n=4, with pointwise 99% Student intervals; those are neither simultaneous guarantees nor equivalence tests.

## Final results at 1,048,576 transitions

Higher return is better. Completion means reaching the specified cap without true termination.

| Arm | Deterministic 10s complete /128 | Deterministic mean return | Stochastic 15s complete /256 | Stochastic discounted mean return |
|---|---:|---:|---:|---:|
| Rust epsilon1e-5, current default | 50 | 489.949 | 98 | 77.331 |
| Rust epsilon1e-8 | 74 | 620.293 | 150 | 80.054 |
| SB3 epsilon1e-5, correctly matched default | 68 | 601.525 | 123 | 81.121 |
| SB3 epsilon1e-8, old C3 setting | 88 | 673.407 | 142 | 78.135 |

### Every final training-seed outcome

| Arm | Seed | Deterministic complete /32 | Deterministic mean return | Stochastic complete /64 | Stochastic discounted mean |
|---|---:|---:|---:|---:|---:|
| Rust1e-5 |201|7|450.151|14|84.003|
| Rust1e-5 |202|4|273.431|6|70.833|
| Rust1e-5 |203|21|656.259|39|79.296|
| Rust1e-5 |204|18|579.955|39|75.193|
| Rust1e-8 |201|5|358.309|19|81.466|
| Rust1e-8 |202|17|551.569|36|78.947|
| Rust1e-8 |203|25|746.192|47|78.696|
| Rust1e-8 |204|27|825.104|48|81.106|
| SB3 1e-5 |201|19|609.489|39|83.758|
| SB3 1e-5 |202|19|606.102|39|80.614|
| SB3 1e-5 |203|17|623.821|32|79.232|
| SB3 1e-5 |204|13|566.686|13|80.880|
| SB3 1e-8 |201|19|627.430|28|81.714|
| SB3 1e-8 |202|19|546.573|24|79.122|
| SB3 1e-8 |203|23|706.794|43|71.727|
| SB3 1e-8 |204|27|812.829|47|79.977|

### Epsilon effects, 1e-8 minus 1e-5

| Implementation / metric | Mean paired change | Pointwise 99% training-seed interval | Improved seeds |
|---|---:|---|---:|
| Rust deterministic return | +130.344 | [-364.189, +624.878] | 3/4 |
| Rust stochastic discounted return | +2.723 | [-12.165, +17.610] | 2/4 |
| SB3 deterministic return | +71.882 | [-307.659, +451.423] | 3/4 |
| SB3 stochastic discounted return | -2.986 | [-11.890, +5.917] | 0/4 |

All four predeclared final intervals cross zero. This is imprecise evidence from four reused training seeds, not proof that epsilon has no effect. Conversely, the pooled completion improvements are not sufficient evidence of a robust fix.

Rust seed201 illustrates the conflict: deterministic mean return falls from450.151 to358.309 and discounted stochastic return from84.003 to81.466, despite a higher stochastic completion count. SB3's smaller epsilon raises pooled stochastic completions from123 to142 while lowering discounted return for every training seed. Completion and discounted reward do not rank these policies identically.

### Earlier checkpoints retained

| Arm | 10s complete at65,536 /128 | At262,144 /128 | At1,048,576 /128 | Stochastic discounted mean at65,536 /262,144 /1,048,576 |
|---|---:|---:|---:|---|
| Rust1e-5 |0|0|50|64.143 /70.296 /77.331|
| Rust1e-8 |0|1|74|66.068 /71.457 /80.054|
| SB3 1e-5 |0|10|68|64.049 /71.749 /81.121|
| SB3 1e-8 |0|11|88|66.210 /73.626 /78.135|

The complete per-seed checkpoint table and contrasts are retained in summary.json. At65,536, smaller epsilon worsens Rust seed201 deterministic and stochastic scores. At262,144 it worsens Rust seed203 on both scores. SB3's smaller epsilon improves early/middle deterministic return on all four seeds, but final seed202 is worse. These reversals are not hidden by reporting only favorable checkpoints.

### Failure types and cross-implementation comparison

Rust1e-5's final deterministic failures are70 angle-only,2 combined angle/position and6 position-only. At1e-8 they are27 angle-only,1 combined and26 position-only. Under stochastic evaluation, angle-only failures fall from145 to49 but position-only failures rise from11 to57. This describes changed failure modes, not proof of their mechanism or a new remedy.

At matched epsilon1e-5, SB3's deterministic return advantage over Rust is +159.339/+332.671/-32.437/-13.269 for seeds201/202/203/204. The mean is+111.576 with a wide99% interval[-387.159,+610.311]. At1e-8 the respective differences are+269.121/-4.996/-39.398/-12.274; mean+53.113, interval[-369.664,+475.891]. Neither comparison establishes statistical equivalence or an implementation defect.

## Validation and reproducibility

Preflight passed strict Clippy, 73 native unit/ABI/epsilon controls, 7 balancing-diagnostic and 9 learning-integration controls. The two separately ignored heavy integration tests were not executed by that ordinary test command. Original C3 reference checks plus epsilon-pair replay, optimizer-history and evaluation-isolation controls passed. Production-source restoration was verified.

Independent offline verification checked all17 main ZIP digests and657 manifest-listed members, plus204 members in nested original C3 archives. It validated6144 unique evaluation records, fixed budgets, explicit panel keys, physical final-state termination labels, reward upper bounds, per-epoch/update/episode counts, and128 saved actor/critic networks against their portable exports. All arms have identical stored random initial actor/critic weights per seed and identical recipe configurations except the intended epsilon and identifying labels.

**Unchanged-arm replay is exact:** every actor/critic checkpoint in Rust1e-5 and every policy state tensor in SB3 1e-8 matches the corresponding original C3 artifact byte-for-byte, across all four seeds and all four checkpoints. Every old-arm evaluation outcome also reproduces its C3 record exactly. The eight repeated arm/seed runs are replays, not eight newly independent trials.

Initial evaluation scores are not universally bitwise identical across all new CI jobs despite identical stored weights and panel keys. Rust1e-8/seed203 and SB3 1e-5/seed204 have small differences; maximum episode-return differences are9.686e-6 and2.083e-5 respectively, with identical durations and endings. Maximum state-component difference is2.146e-6. Their cause was not separately attributed. These outcomes are retained, not asserted to be identical or used to rerun a seed. The offline analyzer was corrected to report this comparison rather than impose an unpromised cross-host exact-score condition; the original training criteria and numerical controls were unchanged.

The standalone analyzer has10 passing controls, including corrupted-member detection, missing/duplicate records, panel mismatch, false completion and paired-statistic checks. It recomputes summary statistics from archived outcomes. It does not independently re-simulate physical trajectories, reconstruct unexported rewards/gradients, or validate full optimizer state through checkpoint exports. No cross-platform bitwise claim is made.

## Costs and retained setup failures

Main measurements used16,777,216 training transitions and2,144,990 evaluation transitions, with524,288 paired actor/critic minibatch transactions and67,108,864 gradient-sample visits per network. Preflight/test/setup work is excluded from these totals and retained in logs. No speed claim is made from heterogeneous CI hosts.

Run35380740707 failed a strict Clippy items-after-test-module check in the temporary ABI. Moving that audit function before the existing test module repaired layout only. No main measurement started. Run35381070680 passed preflight but all16 measurement jobs stopped before training because artifact upload omitted the manifest-listed hidden workflow path. Moving that provenance copy to a visible artifact path repaired packaging without weakening hash checks. Both failed preparations are retained; they are not failed learned policies or favorable-seed retries.

The self-contained conversation bundle `rustrobotics-ppo-epsilon-evidence.zip` contains all17 unchanged main archives, nested original C3 evidence, setup-failure archives, the independent analyzer/tests and the full per-seed summaries. Artifact IDs/digests are in artifacts.json. It runs offline without loading the archived native library:

```sh
python analysis/analyze.py
python -m unittest discover -s analysis -p 'test_*.py' -v
```

NumPy and SciPy are required for offline analysis. Actual runtime sources, compiled library and dependency versions needed for new execution are retained in the native archive. Do not treat a previous sandbox path as persistent storage.

## Decision and separate PR37 status

Keep production epsilon1e-5 unchanged: the study does not establish an epsilon-only reliable balancing fix. #35 and #33 remain open; no held-out qualification, supplied controller, production default change or artificial gate operation was introduced. The next learning-level investigation should use the retained failure modes and objective/completion disagreement rather than treating cross-library bitwise agreement as the acceptance criterion.

Separately, exact PR37 head `d3f59f9` now passes **all four workflows**: dependency audit35378873579, configured PPO-learning35378873566, numerical invariants35378873559 and Rust35378873556, including the previously pending Windows tests. This result is for the integration/security fix, not held-out sustained balancing. No PR merge or production-default change was performed as part of this experiment.
