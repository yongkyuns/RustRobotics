# PPO exploration-scale factorial (October 15, 2026)

Issue #35 protocol: [6201344116](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6201344116). Continues the complete 12-run four-versus-ten-epoch audit [5815f1f](https://github.com/yongkyuns/RustRobotics/commit/5815f1f8221ee752192352b20c44108fb3032f51).

**Diagnostic complete; not a trained correction, deployment candidate, or held-out qualification.**

## What changed

At the fixed 1,310,720-interaction checkpoint, cross the complete final actor mean network **and its own frozen VecNormalize statistics** from DECAY4 (M4) or DECAY10 (M10), with the learned scalar log standard deviation of either DECAY4 (S4) or DECAY10 (S10). The critic is irrelevant to action selection. Crossed policies are inference-only counterfactuals, not original trained checkpoint products.

All four exposed development training seeds 201–204; exactly the original 64 ordinary and 64 difficult native reset cases per seed, deterministic and stochastic modes, unchanged latent standardized innovations, native disturbance/observation noise/force clipping, physical rewards/failures, 1,500-tick horizon and gamma=.999 return. No new training interactions/optimizer steps or independent cases. No reward shaping, new limits, teacher policy, or best-checkpoint selection.

## Ordinary stochastic results

| Policy composition | Mean physical raw return | Completion /256 | Mean force-effort penalty |
|---|---:|---:|---:|
| M4S4 (original 4-pass) | 1424.620 | 254 | 39.409 |
| M4S10 | **1451.191** | 254 | **13.452** |
| M10S4 | 1427.031 | 254 | 38.818 |
| M10S10 (original 10-pass) | **1453.601** | 254 | **12.651** |

The original M4S4-minus-M10S10 mean raw-return difference is **-28.980447**, paired 99% Student interval across four independent training seeds **[-36.870284,-21.090610]**. Changing only S4→S10 at fixed M4 improves return by **+26.571299** [**+15.398550,+37.744049**], recovering 91.7% of that exposed-seed deficit. **25.956386** of this improvement is due to reducing the original effort-penalty total (97.7% of the recovered return). Mean non-effort return gain +0.614914 has interval [-0.600859,+1.830687]. The residual M4-versus-M10 mean at S10 is -2.409147, interval [-7.918760,+3.100465]; unresolved is NOT equivalent.

The four-pass learned latent Gaussian sigmas are **0.240099/0.230695/0.186032/0.208140** for seeds201–204, versus ten-pass **0.138813/0.098251/0.063207/0.061756**. These are normalized latent-action sigmas, not newton-force standard deviations. The force penalty is computed from the actual recorded applied force on physical nonterminal steps; original float32 reward bits are independently checked.

## Difficult recovery: sigma alone is insufficient

| Policy composition | Difficult stochastic completes /256 | Mean raw return |
|---|---:|---:|
| M4S4 | 239 | 1277.714 |
| M4S10 | **241** | **1313.606** |
| M10S4 | 246 | 1318.781 |
| M10S10 | **246** | **1344.334** |

The M4 scale swap gains exactly two previously failed seed204 difficult cases and loses none; all other individual completion identities remain. The sigma swap at M10 changes no completion outcome. The difficult raw-return effect at M4 +35.892248 has four-seed 99% interval [-23.909903,+95.694399], spanning zero. The remaining mean-network/normalizer return difference at common S10 likewise spans zero.

For deterministic actions the log sigma has **no behavioral effect**: all same-mean sigma-swapped full trajectory arrays are byte-exact. Difficult deterministic completion remains M4 **241/256**, M10 **249/256** regardless of scale. Thus the residual learned recovery-control difference is not solely exploration-scale noise.

## Verification and exact costs

- 64 planned evaluation groups / **4,096 physical trajectories**, **5,985,142 native transitions**, zero optimizer/training steps. Two additionally retained original-diagonal preflight control groups add **128 episodes / 189,161 native transitions**. Total simulator work **6,174,303 transitions**. The two controls were also included in the registered main evaluation and reproduced exactly; their extra cost is counted. No completed main group was retried.
- ALL **32 original diagonal groups and 2,048 original case traces** reproduce original saved files byte-for-byte, including ZIP SHA256 identity. All 16 deterministic sigma pair-groups are byte-identical.
- Independent NumPy reader validates all records including native float32 reward bits, actor/statistic identity, standardized action innovations, signed cart clearance, physical endings/padding, applied forces and exact stopped returns; maximum normalized-command discrepancy **4.00098870035e-7**, below inherited **3e-6** tolerance.
- **23 new scientific and genuine-record corruption tests pass.** Full extracted evidence: **966 SHA256 payloads**, all tests, all raw paths, and **RESULT.json/CASES.json/SUMMARY.json reproduced byte-for-byte**. Compact extracted evidence: **72 payloads** and applicable genuine-fixture tests; it explicitly cannot replay the omitted full raw panel.
- Current host AMD EPYC 9V74 with pinned Python3.11.14/Torch2.8.0+cpu/NumPy2.2.6/SB3 2.9.0/Gymnasium1.3.0. Original native/SB3 and saved-policy source inputs are bound to the verified parent cohort. Independent reader does not reintegrate native physics or regenerate private disturbances.

## Artifacts

- Full `ppo_sigma_factorial_evidence_20261015.zip`: **267,316,480 bytes**, SHA256 **ada29ba68153d11b831a658d34feb09ed86eefcfa5413543ca1bd44ca6c05973**.
- Compact `ppo_sigma_factorial_report_code_20261015.zip`: **6,146,080 bytes**, SHA256 **a901f45e0ce86fa6d7c551a6f7f07a7fa40d8f1ac5d6a85c2ae49b8831ae49ab**.
- Full human-readable REPORT.md, exact executed run.py, read-only analyze.py, 23 tests, genuine raw controls, original final checkpoints/statistics/selection subsets, manifests and README/REPLAY.py are included.
- The first shell launcher attempted to append to an uninitialized PID variable after starting seed201. That child finished successfully; seeds202–204 ran separately. No main numerical result was replaced or selected based on outcome.

## Interpretation and next test

The extra ordinary stochastic return cost of fewer fitting passes is primarily the higher learned **Gaussian exploration scale**, not reduced completion, on these four exposed histories. This does not establish that copying the ten-pass *final* sigma into four-pass training is a valid correction or that its early exploration would remain adequate.

A mechanism-driven NEXT training comparison should decouple the actor mean response, learned log_std and/or critic fitting budgets while keeping the same physical task, total interaction budget, positive LR floor and *original unrelaxed* ordinary-return/difficult-retention criteria. A useful candidate must retain four-pass tail robustness **and** close its stochastic effort deficit without losing early learning. No such follow-up training outcome has been measured here.

Issues #33/#35 remain open. No production Rust optimizer/default, task/reward/physics/noise/reset, protected source, master, deployed weights or production PR has been altered.