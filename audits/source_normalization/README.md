# Source-normalization experiment — published, execution pending

This study follows the outward-mixture trade-off attribution in #35. It is not a report of controller performance.

## Registered comparison

The protocol was posted before dispatch: issue #35 comment **5746872084**. The training source is pinned to **a799c6d35c3753266fa382230d734e2d7de18312**; workflow run **35482824488**. At this status note, GitHub reports the native preflight as **queued**, with no executed steps. No new Rust compilation, preflight pass, training completion, or controller score is claimed.

The three arms use the exact same executed outward-start mixture, target construction, optimizer cadence and initial complete learner state. The groups are first768 retained nominal-origin samples and final256 outward-origin samples, not labels for every observed state.

| Arm | Centring | Scale |
|---|---|---|
| global | Whole1024-sample mean | Whole1024-sample standard deviation |
| source-center | Separate means for768/256 groups | Same global standard deviation |
| source-standardize | Separate means for768/256 groups | Separate group standard deviations |

All moments use the inherited sequential-f32 population formula and1e-6 standard-deviation floor. Only actor advantages change. Raw return targets, critic loss, primary and supplemental sampling, nonlinear plant, reward, noise, exploration, network architecture and optimizer settings remain unchanged. There is no teacher, imitation penalty, gradient projection, adaptive mixture or evaluation-selected update.

The predeclared candidate is source-standardize; source-center is a mechanistic control, not a post-hoc alternative winner. Each of eight exposed histories41001–41008 is replayed through4096 prior updates; three full-state descendants then receive512 additional updates. Checkpoints0/32/128/512 are fixed. Global control weights must reproduce every historical half-outward checkpoint exactly. A replay mismatch stops and is retained.

Fresh evaluation draws use training_seed+0x04000000, the same existing six panels, first-failure stopping, explicit completion denominators and final-ten-second centring. This is development on exposed histories, not fresh held-out qualification. The twelve known seed41008 angle-failure cases are separately labelled historical witness tests, never new acceptance data.

Advancement requires resolved long-outward completion AND discounted-return gains over global normalization under the registered paired-history99% multiplicity-adjusted intervals, plus nominal survival retention versus both control and incoming policy and at least61/64 nominal successes for each history. The original stricter absolute screen is separately reported. No normalization formula or average-score gain overrides a failed controller screen.

## Accounting and verification

Each continuation arm/history uses262144 main interactions,2359296 supplemental interactions and at most1048576 sampled-tail interactions. Each network performs8192 Adam steps and2097152 gradient-sample visits. Prefix replay, evaluation and diagnostic witness costs are additional. Equal main counters do not imply sample efficiency versus ordinary PPO.

The workflow requires strict Clippy, all inherited native/unit/integration controls, normalization mathematics and zero-variance tests, scope restoration, unchanged first critic updates, and ordinary/global replay with persistent Adam/RNG history. These are requirements, not results while preflight remains queued. Audit preparation adds a reversible advantage-only hook; protected production sources are restored after building.

A standalone Python verifier has passed **59 synthetic/unit tests** locally, using Python3.13.5,NumPy2.3.5,SciPy1.17.0. It checks declared identities, missing/corrupt evidence, fixed statistical rules, normalization formula/isolation, GAE, retained critic targets and numerical snapshots. This does not compile Rust or establish any learning result. It explicitly emits an incomplete result when any of the nine required raw archives is missing. Selected real trajectories/optimizer outputs can only be verified after artifacts exist.

Do not dispatch a duplicate run merely because this one is queued. Do not retry a completed measurement for a better score. No production default, PR38, master or deployed asset has been modified by this study. Passing execution checks would still not imply passing controller qualification.