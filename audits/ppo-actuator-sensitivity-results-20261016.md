# PR #41 / issue #42 — actuator sensitivity: completed experimental evidence

**Verified at** 2026-10-16 UTC. **Result: full diagnostic execution PASS;
hardware/production control qualification NOT established.**

- Source SHA: `1a2433a77ec4dca9392b0d17832cb1e29555bb88`.
- [Successful actuator workflow #38899322793](https://github.com/yongkyuns/RustRobotics/actions/runs/38899322793)
- [Full artifact #12045368141](https://github.com/yongkyuns/RustRobotics/actions/runs/38899322793/artifacts/12045368141)
- [Native validation #38899329913](https://github.com/yongkyuns/RustRobotics/actions/runs/38899329913), green.
- Data: **4096 CSV episode rows**, **5026 CSV per-tick witness rows**. All groups, key cardinalities, baseline outcomes and per-episode trace lengths passed the workflow verifier.
- Rechecked SHA256: episodes.csv `2b57a28badcebb7d6e6263781eefedd96f5f42ce5fe0882a20b95215a92f9be5`; witness-traces.csv `ee3da78a27c9647b9114e431a446123b6519eb6f1097e5a30cb5b18342c3ce4a`.

## Frozen experiment

Four **previously exposed development training seeds** 201–204, all newly
retrained for 976 updates / 499,712 training transitions each, one continuous
Adam learning clock per seed; no evaluation data enter optimization.

Only the inference-time actuator changes: ideal instantaneous ±20 N, 10-ms
one-tick command delay (initially zero), 20-ms two-tick force hold, and
5 N per 10-ms tick finite slew (initially zero). Physical dynamics, noise
streams, reset keys, measured observation, reward function, policy weights,
1500-tick horizon and success criteria remain fixed. Both deterministic
and stochastic policies are tested at every actuator condition, on common
64-key ordinary and 64-key difficult reset panels. Per-key random streams
are seeded identically for paired comparisons, but the diverging policy
trajectories can draw different actions; the runs are **correlated**.

The original nominal Rust evaluator was left unchanged. New Rust tests
assert exact episode equality between it and the trace-enabled ideal
evaluator at fixed keys in both inference modes. GitHub CI also checks that
all final *nominal* 4-seed panel counts reproduce the previous experiment.

## Survival results

15 seconds = 1500 × 0.01s, completed episodes out of 512 on the common
difficult panel (4 train seeds × 2 inference modes × 64 reset keys):

| Actuator | Completed | Rate | Failures | Unique failed reset keys | Paired losses vs ideal |
|:--|--:|--:|--:|--:|--:|
| Ideal | 505/512 | 98.63% | 7 | 1 | baseline |
| 10-ms delay | 504/512 | 98.44% | 8 | 1 | 1, no gains |
| 20-ms hold | 505/512 | 98.63% | 7 | 1 | 0 losses/gains |
| 5-N/tick slew | 480/512 | 93.75% | 32 | 8 | 25, no gains |

Every actuator condition completed **512/512 common ordinary episodes**.
On the difficult panel, the slew-limited condition gave 247/256
deterministic and 233/256 stochastic completions. The additional seven
failing reset keys under slew are `4100100015`, `4100100170`,
`4100100249`, `4100100273`, `4100100474`, `4100100556`,
and `4100100557`. These are *not* independent random outcomes:
the same keys repeat across train seeds/modes.

Original ideal, one-tick delay and hold failures all crossed the **rail**
boundary. Under slew, 29/32 failures crossed the rail; 3 failed with
nonnegative rail clearance (angle-limit terminations under the
environment's documented two-boundary termination rule). Only the
frozen witness key below has per-tick true-state records, so detailed
post-terminal angle trace evidence is limited to that witness.

### Fixed witness: reset key 4100100100

Original true state (SI units): x = **+0.123918 m**,
x_dot = **+0.397964 m/s**, theta = **−0.232578 rad**,
omega = **−0.433484 rad/s**. Every evaluated policy receives only its
noisy observation, not this privileged state.

All failures in this fixed witness were **rail-only** at their last
step (the pole angle was inside its 0.6-rad limit). Specifically:

| Seed-204 deterministic policy | Steps survived | Terminal result | Minimum rail clearance |
|:--|--:|:--|--:|
| Ideal | 1500 | Completed | +0.006070 m |
| Hold 20 ms | 1500 | Completed | +0.013239 m |
| Delay 10 ms | 68 | Rail crossing | −0.002867 m |
| Slew 5 N/tick | 57 | Rail crossing | −0.017782 m |

The 15-second ideal survivor came **within ~6 mm of the 2.4-m rail**
(max true x 2.3939297 m) before recovering. Under delayed actuation
the same policy crossed the rail in 0.68 s; slew-limited control crossed
in 0.57 s. This is stronger evidence of a **narrow recovery margin**
than the aggregate 98% figure alone.

Seed-204 deterministic surviving ideal control issued **1192 full-scale
sign reversals** in 1499 adjacent tick pairs (~79.5%, about 79.5/s).
The surviving two-tick-hold trace issued 564 output force reversals
in 1499 transitions (~37.6%, about 37.6/s); this case does not require
the original ~80/s reversals for 15-second survival, but its cart
also ran within 13 mm of the rail. The ideal policy commanded ±20 N,
with no measured actuator bandwidth or power limit.

### Why raw return is not a suitable cross-actuator success metric

The configured physical reward penalizes `0.001 × actuator_command_N²`
per step. A ±20 N ideal force therefore incurs a 0.4-step effort cost;
slew-limited, often lower-magnitude forces incur less of that cost.
Across common difficult evaluations, the slew condition had a **higher
mean raw return (~1095 vs ~800)** while it had **25 additional failures**
and shorter mean survival. Comparing raw reward across actuation modes
would be misleading; compare fixed-horizon completion, physical margin,
failure type and actuator effort separately.

## Evidence limitations and disposition

- These are **not fresh confirmation training seeds**; both training and
  evaluation reset keys were already used for experimental development.
- A 10-ms delay, 20-ms hold or 5 N/tick slew is **only a diagnostic stress
  model**. No physical motor data justify treating these values as
  representative of a real platform.
- The evidence does **not** isolate categorical-vs-continuous PPO. Both
  action space and optimizer/rollout recipe differ from the current
  production continuous Gaussian PPO. Issue #42 still requires a
  controlled action-space causal ablation on matched physics/data/seed
  budgets.
- No production changes, merges, global stability, real hardware safety,
  or independent #33 acceptance are justified. **Keep PR #41 draft.**

## Next bounded work

Freeze a matched-optimizer/action-space test without tweaking against
reset key 4100100100; separately quantify viable recovery under a
predeclared classical bounded controller as a diagnostic only, not a
privileged actor or a pretrained PPO initialization. Run fresh training-
seed confirmation for #33 only after the continuous PPO candidate and
acceptance thresholds are frozen.
