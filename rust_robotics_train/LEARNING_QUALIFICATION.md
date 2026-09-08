# PPO short-learning qualification — protocol v1

This is the remaining learning-evidence increment of issue #5 after bounded
objectives (#29), rollout boundaries (#30) and session-owned randomness (#31).
It tests the existing implementation; production training code is not modified.

## Frozen experiment

The protocol was recorded in [issue #5](https://github.com/yongkyuns/RustRobotics/issues/5#issuecomment-5585230895)
before inspecting exploratory or confirmation results. The complete recipe is
pinned in `tests/ppo_learning.rs`; a normal test checks that it still equals the
production defaults at #31 (`1970fc1`). A future default change must not silently
change the experiment.

| Item | Fixed value |
| --- | --- |
| Independent training trials | 12, seeds 101 through 112 |
| Training budget per trial | 128 updates, 65,536 environment transitions |
| PPO | 512-step rollouts, batch 128, 4 epochs, learning rate 0.0003 |
| Discount / GAE / clipping | 0.99 / 0.95 / 0.2 |
| Critic / entropy coefficients | 0.5 / 0 |
| Network / pre-squash exploration | Two 64-unit ReLU hidden layers / 2 N |
| Training environment | Unchanged default linear pendulum, dt 0.01 s, 20 N action bound |
| Reset, reward and noise | All unchanged defaults; observation/action noise and disturbances enabled |
| Training time limit | 5,000 steps; existing time-limit bootstrap semantics |
| Evaluation | 32 independent episodes per initial/final policy, capped at 1,000 steps |
| Evaluation action | Deterministic portable `PolicySnapshot::act`; no exploratory action samples |

For training seed `s`, evaluation seeds are
`1_000_000 + (s - 101) * 1_000 + episode_index`, with episode indices 0 through 31.
Each episode starts with a new environment and explicit RNG. The initial and final
policy get the same random conditions within each pair. Because episode RNGs are
independent, early termination cannot shift later pairs. Evaluation never consumes
the trainer's environment or RNG state. It uses the exported snapshot, the same
inference path used by the simulator, not privileged noiseless state or the critic.

The only evaluation environment change is the external 1,000-step cap. Rewards
are accumulated in f64, undiscounted, including the terminal penalty exactly once.
No partial episode is dropped, no failure is reset within its evaluation episode,
and no post-training checkpoint is selected by evaluation score. The final actor
is the one after exactly 128 updates. Both actor/critic parameters and training
metrics are checked for finiteness after every update; evaluation validates shapes,
all raw parameters, actions, observations, physical states, rewards and endings.

## Acceptance and uncertainty

For each training seed, calculate its final-minus-initial mean episode return.
The **training run** is the statistical unit: n = 12, not 384 episode pairs.
Require both:

1. At least **10 of 12 gains strictly greater than 25 return points**.
2. Overall mean of the 12 gains **at least 50 return points**.

Ties at 25 count as failures. Under the null that a trial exceeds the 25-point
margin with probability at most 1/2, the one-sided conservative binomial tail is
`sum(comb(12, k), k=wins..12) / 4096`. For 10 wins, this is
`79 / 4096 = 0.019287109375`. No normality assumption on the gains is required
for this direction/margin test. One exceptional training seed cannot make it pass.

Also report every trial, median gain and the order-statistic interval
`[third-smallest, tenth-smallest]`. Its nominal central coverage for the median
of continuous iid trial gains is `1 - 2*79/4096 = 96.142578125%`. Random-seed and
independence assumptions remain experimental assumptions. The interval and test
refer to the distribution of **finite-panel trial gains**, including evaluation
noise, not exact infinite-episode expected return. The magnitude of the mean is
a separate engineering threshold, not an unqualified confidence interval.

References: [NIST sign test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm),
[Agarwal et al., statistical uncertainty in RL evaluation](https://arxiv.org/abs/2108.13264),
and [Patterson et al., empirical design in reinforcement learning](https://www.jmlr.org/beta/papers/v25/23-0183.html).

## Exploration is not confirmation

The temporary probe used training seeds 0, 1 and 2, with independent evaluation
seeds `60_000 + training_seed * 100 + episode_index`. All are excluded from the
confirmation cohort. It reported checkpoints 0/32/64/128 but the confirmation
budget and decision rule were already fixed before these outputs were inspected.
No default hyperparameter, reward or noise was adjusted.

| Exploratory seed | Initial mean return | Return after 128 updates |
| --- | ---: | ---: |
| 0 | 45.4467 | 107.2494 |
| 1 | 55.3749 | 143.0315 |
| 2 | 44.4162 | 118.9642 |

[Exploratory run 34226939664](https://github.com/yongkyuns/RustRobotics/actions/runs/34226939664)
ran source `c74cefc8e5b232f1ea921785fde440b27c6bacf7`. Artifact `10056283773`
contains all 384 exploratory episode records and provenance. ZIP SHA256:
`5d960871dc0f637c01075df05655bae864b0f80673ac474974a509b83b0da405`.
These three exploratory gains are not the acceptance result.

## Initial Linux confirmation — passed without protocol adjustment

[Qualification run 34228027590](https://github.com/yongkyuns/RustRobotics/actions/runs/34228027590)
ran formatted test source at `f0b4d99c0946fb35ed08af8c2f50ad8748f4b833`.
All 12 confirmation seeds exceeded the 25-point margin. Mean return increased
from **55.603606 to 127.816198**, a gain of **72.212593**. Median gain was
**73.472756**, with nominal 96.142578125% median interval
**[60.876861, 78.684983]**. The one-sided margin-test probability was
**1/4096 = 0.000244140625**. The complete second execution replayed all numeric
records and actor bytes exactly; it is not another independent cohort.

| Training seed | Initial mean return | Final mean return | Gain |
| --- | ---: | ---: | ---: |
| 101 | 63.989831 | 128.510680 | 64.520849 |
| 102 | 41.224085 | 132.203891 | 90.979806 |
| 103 | 70.287631 | 143.408106 | 73.120475 |
| 104 | 58.115070 | 118.991932 | 60.876861 |
| 105 | 40.149255 | 121.399161 | 81.249906 |
| 106 | 58.422439 | 132.247478 | 73.825038 |
| 107 | 60.462263 | 120.834642 | 60.372379 |
| 108 | 53.135535 | 131.820517 | 78.684983 |
| 109 | 58.078317 | 129.232856 | 71.154539 |
| 110 | 60.366210 | 118.197170 | 57.830960 |
| 111 | 52.719071 | 129.205220 | 76.486149 |
| 112 | 50.293563 | 127.742728 | 77.449165 |

**This is improvement, not solved balancing.** Mean episode length increased
from 80.666667 to 175.130208 steps (about 0.81 to 1.75 seconds). All 384 final
policy evaluation episodes still terminated before the 1,000-step/10-second cap;
the longest lasted 529 steps. Sustained balancing remains unqualified.

All 9 evaluator controls and 12 Python validator tests passed. Four injected
evaluator/statistics defects (ignoring the policy, accepting nine good seeds,
double-counting rewards, excluding the observed binomial count) each reached the
intended named runtime assertion. Restored training-crate tests and strict Clippy
passed. No seeds, thresholds, trial counts, numerical code or assertions were
changed after execution to obtain a pass. Only formatting was applied beforehand.

Artifact `10056676921` contains 14 command logs, structured results, 768 episode
records and 24 actors per confirmation execution, plus qualified source files.
Its downloaded SHA256 and all archived source hashes were independently checked:
`a31a6415d2497d42b7b563238715e96c2b09264c168d2055429b8491563eb4d7`.
The executed test blob is `59fd74a4d18bd7ed81462e02947498e39363632d`; the independent
runner blob is `6273cdf4175f86494b2ab7973ea145641e4f5372`.
Temporary probe/preparation files are removed from the submitted change.

## Execution and evidence

```sh
# Fast controls run as part of ordinary workspace tests (9 tests).
cargo test --locked -p rust_robotics_train --test ppo_learning
# Independent Python evidence validation controls (12 tests).
python3 -m unittest discover -s scripts -p 'test_qualify_ppo_learning.py'
# Explicit release-mode endpoint, independent arithmetic and exact replay.
python3 scripts/qualify_ppo_learning.py --output target/ppo-learning-evidence
```

The expensive endpoint has an explicit `#[ignore]` annotation so the four ordinary
platform jobs do not accidentally run it in debug mode. The permanent **PPO
learning** workflow executes it by exact name with `--ignored`, requires one
completed runtime test, and checks 12 trials, 768 episode records, 24 actors and
exact budgets. Python independently recomputes every mean, margin count, threshold,
binomial probability and interval instead of trusting Rust's success flag.
Missing/duplicate records, unexpected seeds, malformed policies, nonfinite values,
impossible reward sums and contradictory outcomes are rejected.

Raw actor evidence encodes big-endian f32 words in this fixed order: action limit,
action scale, input weights/bias, hidden weights/bias, output weights/bias. The
shape is fixed at 4→64→64→1 (4,547 floats including metadata). SHA256 values refer
to these bytes. This is fixture evidence, not a new production snapshot format.

The complete endpoint runs twice in the same numerical build. Equality checks
all numeric records and actor bytes; the repeated run is **not** counted as 12
additional trials or pooled to obtain a smaller p-value. Logs, parsed results,
source hashes, lockfile, source/tree revisions and compiler details are retained,
including when the scientific criterion fails. Failure never authorizes seed
exclusion, threshold changes, checkpoint selection or reruns until a pass.

Controls include independent enumeration of all 4,096 Bernoulli sequences,
known order statistics, ties/missing/nonfinite/outlier cases, supplied-policy
execution, terminal/time-limit accounting, per-episode RNG isolation and an actual
four-update **zero-learning-rate negative control**. That control executes 2,048
training transitions but preserves actor/critic weights and paired scores;
synthetic zero gains then check the gate, not additional independent training runs.

## Limits

This is a bounded regression fixture for short learning on the existing noisy
**linear** pendulum model. It does not prove global convergence, optimal control,
nonlinear or hardware robustness, successful swing-up, learning on other tasks,
or the simulator's multi-replica scheduling/aggregation. It does not qualify
seeded browser training merely because the web bundle builds. Observation noise
still makes this feed-forward policy's input partially observed.

Numerical replay is within a fixed backend/build and locked dependency set, not a
cross-architecture/compiler/version bitwise guarantee. Weight-transfer snapshots
still omit optimizer moments, RNG cursors, environment and partial-episode state;
this work does not provide exact-resume training checkpoints or a speedup claim.
