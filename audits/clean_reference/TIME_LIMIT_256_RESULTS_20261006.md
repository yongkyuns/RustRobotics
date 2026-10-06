# Timeout-256 coverage intervention: reset coverage is fixed, sustained control is not

October 6, 2026. Exposed development seeds 201-204. This is a negative development result, not a production/default change.

## Result

Replacing the 5,000-step training episode cap with a 256-step external time limit guarantees repeated sampling from the existing task reset distribution while retaining ordinary SB3 timeout bootstrapping. It materially improves early learning and eliminates reset-free batches, but it does **not** make long PPO training reliable.

At the fixed 1,048,576-interaction endpoint, pooled completion changes from 70/128 to 62/128 deterministic and 137/256 to 107/256 stochastic. The registered absolute/retention screen fails. Effects are strongly heterogeneous: seed 203 becomes excellent, while seeds 202 and 204 collapse to rail failures.

Every endpoint failure under the timeout-256 candidate is a **position-limit** failure; there are zero pole-angle failures on the registered short panels. Thus the intervention successfully changes recovery/reset coverage but leaves a separate centering/long-horizon failure.

## Fixed intervention

Control is the immutable stock-SB3 1x2048 result from run 37519334101. Candidate changes only native training max_steps 5000 -> 256. Both use one environment, 2,048 rows/update, 512 updates, 1,048,576 training interactions, minibatch64, ten epochs, tanh64x64 actor/critic, learned Gaussian std, Adam epsilon1e-5/lr3e-4, gamma.99/lambda.95, clip.2 and max-grad-norm.5. Reward, plant, observation/action noise, reset law and physical failure limits are unchanged.

Candidate initial-policy digests equal their historical controls. The local exact-runtime control reproduced every original actor tensor at checkpoints0/65536/262144 for all four seeds before duplicate control execution was stopped. The qualified native ZIP SHA256 is 58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0.

An independent timeout control verifies native max_steps produces terminated=false/truncated=true, and stock SB3 stores reward + gamma*V(terminal_observation) at the timeout before GAE. The artificial boundary is therefore not treated as a task terminal.

## Coverage

| Seed | training episodes | true failures | timeout truncations | min starts/update | mean starts/update | zero-start updates |
|---|---:|---:|---:|---:|---:|---:|
|201|4295|421|3874|8|8.391|0|
|202|4301|498|3803|8|8.402|0|
|203|4292|445|3847|8|8.385|0|
|204|4287|348|3939|8|8.375|0|

This is qualitatively different from the prior 8x256 stream experiment, where 853/2048 updates had no episode-start observation because live streams continued across rollout boundaries.

## Short evaluation

| interactions | control det /128 | timeout256 det /128 | control stoch /256 | timeout256 stoch /256 |
|---:|---:|---:|---:|---:|
|65,536|116|125|228|237|
|262,144|109|113|215|221|
|1,048,576|70|62|137|107|

Endpoint per seed:

| seed | deterministic control -> candidate /32 | stochastic control -> candidate /64 | stochastic common-gamma discounted-return change |
|---|---:|---:|---:|
|201|20 -> 21|33 -> 36|+6.946060|
|202|14 -> 1|38 -> 0|-1.076538|
|203|11 -> 32|20 -> 63|+18.238010|
|204|25 -> 8|46 -> 8|-0.326232|

Mean training-seed stochastic discounted-return effect is **+5.945325**, even though pooled completion gets worse. The candidate endpoint failure counts are position-only: deterministic 11/31/0/24 and stochastic 28/64/1/56 for seeds201/202/203/204. There are zero angle-limit endings.

## Interpretation

This rejects the hypothesis that merely guaranteeing ordinary reset/recovery samples is sufficient. It also narrows the failure. The learner can improve its configured discounted score while losing multi-second rail containment.

At dt=.01s and gamma=.99, reward five seconds ahead has weight approximately .00657 and reward ten seconds ahead approximately .0000432. The terminal and future consequences of slow rail drift can therefore have almost no influence on the objective by the time the evaluation failure occurs. This is consistent with the observed position-only endpoint failures and positive mean discounted-return change despite worse survival; it is not by itself proof that gamma is the only remaining cause.

The next registered experiment (issue #35 comment 6027141679) keeps timeout256 fixed and tests a physical-time-horizon PPO setting, gamma=.999/lambda=1, against this reference. No timeout horizon sweep is authorized from this negative result.

## Scope

The necessary short screen failed, so this result cannot advance to held-out qualification or production. Long/outward panels are not used to rescue the failed candidate. No production/default/master/protected-source/deployment change or merge is made.
