# PPO value scaling from initialization — October 7, 2026

Exposed-development experiment for #35. This is **not** a production/default qualification.

## Conclusion

Constant value/reward scaling fixes the severe critic-conditioning pathology and substantially improves learning speed and final performance, but it does **not** make continued PPO reliably preserve a learned controller.

At 1,048,576 interactions the scaled arm finishes **128/128 deterministic and 253/256 stochastic** across seeds 201–204, versus raw **122/128 and 240/256**. At 131,072 interactions scaled is already **127/128 and 248/256**, versus raw **104/128 and 172/256**.

However, scaled seed202 goes **32/32, 63/64 at 262,144 -> 30/32, 59/64 at 524,288 -> 32/32, 63/64 at 786,432**. That violates the preregistered continued-training preservation condition. The endpoint screen passes; the preservation screen fails.

## Fixed intervention

Protocol: issue #35 comment 6036912299. Setup-only RNG-ordering failure/correction: comments 6036950583 and 6036963229.

Pinned runtime: Python3.11.14 / Torch2.8.0+cpu / NumPy2.2.6 / SB3 2.9.0 / Gymnasium1.3.0 on AMD EPYC9V74. Native archive SHA256 `58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0`.

Both arms use stock SB3 PPO: one env, 2048 transitions/update, 256-step training timeout, gamma=.999, lambda=1, batch64, 10 epochs, lr3e-4, clip=.2, vf_coef=.5, ent_coef=0, global grad norm .5, Adam(.9,.999), eps1e-5, separate tanh64x64 actor/critic.

Only candidate change: with `S=1/(1-gamma)`, divide training rewards by S and divide the critic's initialized final output-head weights/bias by S. Actor/log_std and critic hidden layers are initially identical; physical initial values are equivalent. Evaluation always uses the unchanged raw native reward.

## Results

| Seed | Arm | 131k | 262k | 524k | 786k | 1049k |
|---:|---|---:|---:|---:|---:|---:|
|201|raw|30/44|32/63|30/55|32/63|32/59|
|201|scaled|32/63|32/62|32/63|32/63|32/62|
|202|raw|29/54|32/60|32/63|32/63|32/63|
|202|scaled|32/62|32/63|**30/59**|32/63|32/63|
|203|raw|24/48|31/60|32/64|30/57|27/54|
|203|scaled|32/61|32/61|32/64|32/64|32/64|
|204|raw|21/26|32/64|32/64|32/64|31/64|
|204|scaled|31/62|32/64|32/64|32/64|32/64|

Cells are deterministic /32, stochastic /64.

Scaled stochastic gamma=.999 endpoint effects by training seed: **+18.961, +0.220, +83.234, -1.788**; mean **+25.157**, descriptive SE across four seeds 19.915. This passes the preregistered positive-mean endpoint condition but is not treated as precise population evidence.

## Critic conditioning

Mean second-hidden-layer tanh saturation across seeds:

| Update | Raw | Scaled | Raw targets above current head ceiling | Scaled |
|---:|---:|---:|---:|---:|
|32|81.7%|**1.8%**|14.7%|0.0%|
|64|94.5%|**4.7%**|90.2%|0.0%|
|128|97.9%|**5.6%**|85.4%|0.0%|
|256|94.7%|**6.2%**|20.5%|0.0%|
|384|92.8%|**9.0%**|0.0%|0.0%|
|512|92.0%|**9.4%**|0.0%|0.0%|

So scaling clearly fixes the measured saturation/head-scale pathology. Seed202 nevertheless erodes while the scaled critic is unsaturated. Severe tanh saturation is therefore **not necessary** for PPO policy erosion.

This does not establish that scaled bootstrap targets are accurate long-horizon values. Healthy activations and lower fitting error are conditioning checks, not independent continuation-value calibration.

## Verification

Independent NumPy-only reducer checks all four manifests/executed sources, pinned runtime/native digest, 56 policy snapshots and the initial 1/S transform, 5,376 evaluation records and reward-trace score reconstruction, 4,096 rollout diagnostic records, budgets, and both preregistered screens.

Verified main-run cost: **8,388,608 training interactions + 5,509,511 evaluation native calls**, plus preflight/control work.

Fresh evidence-package replay:
- 864 package payload hashes verified
- VERIFIED.json reproduced byte-for-byte
- endpoint_screen = true
- preservation_screen = false

Evidence ZIP: `ppo-value-scale-evidence-20261007.zip`, 32,612,487 bytes, SHA256 `10a6f70c680910dfebc6fd1e1de2c893de6e6e7c2c3c14dd68ed61cead763724`.
Measurement source SHA256 `583a304719a3c850cbd21ed7ea38ceea47141ae006066fcde399caab4b2db097`.

## Next causal question

Do **not** retune S or launch another sweep. Exact-replay/localize the exposed scaled seed202 degradation between 262,144 and 524,288, then test same-state continuation value/action credit around a harmful update. That can distinguish residual bootstrap-target error from finite-batch credit, policy-step effects, or another optimizer dynamic while the severe critic-saturation confound is absent.

No production/default/master changes or merge. #33/#35 remain open.
