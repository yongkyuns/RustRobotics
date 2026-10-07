# Scaled seed-202 residual erosion diagnostic — October 7, 2026

Development diagnostic for RustRobotics issue #35. This follows the value-scaling experiment and the fixed localization protocol in comments 6037349641 / 6037717516. It does **not** change production/default PPO settings.

## Status and replay gate

The diagnostic was rerun on the recorded host class (AMD EPYC 9V74) with the preserved Python 3.11.14 / Torch 2.8.0+cpu / NumPy 2.2.6 / SB3 2.9.0 / Gymnasium 1.3.0 runtime and the same native archive SHA256 `58af5206058b24a8cd39c2665d9bc187200a4e8acaf3841a367d27731e09cdc0`.

The locally reconstructed scaled seed-202 run reproduces the published completion counts at the key historical checkpoints:

- 131,072 interactions: **32/32 deterministic, 62/64 stochastic**
- 262,144 interactions: **32/32 deterministic, 63/64 stochastic**
- 524,288 interactions: **30/32 deterministic, 59/64 stochastic**

The original `ppo-value-scale-evidence-20261007.zip` is not available in the current conversation/library, so byte-for-byte policy/evaluation identity against the original archive could not be checked. Therefore the results below are a strong same-runtime reproduction, but **not promoted as a completed historical exact-replay attribution** until that archive gate is satisfied.

## Localization

Using the predeclared domain `12970202`, coarse policies 128,136,...,256 select block **216 -> 224**:

- update 216: 32/32 deterministic, 59/64 stochastic
- update 224: 31/32 deterministic, 55/64 stochastic

Fine evaluation of every policy in that block selects **update 223**, i.e. policy 222 -> 223:

- update 222: **31/32 deterministic, 58/64 stochastic**, stochastic mean discounted score 79.681477
- update 223: **31/32 deterministic, 55/64 stochastic**, stochastic mean discounted score 79.602342

Fresh confirmation on eight fixed domains `14220202 + 100000*k` gives:

- deterministic: **0 lost / 0 gained completions**; mean discounted score change -0.100964
- stochastic: **10 lost / 0 gained completions**, pooled **488/512 -> 478/512**; mean discounted score change -0.073064; mean episode-length change -9.555 ticks

No confirmation panel changes the selected update.

## Selected-batch conditioning and coverage

Update 223 has 2,048 rows, 8 episode starts, 8 timeouts and no physical terminal inside the fitting batch. State coverage is not the old near-equilibrium/reset-free witness:

- position: -0.617 to +1.423 m
- velocity: -2.149 to +2.542 m/s
- angle: -0.236 to +0.293 rad; 923 rows exceed |angle| .05 and 495 exceed .1
- angular velocity: -0.515 to +0.749 rad/s

Incoming critic numerical conditioning is healthy by the earlier saturation criterion:

- second hidden tanh |activation| > .99: **4.89%**
- |activation| > .9999: **2.52%**
- physicalized current output-head ceiling: **3653.91**
- fitting targets above the ceiling: **0%**

Physicalized incoming critic prediction mean/SD is 935.325 / 67.765; target mean/SD is 965.710 / 21.104. Target RMSE is 61.276 before the update and 11.464 after it. Thus the critic fits this rollout target much better while the actor's real control performance worsens.

## Recorded PPO credit versus independent continuation

The actual incoming-to-outgoing actor displacement has **positive** directional credit on the realized fitting batch:

- recorded physicalized raw-GAE directional diagnostic: **+0.073314**
- normalized-GAE directional diagnostic: **+0.001761**
- full-batch normalized clipped surrogate: approximately 0 -> **+0.000991** (higher surrogate is better)
- mean Gaussian KL old->new on fitting observations: **0.001768**

So ordinary PPO regards the outgoing direction as an improvement on the sampled batch.

Independent arbitrary-state continuation uses the fixed 64 fitting rows `16 + 32*k`, 16 common future-noise replications per row, unchanged native physics, and the old policy after the first-action branch. Critic values are converted back to physical reward units before bootstrap accounting.

New first action then old policy, stopped gamma=.999 return effect:

| Horizon | Effect | approximate pointwise 99% interval across 16 future-noise replication means |
|---|---:|---:|
| 1,024 ticks | -0.003091 | [-0.003469, -0.002714] |
| 4,096 ticks | -0.003112 | [-0.003493, -0.002731] |
| 8,192 ticks | **-0.003112** | **[-0.003493, -0.002731]** |

New policy throughout at 8,192 ticks is also worse by **-0.162475 [-0.175639, -0.149310]** on the same conditional panel.

Most importantly, the independently resampled **physicalized bootstrapped GAE** first-action effect is **-0.004923 [-0.005737, -0.004109]**. Therefore the critic/bootstrapped estimator, when averaged over new future-noise draws from these same states, agrees with the long-horizon direction: the new action distribution is harmful.

This is opposite to the **positive** directional credit in the one realized 2,048-row training batch.

## Cutoff calibration

At the eight actual timeout anchors plus the final rollout cutoff, the incoming physicalized critic predicts 950.4 to 989.7. Independent 8,192-tick old-policy continuation means span 971.8 to 984.7. Across the nine anchors:

- RMSE of prediction versus continuation mean: **13.87**
- MAE: **10.45**
- largest observed errors: approximately **-32.39** and **+17.90**

The critic is not perfect, but this is not the previous nearly constant/saturated critic with order-100 state-dependent continuation errors.

## Interpretation

For this selected scaled witness, severe value-network saturation is absent and resampled bootstrapped action credit has the **correct harmful sign**. The destructive update instead appears because the single realized PPO batch gives the actor displacement positive credit while independent resampling gives negative credit.

That supports **finite-batch/stochastic action-credit noise (and possibly its interaction with multi-epoch PPO optimization)** as the next causal mechanism to isolate. It does **not** support a claim that persistent wrong-sign critic bootstrap expectation is necessary for this residual erosion.

This remains one exposed, outcome-localized development witness. It does not establish a universal PPO failure mechanism, a production fix, or a stopping rule. No scale/gamma/lambda/lr/clip/batch/epoch/reward/reset/noise/default/master/deployment change was made.
