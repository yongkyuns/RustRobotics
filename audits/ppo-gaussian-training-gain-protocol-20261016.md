# Experimental native Gaussian PPO — training-time force-gain test

**Issue #42, PR #41. Exposed development evidence ONLY.** The earlier
[evaluation-only gain experiment](ppo-gaussian-force-gain-results-20261016.md)
found that multiplying unchanged Gaussian policy commands by 1.5 or 2.0
recovered 70–75 of the 85 fixed difficult-panel failures, without a new
failure on the paired reset-key/mode/seed tuples. That is a mechanistic
motivation, **not** an independent confirmation or guaranteed training fix.

## Frozen intervention

Change one physical actor-to-plant mapping and use it consistently in
**both PPO training and evaluation**:

- gain_1p0: F = 20*tanh(z) N (original exact baseline).
- gain_1p5: F = clip(1.5*20*tanh(z), -20, +20) N.
- gain_2p0: F = clip(2.0*20*tanh(z), -20, +20) N.

All arms use the same Gaussian mean/logstd head with latent sampling,
same 64×64 tanh actor/critic, seeded initialization, 4×128 per-update
experience, 976 PPO updates, persistent Adam (epsilon 1e-5), global
gradient clip, PPO/critic clipping, GAE, reward, plant, noise, learning
rate and fixed seven checkpoint schedule. **No reward or regularization
coefficient changes.** PPO ratio is computed over sampled latent z,
which remains a valid latent-policy importance ratio even when physical
action clipping maps multiple z values onto the same force. Do not claim
this is the exact clipped-physical-action density or entropy.

Every arm runs all four **previously exposed** development train seeds
201–204 and the same ordinary/difficult per-seed and common reset panels,
64 keys each, deterministic and stochastic policies, 1500-tick horizon,
at updates 0/64/128/256/512/768/976. Keep all failures, checkpoint
results and logs; do not select a winning policy or checkpoint from
the confirmation set.

**Expected work**: 12 × 976 = 11,712 PPO updates, 5,996,544 native
training interactions, 43,008 individual evaluation rows. On the
gain_1p0 arm, normalize the first CSV column name and value back to
the original "distribution=squashed_gaussian" representation, then
require the resulting raw CSV SHA256 to match all four immutable
[matched-action experiment](ppo-matched-action-ablation-results-20261016.md)
baseline files exactly. This checks all seven checkpoints and every
episode, not merely final scores.

A green workflow **only** means complete evidence and exact old-control
reproduction. Gains 1.5 and 2.0 were chosen after seeing development
force data; no independent #33 holdout inference is justified. The
original supported production continuous PPO and MuJoCo/browser
packages remain unchanged, and PR #41 stays draft.
