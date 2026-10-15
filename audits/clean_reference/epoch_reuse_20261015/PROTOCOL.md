# PPO rollout reuse: twelve-run development experiment

Registered before new main outcomes in issue #35 comment 6194736723. Starting audit head: aa66d7a5134e0314f26b2391b6566bb3e2e88594. This is not production qualification.

## Fixed intervention

Four exposed development seeds (201–204), three NEW same-host arms:

- CONSTANT10: normalized reference, constant learning rate 0.0003, ten PPO epochs.
- DECAY10: the previous normalized nonzero-decay recipe, ten epochs.
- DECAY4: identical to DECAY10 except four PPO epochs.

Changing epochs reduces both actor and critic passes through each rollout. It is not a separate critic intervention or an equal-optimizer-work comparison. All arms retain stock PPO training, rollout collection and GAE, two 64-unit tanh layers, gamma 0.999, lambda 1, 2048 samples per rollout, minibatch 64, Adam epsilon 1e-5, policy clip 0.2, global gradient norm 0.5, entropy coefficient zero, and the inherited reward-unit scale 1/1000 with divided initial critic head. All use observation-only VecNormalize (clip 10, epsilon 1e-8), training-only statistics and frozen evaluation.

Original nonlinear native plant, physical reward, observation/action disturbances, reset distribution, termination and 256-step external training timeout are unchanged. No teacher, reward shaping, extra data, curriculum, rollback, best-checkpoint selection, optimizer reset or added KL stopping.

Decay is driven by absolute completed training interactions: 0.0003 linearly down to 0.00003 at 1,048,576 interactions, then a positive-floor tail through 1,310,720. Checkpoints: 0, 65,536, 262,144, 524,288, 786,432, 1,048,576, 1,310,720. Each arm runs 640 rollouts. Expected optimizer transactions: 204,800 for each ten-epoch arm; 81,920 for four epochs. No outcome-adaptive budget.

Current CPU: Intel Xeon Platinum 8370C. The preceding AMD results are not substituted as a comparator. Both reference arms are retrained on the current host. Three initial models match exactly per seed. Fixed sequential order within four single-threaded workers: 201 CONSTANT10/DECAY10/DECAY4; 202 DECAY10/DECAY4/CONSTANT10; 203 DECAY4/CONSTANT10/DECAY10; 204 CONSTANT10/DECAY4/DECAY10. No automatic retry.

## Original evaluation and acceptance rule retained

All original frozen panels: 64 ordinary and 64 difficult native resets per seed, both deterministic and stochastic modes, original keys and standardized innovations, 64 inference lanes including inactive lanes, 1500-tick cap, original first physical ending, no bootstrap. Record full trajectories, raw/gamma-0.999 returns, completions, casewise gains/losses and minimum signed rail clearance. Completion with at least 50mm clearance is a descriptive category, not a changed physical limit or hardware safety requirement.

At BOTH 1,048,576 and 1,310,720, compare DECAY4 with the new same-host CONSTANT10 under the original conditions: no per-seed/mode decrease in difficult completion; no per-seed/mode decrease in ordinary mean raw return; mean difficult completion-with-50mm gain at least five percentage points. Independently require no decrease in DECAY4's own difficult completion or 50mm count during the positive-floor tail, per seed/mode. Apply that same rule to DECAY10 for context without rewriting the old result.

The key isolation contrast is DECAY4 minus DECAY10, reported at every checkpoint/panel/mode with all four seed differences and pointwise 99% Student intervals across those four training-seed means. Modes and repeated episodes are not independent trained agents. This isolation comparison does not silently replace the original progression comparator or thresholds. No intermediate checkpoint is a selected deployment candidate. A pass would be a development result only; issue #33 qualification and production Rust checks remain separate.

## Execution gates

Pinned Python 3.11.14, Torch 2.8.0+cpu, NumPy 2.2.6, SB3 2.9.0 and Gymnasium 1.3.0. Original native/PPO/collector/GAE/normalizer source hashes remain gated. Before main runs: prior DECAY10 versus new wrapper/observer identity; stock four-epoch versus observed four-epoch identity; one-call versus split-call training identity; actual 128 versus 320 optimizer transactions per rollout; unchanged absolute LR; equal initial tensors; frozen-evaluation and independent physical reward/action/clearance controls. Their successful live checks are not evidence that the candidate works.

Actual collection source hashes are frozen before launch. Complete code, controls, checkpoint/statistics, all successful/failed outcomes and exact costs will accompany the result. This protocol does not assert main-run completion. No production default, master, protected MuJoCo/control source, held-out training seed or deployment is changed; no merge is authorized by this experiment.
