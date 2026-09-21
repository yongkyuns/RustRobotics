# Recovery-data lambda factorial — submitted

Protocol: issue #35 comment 5768296822.

This audit follows the correctness-first PPO review. It compares only the strongest recovery-data recipe at GAE lambda 1.0 versus 0.95. The lambda1 arm must exactly reproduce archived fresh-cohort recovery-union results from run 35453178361 before the comparison is interpreted.

## Motivation fixed before execution

The independently confirmed harmful seed41006/update4140 uses a 1024-row fitting batch comprising only nine physical trajectory segments. Under lambda1, the incoming policy-gradient direction is cosine -0.503 versus an independently sampled conditional first-action-credit gradient. Decomposing the lambda1 GAE gradient shows a trajectory-common component norm 2.303, within-trajectory component 0.765; their cosines versus conditional action credit are -0.729 and +0.831 respectively.

On the exact same recorded rewards/actions/critic/boundaries, recomputing GAE at lambda.95 rotates that seed41006 gradient to cosine +0.700 versus conditional action credit and reduces the trajectory-common gradient norm to 0.460 while the within-trajectory component rises to 1.898. No policy was retrained in that arithmetic sweep.

This is consistent with the older lambda experiment: lambda1 fixed selected first-action witnesses but made equal-budget from-scratch control worse than lambda.95 on every exposed seed. It does not prove lambda is the sole problem: seed41004's recorded update remains anti-aligned with conditional action credit even at lambda.95, so insufficient independent future trajectories remains a broader mechanism.

## Fixed execution

Branch: audit/ppo-recovery-lambda-factorial-20260921
Workflow run: 35662214716
Current source: 38733d9f81684f1d623576a2ecc4a4481b11c896

Eight exposed development seeds 41001–41008, random initialization, 4096 updates. Both arms use the same recovery-data collection, sampled tails, ordinary-reset supplemental streams, 1024 fitting rows, minibatch256, four epochs, learning rate, gamma, reward, noise, plant and evaluation keys. Lambda is the only intended learner difference.

Candidate lambda.95 recomputes corrected-primary and supplemental returns/advantages consistently at .95; no lambda1 auxiliary target is retained accidentally. The gamma-discounted sampled future itself is unchanged.

Final checkpoint4096 only. Historical lambda1 weights/updates/evaluations must reproduce exactly. Comparative and absolute gates are fixed in protocol comment5768296822. These are development seeds, not fresh qualification.

Production source/defaults and PR38/master/deployment are untouched.
