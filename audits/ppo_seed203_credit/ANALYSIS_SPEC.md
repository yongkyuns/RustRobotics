# Frozen recovery action-ranking analysis

This describes the predeclared analysis of run35394710493 at executed commit527f44b87aef1f973180ab4f1202beec5584690c. It is not a results report.

The exact training baseline remains d3f59f9bf38d4038ba7c5008e3a3b41c15e99835. Protocol: issue35 comment5736087948. Three final seed203 actors and their pre/post-extra-phase critics are frozen. Each actor supplies16 distinct-episode recovery states selected without reading rewards, failures, critic errors or new probe outcomes. Each state receives256 paired native continuations, first latent action shifted by -0.02 or +0.02, with identical subsequent controller and common temporal randomness within the pair. No training or parameter update is performed by probe.py.

## Primary contrast

The positive direction is always plus-first-action minus minus-first-action. Both branches use the actor's unchanged policy after the first action. For each state separately compute:

- finite-horizon discounted-return difference, using all recorded rewards through termination or1500 steps and no critic bootstrap;
- lambda0.95 GAE difference using the final critic, retaining true-terminal, timeout and live-final-observation semantics.

The two estimates use the same independently generated paired trajectories, so they may be correlated. Independence is between fresh continuation replications conditional on the prescribed physical state and initial noisy observation, not between the two estimates or between observations from an episode. Bonferroni correction does not require independence among the96 estimated quantities.

Use two-sided Student intervals at1-0.01/96 for each primary quantity:48 states times two differences. An opposite-sign cell requires BOTH intervals to exclude zero on opposite sides. An agree cell requires both to exclude zero on the same side. All other cells are unresolved. Retain raw mean signs and unadjusted99% intervals separately. These are approximate conditional Monte Carlo intervals, not exact finite-sample guarantees, independent training-seed inference, or held-out qualification.

The pre-extra critic, one-step TD credit and the lambda-weighted reward/future-critic decomposition are secondary diagnostics. Their comparisons do not enlarge the primary tested family or silently replace its decision criterion. Initial-state value is identical within each pair and cancels from its difference. Therefore a wrong directional credit cannot be attributed merely to a constant offset in that initial value estimate.

## Controls and offline checks

The actual native ABI uses unchanged PendulumEnv::step_with_rng, with exact saved physical state and fresh diagnostic randomness. Native tests compare the ABI against direct environment stepping and check terminal/timeout precedence and invalid calls. Identical-first-action pairs must produce exactly equal trajectories and zero contrasts. Lambda1 TD sums must telescope to recorded discounted return after explicitly removing any time-limit bootstrap and adding back the initial value.

Offline analysis verifies archive/member/source hashes; independently reselects the panel from the original NPZ files; checks every outcome shape, length, ending, paired initial baseline and interaction count; and reconstructs the complete predetermined replicate0 traces. Trace checks cover observation continuity/noise bounds, initial state identity, coupled Euler dynamics, bounded force innovations, reward formula, actor sampling, critic inference and both GAE sums. This uses independent float64 reconstruction of float32 network snapshots, not a claim of exact Burn/PyTorch arithmetic or full resimulation of aggregate-only replications.

All main replications retain returns, both critics' GAE/TD sums, lambda-weighted rewards, terminal bootstraps, endings, lengths and initial actions. Full step traces exist only for the declared replicate0 of each state/branch. No claim of complete offline trajectory resimulation for the other255 replications.

## Interpretation boundary

A wrong-sign result is evidence of inaccurate local action ranking for the tested frozen policy, finite action perturbation, fixed physical state/observation and finite horizon. It is not a replay of an actual optimizer direction and does not establish the causal training history of seed203's final regression. Across-arm panels come from different policies' visited states; their counts are not matched-state causal effect estimates. Only within-arm pre/post critic comparisons hold actor, states and continuations fixed. No diagnostic state, outcome or inferred controller enters training.

Actual interaction counts and all contrary/inconclusive results must be reported. Native regression/control work is additional to the measured continuation budget. No production-default change or sustained-balancing claim follows from this diagnostic alone.
