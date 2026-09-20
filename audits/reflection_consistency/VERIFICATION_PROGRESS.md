# Reflection consistency — five histories verified; cohort incomplete

September 20, 2026 UTC / September 19 Toronto. This is a progress report, not a completed eight-history controller comparison.

## Execution identity and status

The unchanged registered source `d11b9fbf99edd0d221c4dd179a421ca46466b6eb` is running as GitHub Actions run **35486221478**. The native preflight now PASSES: 73 unit/audit tests (including actual noiseless plant reflection), seven ordinary balancing controls, nine ordinary learning controls, strict Clippy, formatting and protected-source restoration. The main diagnostic endpoint is ignored in preflight and explicitly invoked by measurement jobs. Two historical heavy integration tests remain ignored.

At the latest job check, seeds **41001, 41003, 41004, 41005 and 41008** have completed successfully and their artifacts have been independently verified. Seeds **41002, 41006 and 41007** remain queued. No missing history is omitted from an overall decision; no eight-history interval or qualification result is computed from the available subset.

Registration5747262659 and its pre-execution point-count correction5747271701 are unchanged. All eight actors are the final global-normalization policies from run35482824488. The three frozen mappings are original mu(o), reflected -mu(-o), and odd projection [mu(o)-mu(-o)]/2 before the same Gaussian innovation and tanh/20N force transform. There is no training, critic modification, teacher, fitted coefficient, reward change, safety switch or deployment. Two forward evaluations are required for the odd mapping; no latency benchmark was performed.

## Available results, without cohort-level inference

Each cell is five-minute deterministic nominal / five-minute stochastic nominal / sixty-second stochastic outward completions, each out of64 attempted episodes. The environment remains noisy. Different panels use different keys. These are the same previously exposed cases, not fresh qualification.

| Seed | Original | Reflected | Odd projection |
|---|---|---|---|
|41001|64 /64 /52|64 /64 /54|64 /64 /60|
|41003|61 /62 /63|62 /63 /63|62 /63 /63|
|41004|64 /64 /62|64 /64 /62|64 /64 /62|
|41005|63 /63 /61|63 /63 /62|63 /64 /62|
|41008|60 /59 /60|60 /58 /59|60 /58 /59|

The available outcomes are mixed. Seed41001 gains eight recovery successes with the odd projection. Seed41008 loses one stochastic nominal success and one outward success. Seed41004's completion counts are unchanged. These descriptive differences are not a population conclusion, proof of harmlessness, or a statistical test on a selected favorable subset. The remaining three histories must be included before the registered paired-history analysis.

## Verification completed on these five histories

All six available GitHub ZIP digests (one build, five results) and **164 payload hashes** verify. The first-party build is artifact10597976382, ZIP SHA256 `b0852326edccc0c679a2e520ae0bddab93e8d620cb58c82e5e789433710ba3b3`; its executable SHA256 is `a287ba162d9e5ff1ffb9f7675b0bd07811f7df171cb8f71d2008fa88174da0ac`.

Checks cover **2,880 episode records**, the fixed initial-condition/key/panel denominators, all five original actors' historical episode fields, the225-point observation grid per actor, and **45 full replication0 traces containing835,579 transitions**. All historical original-policy episode scores reproduce exactly in CI. Exported trajectories independently reconstruct actor inference, action transformation, nonlinear dynamics, rewards, endings, force RMS and final-ten-second centring within the unchanged verifier tolerances. Maximum errors: actor5.489e-7,command1.932e-6,dynamics2.353e-7,reward1.404e-7,aggregate8.669e-13. Other replications retain aggregate outcomes rather than every step. Unexported RNG draws and all possible trajectories are not independently regenerated.

The original25 offline synthetic/property tests pass. Ten additional synthetic trace-corruption checks pass, bringing the total to **35**. Synthetic tests are not controller outcomes. The full-cohort verifier still requires all nine result/build archives and cannot produce a valid eight-history summary from this partial set.

## Separate local execution: failed exact replay, not comparative evidence

To make progress while the evaluation jobs were queued, a separate local execution was declared BEFORE any local episode outcomes in comment5747362636. It used the exact published executable above, without recompilation or changed conditions, on x86_64/glibc2.41. Its73 native tests also passed locally. The original GitHub workflow was not canceled, rerun or replaced.

All eight local seeds stopped at the first original-policy episode's exact historical return assertion, BEFORE executing any reflected or odd episode. All reached the same30000-step cap and matched asserted historical keys/caps/durations/endings, but accumulated returns differed by approximately1.4e-6 to7.35e-5. No assertion/tolerance was relaxed, and no local seed was retried or used in the controller comparison.

Comparing the full retained first-episode traces with the historical traces locates the first difference in commanded force for every seed, while the same-step state/observation/latent/innovation still agree. Differences are approximately0.95–1.91e-6 N. This localizes the initial discrepancy to numerical force computation, not initial conditions or a different sampled action innovation. A different runtime math implementation is plausible, but the precise library/CPU cause has not been uniquely established. These small differences are not a diagnosis of the RL performance failure.

All eight local original-only traces,240,000 steps in total, independently reconstruct their own rewards/dynamics/actions and return sums within the original offline tolerances. That is distinct from—and does not override—the FAILED native bitwise historical-return requirement. The original assertion's left operand is the historical value; an initial offline reader reversed expected/actual fields, was corrected using the immutable source's line216, and its failed log is preserved. No measured field or tolerance changed.

The failed local execution was recorded in comment5747397126. It adds240,000 original-policy replay steps, zero reflected/odd comparisons and zero new training transitions. Local tests and generic CI preflight experience are additional. Local results must not be pooled with the GitHub study or reported as GitHub job completion.

## Disposition

The five verified CI histories account for61,690,956 evaluation transitions so far; this is an incomplete cost count for the full study. No main training is performed in the frozen diagnostic. All remaining histories, contrary outcomes and infrastructure/runtime failures are retained.

No production normalization or learned-weight change, PR38/master/deployment change, merge or fallback installation occurred. The full reflection comparison remains incomplete pending the three original queued jobs. There is no justified production adoption or cohort-level robustness claim at this point.
