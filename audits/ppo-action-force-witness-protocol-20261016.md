# Issue #42 — Selected failure-witness force replay (development-only)

A deliberately *non-independent* diagnostic of completed
[categorical-vs-Gaussian action-space ablation](ppo-matched-action-ablation-results-20261016.md).
The four keys **4100100015**, **4100100021**, **4100100170** (categorical success
and Gaussian failure in each seed/mode) and **4100100100** (a shared hard
failure) were selected AFTER seeing the development outcomes. They must
not become a new held-out acceptance cohort or be used to tune a winning
checkpoint.

Run all four original development training seeds in both action laws for
976 updates each, on the exact same native plant, PPO recipe and 64-key
common difficult panel as the prior successful experiment. At the final
checkpoint, evaluate each key in deterministic and stochastic modes on
the full **64-lane batch**, and assert that all 64 episode summaries are
bitwise identical to the original uninstrumented evaluator. Output the
entire physical 10-ms trajectory for all eight policies/modes/key sets:
noisy observation, true physical state before/after, desired and
realized force, cart/pole boundaries and terminal status.

In a separate verifier, calculate predeclared diagnostic summaries:
**mean absolute desired/applied force over first 30 ticks**, full-episode
RMS commanded force, fraction of commands >=18 N, survival and which
physical boundary was violated. This can test whether weak-force amplitude
or direction/timing plausibly explains the Gaussian rail failures, but
cannot by itself establish a causal mechanism. No training uses selected
witness keys. Existing training algorithm and production PPO stay untouched.

An all-green workflow means eight complete training replays, exact
nominal checkpoint reproduction, complete force traces and valid data.
It is **not** independent #33 acceptance or evidence of hardware safety.
