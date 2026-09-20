# Source-normalization experiment — complete, candidate not qualified

The experiment and independent verification are complete. See [RESULTS.md](RESULTS.md) for the full protocol-bound report, contrary outcomes, costs and limitations.

Executed source: `a799c6d35c3753266fa382230d734e2d7de18312`. Workflow run `35482824488` completed on its first attempt with all nine jobs successful. Registration: issue #35 comment `5746872084`, posted before execution. The prior queued status is retained in Git history; no duplicate run was dispatched to replace it.

The experiment holds the four-of-eight outward-start mixture fixed and compares global advantage normalization, separate source means with a common global scale, and separate source means and scales. Only actor advantages change. Eight exposed training histories each branch into three 512-update continuations. This is development, not a fresh held-out cohort.

## Decision

**The preregistered separate-mean/separate-scale candidate fails the development screen and all three absolute robustness panels.**

| Final policy | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
| Global normalization |504|503|474|
| Separate means, common scale |504|503|460|
| Separate means and scales |503|500|465|

The candidate repairs eight of twelve previously exposed angle-failure cases, but those historical examples do not override poorer whole-panel completion. Normalization formulas and first-update isolation were verified; mathematically removing cross-source coupling was insufficient for reliable control.

Native preflight passed 102 tests plus ordinary integration controls. Independent analysis checked all nine original ZIPs, 21,504 new-domain episode records, 36 historical witness records, selected optimizer transactions and 228 retained full trajectories. The complete offline replay passed 67 tests and regenerated all ten numerical reports byte-for-byte, explicitly retaining both failed controller screens. See the full report for ignored endpoints, aggregate-only replications, historical-input retention and unexported-state limitations.

Keep the existing recipe's global normalization. No production default, PR #38, master, deployed asset or fallback controller was changed or merged. Successful CI and evidence replay do not establish controller qualification.
