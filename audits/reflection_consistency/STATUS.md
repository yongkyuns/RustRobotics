# Reflection consistency — completed, no controller promoted

[Complete verified results](RESULTS.md), report commit `2200ddb16811c3c06bef0e30484e825fd006225b`.

The original registered GitHub run **35486221478** at source **d11b9fbf99edd0d221c4dd179a421ca46466b6eb** completed successfully on its first attempt at **2026-09-20 03:59:20 UTC**. All nine jobs passed. The previous queued status is historical, not the current state.

| Frozen mapping | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
| Original |504|503|474|
| Reflected |504|503|479|
| Odd projection |505|504|483|

All six registered odd-minus-original paired-history intervals include zero. The original absolute reliability requirements fail in all three panels. Odd projection repairs13 original failures across the three panels but loses2 original successes. These are eight exposed trained actors and reused evaluation cases, not fresh/hardware qualification or an improvement in from-scratch learning.

Primary CI verification covers4,608 episodes,1,800 probes,72 full traces and exact replay of all1,536 historical original-policy episodes. Native preflight passes73 tests plus ordinary controls;37 offline tests pass. No main training was done.

A separately declared local execution of the exact CI-built binary failed strict historical return equality at each seed's first original episode, before any reflected/odd episode. Those eight failed attempts are retained and excluded from the comparative statistics. Their small numerical differences were not repaired by relaxing assertions or rerunning seeds. Full details and costs are in RESULTS.md.

No production default, normalization, learned weights, PR38, master, deployed asset or fallback controller was changed or merged. PR38 remains draft/open at4739f37. The earlier pending status is preserved in Git history at5cbf92d9; the five-history progress snapshot is preserved at9bc456fd. Both are superseded by the complete report.