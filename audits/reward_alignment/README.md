# Reward-alignment audit — completed

Both parts are complete:

1. [Retained-data audit](RESULTS.md), recorded at a95fb428: all 4,608 reflection-study outcomes and the original preselected traces. Its queued-replay passages describe the historical status at that time, not the current status.
2. [Completed selected-case replay](SELECTED_REPLAY.md), recorded at deb3bb85: run35508137816, executed114fe450771f4fb9d2e3853d4a2c5757abf94891, completed successfully at2026-09-20 11:59:48UTC on its first attempt.

The selected replay reproduces all three historical episode records exactly. It explains why the reflected300-second survivor scores below the1.2-second rail failure atgamma0.99: its larger early discounted running costs outweigh its survival benefit. The symmetric survivor already has higher current reward, and the previous15 primary discordant pairs all favor survival. Re-scoring the selected trajectories atgamma0.995 reverses the secondary ordering, but no controller was trained at that discount.

There is no new robust-controller qualification or production change. The update adds60,120 diagnostic simulator transitions, zero main training transitions, and no extra noise replications. See SELECTED_REPLAY.md for full costs, verification and limits. PR38 remains separate and unmerged; gamma, reward, normalization, learned weights and deployment are unchanged.