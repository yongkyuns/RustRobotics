# Historical recovery cross-over

Completed diagnostic and supporting literature:

- [Source-linked literature review](../PPO_LITERATURE_REVIEW_20261008.md)
- [Complete cross-over results and limitations](../PPO_HISTORICAL_CROSSOVER_RESULTS_20261008.md)
- [Pre-outcome cross-over protocol, issue #35 comment 6061891252](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6061891252)

`SELECTION.json` freezes the 16 matched episode pairs and original packet hashes. `replay.py` reconstructs the original seed201 learner and captures 32 incoming actor/critic snapshots, enforcing all six original checkpoint/reward-trace matches, all 512 diagnostics, the complete source-array checksum and selected physical-state/noisy-observation checksum. `ledger.py` normalizes tuple/list containers only, preserving exact numeric equality and row order.

Run the nine dependency-free ledger regression tests with:

```sh
python -m unittest discover -s audits/clean_reference/crossover_20261008 -p test_ledger.py -v
```

The companion `ppo_historical_crossover_evidence_20261008.zip` contains the exact executed native measurement wrapper, independent NumPy/SciPy verifier, 22 tests, all 32 recovered policies, all 2048 new trajectories, failure logs and numerical reports. It is self-contained for offline `OPENBLAS_NUM_THREADS=1 python REPLAY.py`; no simulation, Torch, optimizer or pickle loading is performed. Artifact SHA256 and size are recorded in the results document. It excludes the third-party pinned runtime and full prior143MB historical input archive; those are needed only for new native regeneration.

This directory is experimental evidence, not a new production trainer or a default change.
