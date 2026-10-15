# PPO exploration-scale crossover: prepared runner and verified software controls

**October 15, 2026. This package does not contain the registered four-versus-ten-pass crossover result.**

The scientific design is RustRobotics issue #35 comment **6201344116**, following the epoch-reuse report at **5815f1f8221ee752192352b20c44108fb3032f51**. Current continuation/input-recovery disclosure: **6201503174**. The required DECAY4 final checkpoints and original diagonal raw trajectories were not recoverable in this session. They are not replaced by retrained policies, guessed sigmas, or the available ten-pass policies.

This package contains the complete new inference-only runner, exact-diagonal gates, input exporter, independent reduction, 34 software tests, the unchanged required prior LR-decay inputs, and all 1,280 newly executed **ten-pass software-control** trajectories. Nothing here changes production PPO, rewards, termination, normalization, optimizer settings or deployment.

## Completed versus unexecuted

Completed: all 1,024 saved final ten-pass reference case summaries reproduce exactly; 256 additional deterministic trajectories remain byte-identical after changing only the learned sigma to a different genuine ten-pass checkpoint's sigma. Native work: **1,887,424 transitions**. Training and optimizer transactions: **zero**. All 34 tests pass. The source-controlled input check rejects the absent epoch bundle before importing Torch or initializing the simulator.

Unexecuted: all scientific M4S4/M4S10/M10S4/M10S10 comparisons. The positive path of exporting the actual original epoch ZIP and its 32 diagonal gates remains untested on that missing dataset. Synthetic algebra tests are not substitutes for it. `registered_crossover_complete` remains **false** in every current control report.

## Offline reproduction of the work actually completed

From this package's root, with Python, NumPy and SciPy:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 python REPLAY.py
```

Recorded offline environment: Python 3.13.5; inspect `provenance/reader.json` for exact NumPy/SciPy versions. Replay checks the payload manifest, runs all 34 tests, independently re-reads all control paths and requires `analysis/CONTROL_ANALYSIS.json` to regenerate byte-for-byte. It does not load pickle, import Torch, run the simulator or train.

The mathematical cost decomposition uses float64. The separate inherited `physical_reader.py` checks the original float32 reward calculation bit-for-bit. Post-terminal padding is never interpreted as motion or force.

## Force-field clarification

In the retained native bridge, `CStep.force` is the **clipped commanded force**. The dynamics then add actuation noise and disturbances internally. The reward charges the clipped command, not that disturbed force. Thus the RMS and effort metrics here refer to **command force**. Actual disturbed-force RMS is not recoverable from the exported field and is not claimed.

RMS includes the command on every physically executed transition, including a terminal transition. Reward effort and state-cost totals **exclude a physical terminal transition**, whose reward is replaced by -10; a live horizon-ending transition retains ordinary reward and effort. The exact original source excerpts are retained in `provenance/bridge.rs` and `provenance/env.rs`. This clarification changes labels, not rewards, raw measurements, numerical formulas or acceptance criteria.

## Recovering the original scientific inputs

Original full archive:

- Name: `ppo_epoch_reuse_evidence_20261015.zip`
- Size: 1,127,945,770 bytes
- SHA256: `5378a475cd4f904f5601327a4dc309c4f9f5dadcc5a558a115ced79211d1e230`

The published compact archive is `ppo_epoch_reuse_report_code_20261015.zip`, SHA256 `ebe5966e10ac6fdf1694734886ff554a93fb2f5b5714492ff158a49af4de3eaf`. Compact alone omits the original diagonal paths and is not enough for the registered identity gate. The original report also describes compact-plus-four-raw-volume hydration; this exporter currently accepts the exact original **full ZIP only**, not a differently recompressed hydration.

Export the 65 required original payloads—eight policies, matching statistics/receipts, eight source panels plus their selection receipt, and 32 diagonal paths:

```sh
python code/pack_inputs.py /path/ppo_epoch_reuse_evidence_20261015.zip original_inputs
python code/run_crossover.py check-inputs --inputs original_inputs --output input_checked.json
```

The exporter validates the exact original archive hash and each selected member hash before writing a new directory. It rejects unsafe/duplicate ZIP paths. The selected bundle retains the original manifest and checkpoint bindings; it must be created by the trusted exporter, not hand-authored from arbitrary models.

## Executing the registered crossover after recovery

Native runtime: Python **3.11.14**, Torch **2.8.0+cpu**, NumPy **2.2.6**, SB3 **2.9.0**, Gymnasium **1.3.0**, original native/source byte gates. Runtime installer and third-party font files are intentionally excluded. Its existing archive recovery anchor is SHA256 `c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1` (GitHub runtime artifact 11912632514 in run 38595509112).

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/pinned/python3.11 code/run_crossover.py run \
  --inputs original_inputs --reference reference --output scientific_run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python code/analyze.py protocol --inputs original_inputs \
  --results scientific_run --output scientific_analysis
```

All 32 diagonal groups must match original arrays **byte-for-byte before any crossed group is collected**. An exact mismatch is fatal; do not waive it or substitute archived outcomes. The complete actor mean network and its matching observation-normalizer are held together; only `log_std` is crossed. Critic tensors remain unchanged and are unused. Deterministic output must be independent of sigma. No optimizer call is allowed. The reducer reports both conditional sigma effects, both mean-network effects, the interaction, paired completion gains/losses and four-seed pointwise 99% intervals. Four seeds, not modes or trajectories, are the uncertainty unit.

A stopped run is not resumed by overwriting outputs. Preserve failed output and the cause before any separately documented recovery. No fallback runs new training.

## Software controls, separately labeled

The following uses only the genuine available constant/decay **ten-pass** checkpoints. The sigma donor is not S4. Results cannot answer the scientific M4/M10 question:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/pinned/python3.11 code/run_crossover.py controls \
  --reference reference --output new_controls
```

`reference/` is the unchanged prior LR-decay compact extraction: outer SHA256 `5cfa6ec278eaf8f2812b2c84952309b18b14190fdd6d0018147b725919b2f5b6`, 607 payload hashes verified. Its 30 previous tests and saved exact controls passed separately. It is not the missing epoch-reuse dataset. Its nested manifests describe their original packages; the new top-level manifest describes the present package.

## Source patch and repository scope

The accompanying text patch adds only `audits/clean_reference/exploration_crossover_20261015/`. It contains every new module, tests, README and offline replay; binary reference inputs and genuine control fixtures remain in this evidence ZIP. To run the fixture tests in a patched checkout, copy or link `reference/`, `controls/`, `analysis/` and the package manifest there, or run the packaged copy directly. Tests are not claimed to run without those explicit fixtures.

The tests cover actual parameter isolation and deterministic invariance, original case identity, normalizer validation, terminal-aware cost accounting, physical corruption, archive rejection, missing-input failure, factorial orientation and four-seed uncertainty. They do not validate unseen M4 checkpoint bytes, cross-platform numerical equivalence, a learning improvement, production Rust integration or issue #33 qualification.
