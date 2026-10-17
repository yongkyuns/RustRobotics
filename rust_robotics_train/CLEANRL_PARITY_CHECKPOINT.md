# PR #41 parity checkpoint — October 15, 2026

Two numerical mismatches in the draft native port were identified and patched. **Eight operation-level Python tests actually pass**, with a second execution producing byte-identical result JSON. **Five new Rust regressions are committed but have not run.** This is not a compiled Burn parity result, training experiment or production reliability correction.

## Equal losses, different critic gradients

Pinned Burn 0.20.1 `max_pair` uses `self.mask_where(self < other, other)`, selecting the left operand at an exact tie. PyTorch's elementwise maximum splits the gradient equally between equal operands. These are different subgradient conventions; reproducing CleanRL requires matching the convention, not just the scalar objective.

The existing four-row clipped-PPO fixture has an exact float32 clipped-critic tie:

| Formulation | Combined loss | Last value gradient |
|---|---:|---:|
| PyTorch maximum | 0.7970551252365112 | 0.11249999701976776 |
| Strict selector reconstructed from Burn source | 0.7970551252365112 | 0.22499999403953552 |
| Explicit half-gradient at ties | 0.7970551252365112 | 0.11249999701976776 |

The previous forward-only test cannot detect that twofold gradient discrepancy. The patch adds `torch_maximum` using comparison masks and half weight at equality for both the actor and critic objectives.

**Evidence boundary:** the strict Burn selector was reconstructed in PyTorch from the pinned source; no Burn binary was executed. The new Rust tests cover the actual Burn backward operation when compilation is available. This is not proof that a tied loss caused earlier production training collapse.

## Recorded versus replayed action likelihood

The old collector logged the right probability as `ln(1 - p_left)` while PPO replay used tensor log-softmax. Float32 cancellation produces a non-unit ratio even for an unchanged policy. With logits `[16, 0]` and the rare right category:

- old recorded log probability: -15.942384719848633;
- tensor log-softmax: -16.0;
- replayed/recorded ratio with unchanged weights: **0.9440130591392517**, rather than 1.0.

This is a rare-action arithmetic witness, not an observed failure-frequency or learning-causality claim. More saturated examples are retained without claiming those actions were physically sampled.

The patched collector records the selected entry directly from the same batched log-softmax used in optimization. Both probabilities are calculated directly, not by subtracting from one. Single-observation and batched deterministic evaluation share the float32 probability-tie rule (left on ties).

## Executed checks

Pinned runtime: Python 3.11.14, PyTorch 2.8.0+cpu, NumPy 2.2.6. Runtime ZIP SHA256 `c7c581b0cb77530de6ed8f51dfcaebf60f4b9604bf401e9ac164e600b8e5f4f1` and all 11 payload hashes verify.

`python scripts/check_cleanrl_reference_parity.py --output parity.json` passes eight tests: forward constants, hidden gradient discrepancy, corrected fixture derivatives, tie gradients, 1,000 operand pairs in each of float32/float64 with inserted ties, rare-probability cancellation, unchanged-policy unit ratios, and three consecutive Adam transactions on direct logit/value operands. Those three steps are **not** a full neural-network or Burn optimizer parity test. Re-execution reproduces the full JSON byte-for-byte.

Actual native simulator transitions: **0**. New training runs: **0**. Rust compilation/tests: **not executed**. No inference or balancing performance qualification follows.

## Source and infrastructure

Incoming PR head `76f39d48a15a49cf304383fd4a6336f1aaa6193c`.

Corrected source commit `06eb98ae659f051aeb2130ab021d369b99a52bd5`, Git blob `209d4db38542de6146157929e5e0357629ea4c21`, SHA256 `60f2599bf4bd723cfee4fd84a082e821af6de0585edc39ac8e68dcdbfaa008e7`.

Executed Python script commit `2b8015ee6e7d3156c7d1761910b9c82ecb917224`, Git blob `b78ad0904b0ef772b682c79f85747b2031d61e03`, SHA256 `427ea49b77b6e6247ae8d59c99845061843dfbc597558cefa1266ab1bc044924`.

New Rust tests blob `a9cf8c2207d0aff7600e16c58259014cda85e796`.

Two intermediate source-transcription errors were detected and corrected before final blob verification; the evidence retains the note, not a fabricated compiler result.

The workflow now includes the stacked PR base, all reference test modules and branch concurrency. A newly introduced YAML syntax issue in the trailing `cleanrl_reference:: ` filter failed before creating jobs (run38824807631); local parsing reproduced the line47 error. Commit `1422e60bc56e76558e5f5f54abeb84085a06746f` changes that command to a block scalar. The corrected YAML passes local parsing and the new GitHub run38825202374 was **queued** when checked, not passed.

The working container lacks Rust/Cargo; the attempted clone failed DNS resolution. The Python runtime was recovered from retained local bytes. No compiler, dependency, formatting, Clippy or Rust unit-test success is claimed.

## Disposition

PR #41 remains **draft and unmerged**. Production PPO, runtime snapshots, browser, protected MuJoCo/control source and master are unchanged. Next gates remain actual Rust compilation and backward tests, full-network gradient/global-clipping/Adam comparison, then the fixed-budget multi-seed native task experiment. Neither numerical witness is asserted to be the root cause of historical PPO regressions.

Detailed local evidence includes old/new source, the patch, the executed Python checker, authored Rust tests, result JSON, repeat logs, runtime verification, and retained infrastructure failures. Third-party runtime installations and fonts are not redistributed.

Primary source: https://github.com/tracel-ai/burn/blob/v0.20.1/crates/burn-tensor/src/tensor/api/numeric.rs
