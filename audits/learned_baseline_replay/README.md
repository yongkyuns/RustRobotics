# Learned-baseline numerical full-step replay — 22 September 2026

**Registered inputs recovered exactly; both learned-baseline numerical full steps have positive local action-credit projection. Native policy-performance evaluation is still pending.**

This is the registered numerical supplement in issue #35 comment 5782543705, on the exposed seed 41006/update 4140. It does not replace the native protocol in comments 5780978113/5781164214. No candidate, original expected hash, training default, reward, lambda, or evaluation case changed.

## Input recovery

The unchanged preflight verifier now exits 0 / INPUTS_MATCH. Both the verifier and original study.Critic reproduce the exact registered prediction bytes:

| Arm | Registered and recovered prediction SHA256 |
|---|---|
| Independent-uniform | `8d2cdd410a6247ae08e88eb228e21baf08349a932db8941a2062c892e7d8a937` |
| Independent-early-balanced | `e62cfcc96f3458cc74ff7e17110f1c55ac759d63f132381a27c274c741ecf12d` |

The fixed critic hashes remain `9bc8227044ec12d1a7faa38d351c4ae3a2972deaaeefd98fdce51e5a5e8f41ba` and `e5d717c4ee1ca340906cb4acc35d373bd3b070a26d70bfe5ac15c78997679096`. The ordered float32 observation-byte hash is `a913ceac0a2decc47d043a4985b8120e1d9dd83fbc8e42ffaf0f1ccaadba1025`.

The previous unmatched arrays are retained. Their maximum discrepancy from the recovered arrays is now measurable: 4.57763671875e-5 for both critics; 555/1024 and 565/1024 float32 values differed. Software version/build reports match across sessions, but the specific cause remains unresolved. This is not a proved CPU-dispatch explanation. The formerly unmatched arrays were not optimized. No tolerance replaces the original candidate-input hash checks.

Actual matching bytes are preserved in the evidence bundle, together with a standard-library fixture loader checking exact ordered observations, model bytes and prediction hashes. Numerical replay uses these fixtures rather than regenerating predictions on each host.

## Full numerical transaction

Each learned candidate starts from the archived incoming actor and all six actual Adam first/second moments and counters. Use the original 16 minibatches in order, learning rate .0003, epsilon 1e-5, clipping .2, four epochs, 256 rows per minibatch. Only actor advantages change to `normalize(original_return - learned_baseline)`. Observations, recorded actions, old likelihoods and critic targets remain fixed. No actor optimizer reset, new fitting data, extra epochs, checkpoint selection or continuation training.

Independent PyTorch PPO gradients feed the explicit audited float32 Adam recurrence. Before candidate interpretation, recorded native gradients reproduce all 32 original actor/critic transactions, moments and weights bit-for-bit. Independently recomputed gradients reproduce original actor and critic weights within 2.9802322387695312e-8 in this runtime. Ordered input identity across source archives and all epoch permutations also pass.

## Final step 16

The credit column is the INITIAL normalized paired-credit gradient dotted with the complete actor displacement. It is a local derivative of a finite-state surrogate, **not a measured return change**.

| Actor update | Local normalized-credit derivative | Mean policy KL on original states | Displacement norm |
|---|---:|---:|---:|
| Original recorded native update | **-0.0620121** | 0.00586271 | 0.0298143 |
| Uniform learned baseline, numerical replay | **+0.0276369** | 0.00742061 | 0.0285136 |
| Early-balanced learned baseline, numerical replay | **+0.0301000** | 0.00545672 | 0.0336080 |

The independent original replay gives -0.0620119. Final displacement/credit cosines are -0.125643, +0.058549 and +0.054101 respectively. Both candidate signs improve, but final displacements are not strongly parallel to the reference.

Own clipped-surrogate gains are +.00708195 (uniform) and +.01182040 (early-balanced), versus +.00841517 for the original. These use different advantage targets and must not be used to rank policy performance.

All intermediate steps remain visible. Uniform's cumulative credit projection is negative after steps 1–3 before finishing positive. Early-balanced is negative after step 1 and steps 12–14 before finishing positive. The original has positive intermediate projections after steps 8–10 but finishes negative. No favorable intermediate checkpoint is selected.

## Boundaries and verification

The two candidates execute 32 numerical actor transactions. There are zero new simulator transitions and zero new policy evaluation episodes. The same exposed states/actions and eight-draw finite-horizon credit reference are reused; this is not new state coverage or a new training cohort. A local derivative does not certify nonlinear full-policy improvement. The expensive frozen-policy critic fitting budget remains inherited from the pilot, not a demonstrated online tracking correction.

**46 tests pass**: 27 inherited preflight tests and 19 new tests, with actual source archives supplied. They cover original native Adam parity, fixture/input identity and corruption rejection, all 48 saved actor transaction recurrences and minibatch orders, exact candidate advantages, metric recomputation, score sign/linearity and no-evaluation accounting. Separate NumPy backpropagation verifies all 32 candidate gradients: maximum relative L2 error 2.91947e-6 and maximum absolute coordinate error 1.07403e-6. Both remain below the declared 1e-4 checks.

All 288 payload hashes in the two input archives pass; embedded archives are counted as byte streams, not as an additional rehash of their every nested member. A complete repeat reproduces all 59 result files byte-for-byte. A fresh bundle extraction also passes all 46 tests. No native/browser CI pass is claimed.

Native run **35771100199**, executable source **24b31e6ea8e24afd1f8bbdc59dff573bab85a155**, still has preflight and values jobs queued at the final status check. The required native prefix/sham/unchanged-critic controls and 6,144 fixed evaluation records have not been produced. This report is outside that workflow's trigger paths; the existing native experiment and expected hashes are unchanged.

## Retained evidence

Conversation bundle: `rustrobotics-learned-baseline-full-step-20260922.zip`, 5,187,265 bytes, SHA256 `6b3ea5671725a139e78f52988f86373de762d7b424e456213e8122a564dfe223`, 85 verified payloads. It contains the full report, exact recovered fixtures, previous mismatch evidence, all transaction gradients/moments/weights, numerical replay source, tests, logs and repeat receipt. Large original input archives are not duplicated.

Source hashes:
- `source/verify.py`: `b707de5ec99ea7df085d3d1d7d678e29ff312da740c61b439db90c828a4516e7` (unchanged).
- `source/replay.py`: `db58489e9bb50474e93bbf5f36c422f398ae23315c0872ba5ced1310b32b0cba`.
- `source/test_replay.py`: `70fa05eaa6ea58993e1baf508fe2c93d9f9fc33473277e9759160475d21eeb49`.
- `source/registered_predictions.py`: `797c6b932cc1f5430bcf06a31c1945314d1584783c4b0c822f37a8b4a0942a6a`.

Inputs: artifact 10671125548/run 35661200694, SHA256 `21010952eec4c02992cc664c76829c99a150c0296467a29fed9b2d7307ee32b8`; prior critic-generalization conversation bundle, SHA256 `93b61ef1a617bf90046105e004a8c22571978ba5ef702713ae6a4ab0d6e84791`. Retrieve actual archives or saved conversation files rather than assuming an old sandbox path exists. The full bundle README contains executable reproduction commands. The Python implementation and test files are in that bundle; this repository addition is the evidence report only.
