# Recovery-policy cross-over: policy version versus state and noise

**October 8, 2026 — completed development diagnostic, not a qualified corrective PPO recipe.**

This executes the previously registered [issue #35 protocol6061891252](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-6061891252), following the observational early-recovery study. The [literature review](PPO_LITERATURE_REVIEW_20261008.md), committed at `d8280077ad2f2f2458d45c4c51591e78623e6c69`, maps the measured phenomena to primary research and separates hypotheses from established results.

## Main result

On identical physical states with paired future randomness, contemporaneous policies associated with original training timeouts complete **679/1024** new 15-second continuations. Policies associated with the matched original failures complete **406/1024**. The cross-over gains **275** paired completions and loses **2**.

This demonstrates policy-version effects on the selected states; the original observational contrast was not entirely explained by different trajectory noise. It is not a training-population comparison or an attribution to one optimizer update. Both policies are real PPO actors learned in the same development seed201 history. No LQR, teacher labels, fitted replacement actor or fallback is used.

| Physical starting state source | Failure-source actor | Timeout-source actor | Gained / lost completions |
|---|---:|---:|---:|
| Original failed episode |39/512|169/512|130 /0|
| Matched original timeout |367/512|510/512|145 /2|
| **Total** |**406/1024**|**679/1024**|**275 /2**|

The source timeout label means survival to the original256-tick training cutoff, not necessarily completed recovery. The two source states in a pair are only approximately matched; they are not interchangeable. The actor comparison within each row uses exactly the same state and noisy initial observation.

## Fixed cohort and policy contract

Sixteen pairs, four per original training quarter, were selected before new outcomes by fixed midpoint ranks among the77 prior distance-qualified pairs. Each has two actual reset states, two incoming historical actors, and32 future-noise replicas: **2048 trajectories** total. All pairs completed once; no outcome-driven retries, omissions, replacements or horizon extensions.

Keys are `2808202600 +10000*pair_index +100*state_side +replica`; action-Gaussian innovations use key XOR `0xAD219010`. The original native noisy plant, command bounds, physical terminal rules and raw reward formula are unchanged. The actor sees ordinary noisy observations; hidden physical states initialize the diagnostic plant only. The horizon is1500 ticks with no learned endpoint bootstrap. All return contrasts use gamma=.999.

The intervention is the whole frozen policy, including each actor's **own learned Gaussian standard deviation**. Equal standardized innovations therefore do not mean equal applied force perturbations. This experiment does not isolate deterministic mean-response changes from exploration-variance changes.

| Original pair's training quarter | Failure-source successes /256 | Timeout-source successes /256 | Mean return difference |
|---|---:|---:|---:|
|1|27|224|+475.302|
|2|128|144|+31.876|
|3|137|151|+35.622|
|4|114|160|+110.906|

The first quarter dominates the pooled improvement, and its timeout-source actors are all later in training. It mainly demonstrates acquisition, not forgetting. The equally weighted fixed-panel mean return contrast is **+163.426**, with approximate conditional99% Student interval **[152.599,174.254]** across32 independent future-replica panel means. This interval describes future-noise uncertainty for the fixed32 states and recovered actors; it is not uncertainty over independent trained agents. There is only **one exposed training history**.

## A concrete later-policy retention witness

After completing the fixed cohort, pair13 stands out as a useful next diagnostic. The actor used during rollout **406** is earlier than the actor used during rollout **464**. On the exact physical state that later appeared in the failed source episode, the earlier learned actor survives **32/32**, compared with **8/32** for the later actor under paired future innovations. Its mean gamma-.999 stopped-return advantage is **+443.732**. Both actors survive32/32 on that pair's matched timeout-source state.

These are incoming policies captured **before** the optimizer transaction for their named rollout update (i.e., after the preceding update), not outgoing policies silently relabeled. The original source interactions are831121 and950028 respectively. This is an already-learned recovery response becoming less reliable at a later policy version on a selected state; it is not a model-based controller supplied to the learner.

The highlighted pair was selected for interpretation **after** the full cohort's results. It is not an independent predeclared primary hypothesis. The intervening58 updates are not localized, monotonic deterioration is not established, and a critic/gradient/momentum mechanism is not yet identified.

All contrary outcomes remain. Pair7's timeout-source actor loses2/32 successes on the timeout-source state, and several failure-source states defeat both tested actors. The new data do not imply every near-matched failure is easily corrected by swapping policies or that all recovery weaknesses are caused by forgetting.

## Exact historical reconstruction and gate repair

Hosted run37796358018 matched all six actor/critic/reward-trace checkpoints and the complete512-batch checksum, then failed the episode-ledger comparison. The live environment stores episode tuples; JSON deserializes them as lists. Direct Python container equality rejected the format rather than an observed numeric discrepancy.

The corrected comparator normalizes tuple/list containers only. Numeric values and row order remain exact; tests reject even a one-representable-float reward change, missing episodes/environments, reordered entries, changed flags, nonfinite values and boolean counters. No tolerance or source-data gate was relaxed.

A fresh local reconstruction from random initialization completed all1,048,576 historical interactions in the pinned Python3.11.14/NumPy2.2.6/Torch2.8.0+cpu/SB32.9.0/Gymnasium1.3.0 runtime on AMD EPYC9V74. All six archived model/reward-trace checkpoints, all512 diagnostics, the exact numeric episode ledger, full fitting/physical-packet checksum and selected-state checksum passed. All32 recovered incoming actor/critic arrays also match the previously retained hosted-run snapshots. Only after these gates passed was the cross-over executed.

Bindings:

- Original seed201 artifact11504926318, SHA256 `11d9efaeb4225ef70449f7672bc72accc0a0001bffcf3d2f31a814ba25c2b4a2`.
- Canonical checksum across all512 fitting batches and1,048,576 physical packets: `76e003ff39029b70faabe40c9f85aafba8880fb2c86500576637a2b0e063d0f9`.
- Selected state/noisy-observation checksum: `d73d4a3f5142183f51352469eb220093c86abcca4c5eb948da99f487a8da5af1`.
- Exact executed repaired `replay.py` Git blob: `501768004980832201e9ecdbd097132f192f0480`.
- Original recovery evidence SHA256: `ff570b87dee8f0b7f502a57c62fa6fe48db60e7584c92a96e651c455cda5d412`.

Four of the32 source episodes cross an original rollout boundary: pair1's failure and timeout, pair10's timeout, and pair11's failure. The historical ending label may therefore involve more than one policy. The registered cross-over freezes the actor present at the source episode's **start**; it does not replay the later training-policy switch or claim that one actor alone produced that original ending.

## Independent verification

The read-only verifier checks all **2048 trajectories /1,718,073 transitions**, exact sequential-float32 native reward arithmetic, stopped-return scores, first physical failures and absence of post-terminal padding, recorded source-state/actor identities, bounded observation noise, force mapping and regenerated Gaussian innovations. Independent float64 neural inference reconstructs commands with maximum normalized error **4.063e-7**, below the inherited3e-6 threshold. Early25/50-tick state metrics and the100-tick centring window are retained per case. The braking proxy uses commanded force and pre-action velocity, not the disturbed applied force or an optimality oracle.

All **13 trajectory corruption/boundary tests plus9 episode-ledger tests pass**. A genuine final-tick physical failure is correctly distinguished from a timeout. The verifier runs no Torch, native simulation, optimizer or pickle loading, and does not independently integrate the nonlinear plant or regenerate every hidden native disturbance draw.

One verifier setup error is preserved: Python3.13 built-in `sum` differed from the measurement runtime's sequential accumulation on the genuine preflight fixture. Explicit sequential addition repaired exact accounting before any main cross-over outcomes. No numeric tolerance or measured reward was changed. A preparer source-path assertion failed before simulator invocation and was corrected to the actual archived location. Each main pair then ran once.

## Costs and evidence

This continuation used1,048,576 historical training transitions,630,280 original checkpoint evaluation transitions,8192 recorder-control transitions,8192 transitions across both sides of the original/additive ABI comparison,960 identical-policy control transitions and1,718,073 cross-over transitions: **3,414,273 native transitions total**. Historical/control optimization used165120 Adam transactions; cross-over optimization used **zero**. Costs of earlier failed hosted runs are separate. There is no new corrective candidate or efficiency claim.

Complete incremental evidence:

`ppo_historical_crossover_evidence_20261008.zip`

- Size: **78,461,360 bytes**.
- SHA256: **`87421cfc637f2002e6a19d826ebe168b7a88332378616107ae662ff0cfb2c268`**.
- Fresh extraction checks **127 payload hashes**, validates selected-state bindings, regenerates the complete `VERIFIED.json` byte-for-byte, and passes all22 tests.
- Includes exact executed measurement/replay/verification code, policies, complete new trajectories, source identities, numerical results, genuine fixtures and failure logs.
- Offline reproduction: `OPENBLAS_NUM_THREADS=1 python REPLAY.py` with Python, NumPy and SciPy. Native regeneration additionally needs the separately pinned runtime and original recovery-evidence archive; runtime installers and prior full143MB history are not bundled.

## What comes next

The literature review's separation of empirical gradient quality from surrogate improvement is now directly actionable: localize the intervening update(s) in a selected earlier/later PPO retention witness, then compare the actual parameter displacement's sampled advantage/surrogate direction with independent same-state recovery returns. Routine balancing and early recovery must be reported separately. The present cross-over establishes policy-version effects, **not** the mechanism of a particular harmful optimizer transaction.

No new experiment is launched by this report. No production/default/master/protected-source change, LQR label, runtime fallback, deployed-policy change or merge. Issues33/35 remain open.
