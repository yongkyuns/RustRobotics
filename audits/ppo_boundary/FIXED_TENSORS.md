# Fixed-tensor PPO replay and one-gate numerical attribution

Date: 2026-09-18. Parent learning issues: #35 / #33. Separate integration correction: PR #37 / issue #36.

## Finding

The requested **exact-derived-tensor replay through actual SB3 PPO.train was executed**, not substituted with an independently written PPO optimizer. Three of four primary cases pass the unchanged thresholds. Seed204 retains a critic parameter discrepancy, which a one-event diagnostic isolates to numerical sensitivity at a ReLU threshold on the reproduced trajectory. This is not evidence that ReLU caused the long-run balancing failure, and no artificial gate operation enters production.

The original primary failure remains FAIL in its archive and workflow. The later intervention is a conditional causal diagnostic, not a replacement acceptance result. No new physical training transitions, fresh training seeds, reward changes, privileged observations or production hyperparameter changes were used.

## Frozen inputs and real reference boundary

Native traces are the unchanged four cases in artifact10560032269/run35374729957, ZIP SHA256 `8e6948cf6cd032ea8777096e92010f5e592507df6ab414bb19b713df00e3465e`. Their production-source baseline is `ff4934a636974727632355371f575e651dc44a61`. Archive/member hashes were verified before replay.

The primary replay executed commit `6e2740774134edc50f943e757ab2359c32093799`, run35378018991. Software: SB3 2.9.0, Torch 2.8.0+cpu, NumPy 2.2.6, Gymnasium 1.3.0, Python 3.11.16. Both actor and critic use the actual exported native Adam configuration, including epsilon1e-5; the older epsilon1e-8 mismatch is retained as a negative control.

The reference consumes exact native observations, latent actions, stored squashed old log probabilities, returns, normalized advantages, and minibatch indices. Every yielded tensor is checked byte-for-byte. Actor/critic parameters are imported only initially in the primary free-running mode. Optimizer objects, parameter identities and every per-parameter Adam step counter remain persistent and are checked. No target recalculation, extra normalization, value clipping, target-KL stop or gradient clipping is introduced.

A documented distribution adapter evaluates the squashed action density at stored latent actions; this allows the old log probabilities to remain byte-identical rather than converting them with rounded Jacobians. Its Gaussian is Torch Normal with the explicitly fixed native scale; its action-only Jacobian is checked against a separate f64 density and an analytic mean derivative, including saturated-force latents. Entropy weight remains zero. `PPO.train` and `ActorCriticPolicy.evaluate_actions` are the unmodified installed methods; their source hashes are retained.

A separate conditional mode restores the recorded native incoming weights before each transaction while keeping Torch Adam history. It is local-update consistency evidence, not free-running equivalence.

## Primary results — original thresholds unchanged

Parameter maximum absolute threshold:2e-4. Loss absolute/relative thresholds:5e-4/1e-4. All input tensors, epoch coverage and transaction counts must match exactly. The earlier collection/inference/target thresholds are not weakened; this test supplies their recorded outputs rather than rerunning collection.

| Case | Updates / paired actor-critic transactions | Primary result | Maximum actor error | Maximum critic error |
|---|---:|---|---:|---:|
| Default seed201, 1x512 | 8 /128 | PASS | 9.98970e-5 | 3.77297e-5 |
| Default seed204, 1x512 | 8 /128 | FAIL, critic only | 1.30841e-4 | 3.64434e-4 |
| True-terminal control, 3x7 | 4 /24 | PASS | 9.43739e-8 | 1.07819e-7 |
| Timeout control, 3x7 | 4 /24 | PASS | 7.13730e-8 | 1.22908e-7 |

Seed204 first exceeds the critic threshold at update4/minibatch12. Its policy/value losses and actor parameters remain within the original thresholds. All four conditional modes pass; their maximum actor error is4.84e-8 and maximum critic error8.52e-8 over304 transactions. Wrong-epsilon control fails at update1/minibatch1. Reset-Adam control fails at update2/minibatch1. Nothing was dropped, promoted or threshold-adjusted to make the primary workflow green.

## Specific numerical trigger and intervention

Post-hoc selection: the first stored seed204 critic-gradient difference exceeding1e-3 is paired transaction33, update3/minibatch1. This event was recorded before the follow-up was run. Native versus free-reference incoming parameters differ by at most1.490116e-7.

On the reproduced trajectory exactly one second hidden-layer ReLU changes its active/inactive side: minibatch row55, original rollout row290, hidden unit9 (zero-based row/unit indexing). The reconstructed native-state preactivation is+1.639128e-7; the free-reference preactivation is-3.501773e-7. Forward weighted value loss is **76.25802612304688 in both branches**, but their maximum gradient difference is0.0060356855.

Replacing only that backward gate with the native-state gate, without changing the forward loss, reduces the local gradient discrepancy to4.2915344e-6. The local Torch2.10 probe and the successful pinned Torch2.8 follow-up both reproduce the two archived gradient arrays exactly at this event.

The successful pinned follow-up is commit `05f145269e26df5ed55033035d5f346ea2a99ac0`, run35379027728. A forward hook changes only the identified derivative at transaction33, once. It does not reload parameters, reset Adam, change data, change forward activations or apply later gate interventions. The first32 parameter AND gradient transactions replay bit-for-bit, and the actor path is bit-for-bit unchanged throughout all128 transactions.

| Seed204 diagnostic | Maximum critic parameter discrepancy over all128 transactions |
|---|---:|
| Original exact-tensor free reference | 3.6443366e-4 |
| One backward gate controlled at transaction33 | 3.5954384e-7 |

This roughly1,000-fold reduction isolates that threshold crossing as the trigger of the recorded critic-path discrepancy in this conditional replay. It does not establish that replacing ReLU, changing its derivative, or imposing this gate would improve actual PPO learning. The artificial derivative control is never a supported training mode.

### Retained unsuccessful follow-up

Earlier run35378542604 at `1b556214d74cb94bd9a21913e149bcacc3faecc3` failed its exact-gradient replay assertion before the one-event continuation. The initial harness wrote the detailed result only after asserting, so the failed attempt retained its traceback/source but not the measured discrepancy. This failed attempt is retained, not called a setup success or erased.

The follow-up code was changed to write/print measurements first and enforce the **same exact checks at the end**. Its subsequent run succeeds. The reason the earlier pinned attempt did not exactly reproduce the gradients is **not established**; no hardware, backend or cross-run bitwise guarantee is inferred. The numerical attribution above is explicitly conditional on the successful reproduced trajectory, whose actual prefix and gradient arrays were independently verified. Passing later does not retroactively turn the earlier attempt into a pass.

## Independent checks and cost

The four source cases contain8,360 previously recorded native transitions; this work performs **zero new simulator transitions**. Primary free modes use304 paired actor/critic optimizer transactions, conditional modes304, and the two negative controls256. The completed one-event continuation adds128. These repeats are not independent trained agents or long-run learning trials. Small analytic/gradient controls are separate from those counts.

An independent NumPy verifier checks all28 primary artifact members, all47 nested native artifact members, all exact input-field hashes, per-epoch index coverage, and every stored actor/critic parameter-error value/result against the native snapshots. The stored losses come from actual SB3 execution; that offline verifier does not pretend to re-execute SB3. Counterfactual prefix/actor invariance is also checked directly from archived arrays, not merely from log assertions.

## Evidence

| Artifact | ID | ZIP SHA256 |
|---|---:|---|
| Fixed-tensor primary, including failing case |10560769288|`53302b0ec6127e372fbc0836636d3403c9041251bed2cf0e9c72841ea6842b3b`|
| Earlier failed gate follow-up |10561740615|`e39119633caa59447d3566a26539577289497efe6e8fcd6a398524634614cc05`|
| Completed gate follow-up |10561696542|`d99ff3f1c687c4a67e9e0fe07a1ef96da44b4c696dd8dd25e312772f9c3242e4`|
| Resolver output with unrelated edge change, not applied |10560991224|`8eb34efe9176b52d4b4275b169e248ef9ba918494d42ea7dd3ebeb9b2e881b84`|
| Validated minimal rustls-only candidate |10561286567|`d14e08147cf2b075a40436685fd49592c5002aa509d0052d853f321ca9ae73e8`|

The self-contained conversation bundle is named `rustrobotics-ppo-fixed-tensor-evidence.zip`. It includes unchanged raw archives, original native evidence nested inside the primary archive, scripts, summaries, verification and reproduction instructions. Retrieve it by name from conversation files rather than assuming a previous runtime's sandbox path.

## Separate PR37 security blocker

The normal resolver's first candidate also changed tempfile's getrandom edge; that unrelated change was rejected. The minimal patch changes only rustls0.23.44 to0.23.45 and its checksum. Locked cargo metadata and cargo-audit pass with zero reported vulnerabilities before commit; existing unmaintained-package warnings remain visible.

Commit `d3f59f9bf38d4038ba7c5008e3a3b41c15e99835` is now on `fix/ppo-shared-policy-rollouts` / PR37. Its diff is two changed lines in Cargo.lock, with no PPO source/default change. Exact-head dependency audit35378873579 and configured PPO-learning35378873566 pass at this report checkpoint; Rust/platform35378873556 and numerical35378873559 are still executing. Final status belongs to the later PR verification comment. PR37 stays draft and unmerged.

## Interpretation and next learning work

The fixed-data loss/update path is locally consistent on these tested cases, and a specific nonlinear numerical threshold accounts for the remaining reproduced critic trajectory drift. It is not appropriate to treat every cross-library floating trajectory mismatch as proof of an incorrect PPO equation. Nor does this establish reliable learning, late harmful-update equivalence, or calibrated action credit.

The next learning-level comparison should correct C3's epsilon mismatch, hold all other settings/budgets/checkpoints fixed, and retain all exposed seeds and failures. An epsilon-only from-scratch ablation is required before recommending either epsilon as a correction. No additional tuning or held-out qualification was performed here; #35 and #33 remain open.
