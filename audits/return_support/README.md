# Gamma 0.995 with longer sampled return support — COMPLETE

**The experiment completed; do not adopt the 1,024-step candidate.**

[Verified results at d8bf6183](https://github.com/yongkyuns/RustRobotics/blob/d8bf6183bf958cf5debcb563a7fdca08465a91db/audits/return_support/RESULTS.md).

The original workflow **35548063439**, executed source **0a4445f6c5ab83650133a696faaeb764fd5b6c26**, completed on its first attempt. All 17 jobs passed: native preflight and sixteen from-scratch training runs. No duplicate training or favorable retry was used in verification. This supersedes the previous in-progress status.

| Support, both training gamma 0.995 | Five-minute deterministic | Five-minute stochastic | Sixty-second outward |
|---|---:|---:|---:|
|512 control|507 / 512|510 / 512|450 / 512|
|1,024 candidate|494 / 512|500 / 512|455 / 512|

The five additional outward successes do not compensate for nominal losses. The registered outward-completion interval includes zero, nominal retention fails, and every candidate absolute long-panel screen fails. Seed 41004 improves outward completion from 51/64 to 63/64, while 41006 falls from 40/64 to 32/64 and loses nominal reliability. All 87 remaining candidate long-panel failures are position-limit failures. All survivors finish centred; failures remain in denominators.

## Frozen protocol

Registration **issue #35 comment 5753874883** precedes all new outcomes. The complete pre-results protocol/status is retained at [90c6569f](https://github.com/yongkyuns/RustRobotics/blob/90c6569fdf2395f1d1497a7b974ab31cc2248d3e/audits/return_support/README.md). No threshold or intervention changed after outcomes.

Both arms train from random actor/critic initialization with gamma 0.995, eight paired already-exposed seeds 41001–41008, and 4,096 ordinary-reset recovery-union updates followed by 512 half-outward supplemental-start updates. Complete Adam/environment/partial-episode/random histories persist. No learned weight initialization.

Only the sampled support length changes: four cutoff futures of up to 512 or 1,024 steps and eight supplemental streams of 576 or 1,088 transitions. Only the first 64 samples of each supplemental stream enter fitting. The main fitting batch remains 512; the union remains 1,024 rows; both arms use sixteen Adam transactions per network/update. True terminal masks prevent credit crossing resets. Noise, rewards, nonlinear plant, architecture, exploration, global normalization, experimental lambda 1, minibatch 256, four epochs, learning rate 0.0003 and Adam epsilon 1e-5 are unchanged.

All six short-support actor/critic checkpoint pairs, every update row after the arm label and final costs reproduce the preceding gamma995 archives exactly. Historical inputs are immutable comparison files, never warm starts. Both arms' initial fitting/physical data match; later targets and trajectories may diverge.

Evaluation uses fresh draws in fixed domain 0x10000000 and common score-gamma 0.99 for both arms. Every checkpoint retains 64 short nominal deterministic/stochastic and outward cases per history; final evaluation adds five-minute nominal and sixty-second outward panels. True failure stops each trial. Only update 4,608 determines advancement. Replication-zero full traces are retained at 4,096/final; other trials retain aggregate results. Exposed training seeds are not held-out or hardware qualification.

Development advancement requires a positive lower bound on outward-completion gain using eight paired-history two-sided 99% Student intervals adjusted over three long completion comparisons, no pooled nominal regression and at least 61/64 nominal completions for each candidate history. The separate absolute screen requires 507/512 pooled, 61/64 per history, three-panel-adjusted one-sided 99% history lower bound at least 95%, and at least 99% centring among survivors. Both screens fail.

The 1,024-step bootstrap coefficient falls from approximately 0.07681 to 0.00590, but longer support also changes sampled-target variance and later bootstrap states. This experiment rejects the tested correction; it does not prove all bootstrap or longer-horizon methods ineffective. Actual candidate sampling is approximately 1.835 times the control's, despite equal fitting and optimizer work.

## Completed verification

Native preflight passes 108 unit/audit tests, seven ordinary balancing controls, nine ordinary learning controls, Clippy, formatting and source restoration. Nine heavy audit endpoints are ignored in the normal unit pass; the current endpoint is explicitly invoked by measurement jobs. Two historical heavy integration endpoints and the production platform/browser matrix were not rerun.

The unchanged offline verifier passes all 63 tests. Checks cover all 17 published ZIP digests and 4,494 payload hashes, 224 bound historical inputs, 21,504 outcomes, 73,728 update rows, 96 regular actor/critic pairs, 319,488 detailed support rows and 768 recorded optimizer transactions. Targets and normalization reconstruct bit-for-bit. All 144 exported trajectories, 1,170,196 transitions, reconstruct actions, physics, rewards, endings and scores. Unexported gradients, Adam records, random draws and other full trajectories are not independently regenerated.

Fresh-unpack verification checks 35 outer payload hashes, reruns 63 tests and regenerates all five numerical reports byte-for-byte, including both failed decisions. Full evidence: `rustrobotics-return-support-evidence.zip`, 204,886,816 bytes, SHA256 `1955a53546f9fff5683c27a43df15ce45c5b0e65e101a1098124317d5c70d0af`. The detached verification receipt and analysis-only package retain final replay logs. Offline commands execute no native binary, simulator, training or network.

**Production gamma 0.99/lambda 0.95, rewards, normalization, learned policies, PR #38, master and deployment remain unchanged. No merge, wrapper or fallback adoption.** The study is complete with a negative adoption decision.