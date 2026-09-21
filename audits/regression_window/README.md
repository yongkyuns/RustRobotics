# Historical recovery-regression window — native preflight passed, replay pending

Registered in issue35 comment5754579087 before new window outcomes. Executed source **a3276e1efa3071d381480bc61f1abe1ebd818a54**, run **35554192268**, branch `audit/ppo-regression-window-20260920`. This is an observation-only diagnostic, not another controller/hyperparameter candidate.

## Current status

Native preflight passed on its first attempt: **111 unit/audit tests**, **7 ordinary balancing controls**, **9 ordinary learning controls**, strict Clippy, formatting and protected-source restoration. Ten heavy audit endpoints were ignored in the ordinary unit pass; the window endpoint is invoked separately by the two replay jobs. Two historical heavy integration endpoints and the production platform/browser matrix were not rerun.

The most recent child-job check shows seed41006 replay executing and seed41004 queued. Neither completed replay artifact has been independently verified, and no per-update drop or independent before/after return result is claimed. A passing native harness is not evidence of improved control. Documentation-only commits do not launch duplicate measurements.

## Exact historical replay

Reconstruct long-support/gamma0.995 seed41006, where recovery was lost, and seed41004, where final recovery improved. Both are deliberately selected exposed histories, not a fresh cohort. Replay each from the original random actor/critic initialization through update4224 with the same existing recipe:4096 ordinary-reset recovery-union updates, then the existing half-outward-start mixture,1024-step sampled support,lambda1,global normalization and unchanged Adam/plant/reward/noise settings.

The historical archives are comparison inputs, never loaded as warm starts. Every recorded update through4224 must match its original history, and both networks must match checkpoints0/1024/4096/4128/4224. A mismatch stops the measurement and is retained. Adam moments, environment, partial episode and RNG histories remain live during replay. Portable optimizer-resume files are not exported, so saved actor/critic files are not presented as complete training checkpoints.

## What the missing interval captures

At every update4129–4224, retain incoming actor/critic, main rollout, sampled futures, all supplemental support trajectories, raw/normalized advantages and target values, and all16 original optimizer transactions with indices/losses/actor/critic outputs. Save every post-update actor/critic. No fitting sample, controller output, loss, optimizer equation or sampling rule is changed.

Frozen snapshots at4128 and after all96 updates are evaluated on the same three original20.48-second panels,64 cases each: nominal mean-action, nominal stochastic and outward stochastic. All historical episode fields must reproduce at4128/4224. Evaluation uses the old offset0x10000000 and is verified not to mutate the learner. Full traces are fixed replication0 at the two window endpoints; every other case retains aggregate outcomes. The repeated cases localize changes, but cannot justify independent statistical significance for an update selected from them.

## Independent confirmation after selection

For EACH seed, select one transaction using the largest consecutive decline in outward completion count, with earliest-update tie-breaking. Retain all96 changes, not only the selected one. If no decline occurs, retain the largest zero/beneficial change rather than force a harmful case. Even the improving history may have a locally harmful update; that is not an error or an excluded result.

Evaluate the selected transaction's frozen before/after policies on256 paired cases in each of the three short panels using independent offset0x20000000. Neither those draws nor their outcomes enter training or update selection. Save all outcomes and fixed first-replication traces. The two-sided99% Student intervals use a Bonferroni family of12 contrasts:2 histories x3 panels x completion/discounted return. They estimate conditional effects for these fixed selected policies, not reliability across a population of trained agents. No production checkpoint is selected and no actual training is rolled back.

A harmful update could thereby be localized and checked on independent cases. This would not yet establish which advantage component, critic estimate or particular action caused the loss. The replay does not contain an unregistered normalization/learning-rate/reward intervention.

## Verification already completed locally

Actual build artifact10619877204, SHA256 `969e49d56acdd71bfefdede8cbac4b8b4a79d6dfb935fea46d90ccb5319dd876`, was downloaded and all119 payload hashes verified. The55 retained protected Rust/Cargo files are byte-identical to the original return-support build. Removing the diagnostic-child declaration recovers the prepared parent collector/optimizer source exactly. The preparer only attaches that child and a localized lint allowance for its explicit diagnostic I/O arguments; no existing collection or optimizer body is replaced.

The prepared offline suite passes **109 tests**, including inherited target/physics/derivative checks and new chronology,selection,tie,missing-data,random-domain,paired-effect and failure-accounting controls. Running the updated batch reconstruction in pieces on the OLD41004/41006 archived updates reproduces the inherited numerical results exactly. These compatibility tests add no simulator or training interactions and do not supply new window measurements.

With only the build ZIP present, the verifier explicitly reports both replay artifacts missing, exits2 and emits no controller scores. It is not reporting a failed controller or completed experiment. The prepared verifier checks source hashes, historical update/checkpoint/endpoint reproduction, every window target and optimizer transaction, exact selection and independent-pair identities, all exported evaluation physics/rewards and full-batch surrogate/KL/trajectory-source accounting. It does not regenerate unexported Adam moments, gradients or missing trajectories.

## Bounds and disposition

Per replayed history:4224 original updates,2,162,688 main interactions,36,765,696 supplemental interactions and at most17,301,504 cutoff-future interactions. Historical-window evaluation has18,624 episodes and selected-update independent evaluation1,536, separately counted. Generic preflight/control interactions are additional. This is reconstruction of two existing histories, not new successful candidate learning, an efficiency result or hardware qualification.

Production gamma0.99/lambda0.95,reward,normalization,weights,PR38,master,deployment and fallback remain unchanged. No merge or controller adoption. The preceding support1024 rejection remains in force. Current progress establishes a working native diagnostic and submitted replay; the damaging transaction has not yet been measured and verified.