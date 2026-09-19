# Late actor learning-rate decay: completed negative result

September 18, 2026, America/Toronto (execution timestamps are September 19 UTC). Issues #35 / #33. Production baseline is PR38 `4739f370558b9443708c920ac30614f86e3c07bb`.

## Decision

**Late actor-only learning-rate decay did not qualify a robust controller and is not promoted.** It reduces typical measured policy change, but final deterministic and discounted stochastic returns regress on three of four seeds. Nominal five-minute completions fall from107/128 to98/128, and stochastic five-minute completions from104/128 to79/128. Recovery increases only64/128 to68/128, with two improving and two regressing seeds. Every final policy fails the unchanged necessary robustness screen.

A supplementary frozen-policy analysis shows another concrete limitation: nominal local stability does not ensure centred operation. Every final policy in both arms has a locally stable upright resting point, but none of those stable resting points is within0.5m of the centre. The recorded final nominal survivors likewise never satisfy the centred/upright condition throughout their last ten seconds. This is not a proof that off-centre equilibria explain every failure; the decay candidate also reintroduces many angle-limit failures.

Production code/defaults, PR38, master and deployed assets are unchanged. No merge, selected deployment checkpoint, supplied controller, new held-out qualification or hardware claim follows.

## Frozen intervention and execution

[Protocol recorded before execution](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5738281263). Audit branch: `audit/ppo-late-actor-decay-20260918`. Actual execution: **`3412d3d6704fd56625750aaa19c92d6ab34051bc`**, [run **35413206597**](https://github.com/yongkyuns/RustRobotics/actions/runs/35413206597). All nine jobs completed successfully on the first attempt: preflight plus eight full-budget runs. No failed/dropped training outcome or favorable retry occurred. Execution success is not controller-quality acceptance.

Two arms, each with exposed seeds201–204, start from identical per-seed random actor/critic weights and persist through checkpoints0/1,048,576/4,194,304/16,777,216. Adam moments, live environment, partial episodes and training RNG streams persist; weights are never reloaded for resumption.

- Constant arm: actor and critic learning rates remain3e-4.
- Actor-decay arm: actor rate remains3e-4 through4,194,304 collected transitions. It then decreases linearly to3e-5 at16,777,216, with that positive floor thereafter. Critic rate stays3e-4.

The actor factor at collected step s is1 before or at4,194,304; afterward it is `max(0.1, 1 - 0.9*(s-4194304)/(16777216-4194304))`. The4M boundary is explicitly an exposed-development choice motivated by the prior regression, not a blinded confirmation choice.

Both arms retain the actual corrected nonlinear Rust plant, existing2x64ReLU networks, one512-step rollout, minibatch128, four ordinary optimization epochs, epsilon1e-5, gamma0.99/lambda0.95, fixed latent exploration std0.1, and unchanged losses/rewards/noise/resets. No extra critic epochs, replay buffer, schedule-dependent resampling, action modification, gradient clipping, rollback or update acceptance test was introduced.

Each run uses16,777,216 real training interactions,32,768 policy refreshes,524,288 actor and524,288 critic Adam steps, and67,108,864 gradient-sample visits per network. Equal budgets do not make a speed benchmark. The two temporary source hooks replace only the actor-rate argument and record selected indices; a private module wraps the original optimizer for read-only diagnostics. No second PPO implementation is substituted. The critic receives its unchanged configured rate.

Training is actual Rust/Burn, Rust1.98.1/Burn0.20.1 with the committed lockfile. Python3.11/Torch2.8.0+cpu/NumPy2.2.6/Gymnasium1.3.0/SB32.9.0 provide the unchanged read-only short evaluator. The exact native robustness binary and evaluator source are reused from the hash-verified preceding build, artifact10571679369, ZIP SHA256 `ab471447c13df0ca7db3ecc01e09cb3ee705cd8e8b1c759512adde4a0dc25af6`.

## Unchanged evaluation and all fixed checkpoints

Each seed/checkpoint has32 noisy deterministic-policy10s and64 noisy stochastic-policy15s short episodes. At1M/4M/16M, the native portable-inference screen retains224 episodes:32 nominal deterministic and32 stochastic300s,32 outward-recovery60s, six one-at-a-time +/-10% physical shifts with16 deterministic60s trials each,16 doubled-noise trials and16 trials with20ms action delay. Failed episodes end immediately without auto-reset masking. The same panels and thresholds are used in both arms and the preceding experiment.

The necessary final screen remains zero failures for EACH final policy on nominal deterministic300s and EVERY deterministic stress panel. Even a pass would be a finite development result, not fresh-training confirmation or hardware safety. Short and native-long panels have different keys/inference paths: compare within columns, not as identical episodes with different caps.

| Budget per policy | Arm | Short deterministic /128 | Short stochastic /256 | Nominal300s deterministic /128 | Recovery60s /128 |
|---:|---|---:|---:|---:|---:|
|1,048,576|Both|90|184|97|72|
|4,194,304|Both|122|228|114|96|
|16,777,216|Constant|102|208|107|64|
|16,777,216|Actor decay|90|183|98|68|

All saved actor/critic weights and read-only diagnostics match between arms through4M. Consequently, the intervention does not retrospectively alter early learning. Final results are fixed endpoints, not selected best checkpoints.

## Every final training seed

Each arrow is constant -> actor decay. Completion columns count nominal deterministic300s and outward-recovery60s respectively.

| Seed | Deterministic mean return | Stochastic discounted mean | Nominal300s /32 | Recovery /32 |
|---|---|---|---|---|
|201|747.594444 ->428.739390|87.573407 ->85.245901|28 ->16|16 ->8|
|202|352.410680 ->552.919743|80.198016 ->81.222916|17 ->28|8 ->24|
|203|868.881325 ->568.349979|83.549125 ->79.296277|32 ->26|24 ->16|
|204|799.465432 ->579.948215|82.624481 ->80.934222|30 ->28|16 ->20|

Only seed202 improves both return scores. Pooled final deterministic return is692.087970 constant versus532.489332 decay; discounted stochastic mean is83.486257 versus81.674829. The descriptive paired99% Student interval across four training-seed differences is[-871.958985,+552.761708] around deterministic effect-159.598638, and[-8.184514,+4.561657] around stochastic effect-1.811428. These broad n4 intervals cross zero; they neither establish equivalence nor justify promoting the candidate. Pooled episode counts are not independent trained-agent counts.

Final stress completions, constant -> decay: nominal deterministic107->98/128; nominal stochastic104->79/128; recovery64->68/128; length minus10%55->50/64 and plus10%54->50/64; either cart-mass shift55->50/64; pole-mass minus10%56->50/64 and plus10%55->50/64; doubled noise56->40/64; delay55->50/64. Every final policy fails the full necessary screen.

## Smaller typical policy changes did not ensure better control

All32,768 updates per run retain actual actor rate/count and empirical old-rollout pre/post conditional Gaussian KL, maximum state KL, scalar clipped-surrogate values and clipping fraction. These values never select/reject/roll back an update. The Gaussian KL uses the fixed latent variance and averages over the collected states, not all possible states or future trajectories.

Across the98,304 post4M updates in each arm, mean rollout-average KL decreases0.005133621 ->0.004526035. Each seed's median decreases. In the final quarter, mean KL decreases0.005081284 ->0.003697785. However, the largest post4M rollout-average KL across seeds is0.432909 constant versus0.557800 decay; smaller configured rates do not provide a hard KL bound. Later state distributions differ, so these statistics are descriptive rather than an isolated causal explanation of outcome changes.

Updates with the read-only scalar clipped surrogate decreasing by more than1e-6 increase1,668/98,304 ->3,396/98,304. This diagnostic uses recorded f32 scalar densities promoted to f64 and is not substituted for the tensor objective used in training. It shows why neither a lower rate nor a smaller typical KL should be equated with uniformly better batch optimization or true return.

The failure mode also changes. Constant final nominal deterministic failures are21 position-only; decay failures are28 angle-only plus2 position-only. Nominal stochastic failures change24 position-only ->46 angle plus3 position. Recovery changes64 position-only ->44 angle plus16 position. Therefore it would be incorrect to describe all candidate failures as cart drift or claim rate decay merely preserves the old controller.

## Supplementary local-rest-point diagnosis

This analysis was added after the primary protocol was frozen. It is post-hoc diagnosis, not a changed acceptance rule or training intervention. It enumerates the frozen ReLU actor's piecewise-affine output along `[x,0,0,0]`, solves every zero-force upright rest inside the track, and linearizes sampled feedback with force held over the RK4 step. Calculations use f64 on stored f32 weights and nominal noiseless parameters. No simulator continuation or policy training is performed.

Every final actor has at least one smooth locally asymptotically stable upright rest. However, ALL such stable rests are outside the central[-0.5,+0.5]m region:

| Seed | Constant stable upright x, metres | Decay stable upright x, metres |
|---|---|---|
|201|-0.544222|+0.951568|
|202|-1.729481|-1.717986, +1.573498|
|203|-0.917878|+1.748667|
|204|-0.801103|-1.316953, +2.220085|

For example, the unchanged seed202 stable rest moves from-0.725945m at4M to-1.729481m at16M. For decayed seed204, the+2.220085m stable rest is only0.179915m from the+2.4m track boundary; its sampled closed-loop spectral radius is0.996680. A stable rest near a rail is not a robust operating envelope. These calculations do not prove which equilibrium attracts every initial condition, the basin size, behavior under noise, or a global/hardware guarantee.

The independently recorded centring statistic supports the broader concern without inferring a trajectory from a local eigenvalue. At4M,85/114 nominal deterministic survivors remain withinabs(x)<=0.5m andabs(angle)<=0.1rad throughout their last10s. At16M, that count is0/107 constant and0/98 decay. Stochastic survivors likewise go from85/114 to0/104 and0/79. Failure and survival remain distinct from centring.

The actual-root feedback Jacobians also match independent central finite differences to maximum3.171e-10. Five additional unit controls verify axis partitions, roots, input gradients, flat zero-force handling and held-input RK4 linearization. No supplied control law or artificial gradient enters any learner.

## Verification and limitations

First-attempt preflight passes strict Clippy, **78 native unit/ABI/audit tests**,7 ordinary balancing controls,9 ordinary learning/evaluator controls, formatting/build checks and protected-source restoration. Two separately ignored heavy historical integration tests were not invoked. Controls cover unchanged baseline/next-rollout parity, the positive rate floor, first decay boundary, same-data critic and shuffle behavior, nonzero actor updates, scope cleanup after failure, grouping and evaluation isolation. All original pre4M actor/critic checkpoint bytes and logged diagnostics are identical between arms. Every constant-arm actor/critic checkpoint reproduces the preceding long experiment, including16M.

All expected-unchanged native robustness outcome records reproduce the preceding run exactly. Short evaluation records reproduce except constant seed203 and the predecay checkpoints of decay seed201: maximum undiscounted-return differences0.000786245 and0.002089422, respectively, with unchanged durations/endings. Maximum corresponding discounted differences8.0675e-5 and3.0388e-5. These floating differences are retained without reruns; their platform cause is not uniquely attributed and no portable bitwise guarantee is claimed.

Independent offline analysis validates9 current archive digests and409 original payload-member hashes,262,144 training/diagnostic update records,32 regular actor/critic checkpoint pairs,3,072 short outcomes and5,376 robustness outcomes. The48 selected rollout batches contain24,576 rows with exact minibatch-epoch coverage; independent KL reconstruction error is at most1.041e-17 and scalar surrogate error1.215e-8, with no clipping-classification disagreement. Sparse predetermined traces reconstruct nonlinear transitions within3.305e-7, rewards within1.814e-7, and deterministic actions within3.692e-6. Sparse traces do not independently replay every unexported step, whole-episode reward sum, random draw, gradient or optimizer moment. Weight snapshots are not exact-resume checkpoints.

**31 independent analyzer/local-linearization controls pass**:26 general evidence/diagnostic tests plus5 supplementary local checks. The native executable is never loaded by offline replay. Main training/evaluation used the pinned CI runtime; offline analysis uses Python3.13.5/NumPy2.3.5/SciPy1.17.0. No extra local simulator/training interactions were used for the supplementary calculations.

Main costs: **134,217,728 training interactions;2,627,521 short-evaluation interactions;55,514,176 robustness-evaluation interactions;4,194,304 actor and4,194,304 critic Adam transactions;536,870,912 gradient-sample visits per network.** Preflight/test work is additional and not all aggregate-instrumented. No speed, equal-wall-time or all-inclusive compute claim is made.

## Evidence and disposition

The self-contained conversation bundle is `rustrobotics-ppo-late-actor-decay-evidence.zip`. It retains all nine unchanged current raw ZIPs, nested preceding raw evidence, exact executed source/workflow/binaries, every fixed checkpoint/outcome, artifact ID/digest maps, analysis/tests, supplementary local results and offline replay instructions. Retrieve it by name rather than assuming a previous runtime path. No font or unrelated user file is included.

**Do not promote actor-rate decay as the fix.** The next useful boundary is to test whether value/advantage updates push previously centred policies toward these off-centre rests, using the corrected nonlinear task and keeping recovery/angle failures visible. That attribution and fresh held-out training qualification remain unperformed here. The experiment does not uniquely explain every remaining failure, and PR38's integration correction remains separate from learned-controller robustness.
