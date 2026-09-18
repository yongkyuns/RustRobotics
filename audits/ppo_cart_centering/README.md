# C5 — cart-centering versus pole stabilization

2026-09-18. Investigation of #35; sustained-balancing acceptance remains #33. No production training equation, default, reward, noise, reset distribution, or controller was changed.

## Result

The cart-centering lead is supported for part of the Rust failure trade-off, but not as a universal root cause. Eight of sixteen final C4 policies have no locally asymptotically stable upright rest equilibrium inside the track in the nominal mathematical model. Exchanging the learned cart-dependent responses between the Rust epsilon variants makes completion worse in aggregate in both directions. The learned cart and pole responses cannot be treated as independent modules with a universally better cart component.

A particularly important result is objective-versus-survival disagreement: on the exposed panel, one exchange raises pooled discounted stochastic return from77.331 to77.977 while reducing15-second completions from98/256 to50/256. This is an actual frozen-policy comparison, not just a critic prediction. It does not establish a positive expected improvement across unseen training seeds.

The present reward already penalizes cart position. A separate analytical countercheck finds that the current reward/discount admit a locally stabilizing exact quadratic optimum. Therefore neither a missing centering penalty nor short discounting alone establishes the cause of the learned instability.

## Protocol and provenance

[Protocol recorded before trajectory interventions](https://github.com/yongkyuns/RustRobotics/issues/35#issuecomment-5735038229). Preliminary root scans were explicitly exploratory and preceded the exhaustive root calculation. The analytical quadratic-optimum countercheck was added after the main measurement and is labeled post-hoc; no data or score was used to fit its coefficients.

Executed commit: `70a211e47d96c6ccfc0dd7753c7389440ec38f88`. Successful [Actions run35385827285](https://github.com/yongkyuns/RustRobotics/actions/runs/35385827285). Artifact10563588646,148,219,123 bytes, SHA256 `c3ec6daa968df8c2b1b269a3c4ea19510031c84ef91a8e886a927d901a254a2a`.

All16 final C4 actor/critic checkpoints were reused: Rust and SB3 at epsilon1e-5/1e-8, seeds201–204. The native library is the unchanged C4 artifact10562966062 with SHA256 `17ebe5551c5cb1d610580775d3a9831423e9070f6de4257f5dc2c8135b983cd4`. Its source baseline is PR37 headd3f59f9. All17 input archive digests and657 manifest members were checked before running.

Main runtime: Python3.11.16, Torch2.8.0+cpu, NumPy2.2.6, actual compiled Rust plant/ABI. No training was performed. All actor inputs remain noisy observations; true state is recorded only for diagnostics. The existing C4 evaluation panels remain exposed, not held-out:32 deterministic-policy episodes capped at1000 steps and64 stochastic-policy episodes capped at1500, per case. Observation/action noise and disturbances remain enabled.

Main measurements comprise16 original frozen policies plus8 Rust cart-response exchanges:2,304 episodes and1,836,062 physical evaluation transitions (1,314,715 original-policy,521,347 exchange). A separate local Torch2.10/NumPy2.3.5 pilot reproduced one baseline case, consuming96 episodes and67,729 transitions. It was not substituted for the pinned main run. All training counts are zero.

## 1. Coupled equilibrium analysis

At upright rest the state is[x,0,0,0] and the required force is zero. For the actual policy20*tanh(z), this is exactly the condition z(x,0,0,0)=0. The saved ReLU network is piecewise affine along this axis.

The analysis enumerates all first- and second-layer activation breakpoints on the complete track interval[-2.4,2.4], rather than searching a coarse grid. Across16 policies it checks1,476 affine intervals and finds43 upright rest equilibria. Full local closed-loop eigenvalues are computed at those equilibria from A+B*du/ds. It does not call the origin an equilibrium when the policy applies nonzero force there.

| Final policy family | Policies with a locally stable upright rest point | Policies without one |
|---|---:|---:|
| Rust1e-5 |2/4|2/4|
| Rust1e-8 |1/4|3/4|
| SB3 1e-5 |2/4|2/4|
| SB3 1e-8 |3/4|1/4|

There are10 stable rest equilibria distributed across8 policies; some policies have several rest points. Stable rest positions need not be centered. At each policy's rest equilibrium closest to the origin, the angle/angular-velocity response submatrix is stable when cart coordinates are artificially held fixed. That does not stabilize the actual coupled cart-pole:9 of those16 full coupled linearizations are unstable.

For example, Rust1e-8/seed204 completed27/32 deterministic10-second episodes in C4, yet has no stable upright rest point on the track. Its closest rest point is x=0.2221m, with dominant local growth rate approximately0.1642/s, an e-folding time near6.1seconds. Finite-horizon survival is not a certificate of convergence or stability.

These are nominal noiseless local calculations using the saved float32 weights evaluated in float64 and the production model coefficients. They do not prove global instability, exclude stable limit cycles, certify stochastic robustness, or predict every episode ending. All computed equilibria are away from activation boundaries (minimum preactivation margin0.000121). Independent direct-network and central finite-difference checks reproduce every root and stability classification; maximum Jacobian discrepancy1.10e-8.

## 2. Controlled cart-response exchange

For each Rust seed, decompose the latent output algebraically:

z_pole(theta,omega)=z(0,0,theta,omega)

z_cart(x,v,theta,omega)=z(x,v,theta,omega)-z(0,0,theta,omega)

The diagnostic combines the pole response of one learned policy with the cart-dependent residual of the other, then applies the unchanged20*tanh action map. Both exchange directions and all four seeds are retained. This residual contains nonlinear interactions; it is not a physically independent controller block. No weights, optimizer, or training state are updated.

| Frozen Rust policy | 10s deterministic completions /128 | 15s stochastic completions /256 | Stochastic discounted return |
|---|---:|---:|---:|
| Original1e-5 |50|98|77.331|
| 1e-5 pole response +1e-8 cart residual |37|50|77.977|
| Original1e-8 |74|150|80.054|
| 1e-8 pole response +1e-5 cart residual |42|84|78.493|

The exchanges hurt aggregate completion in both directions, although individual seeds can improve. Therefore this is not evidence for copying a donor's cart residual into production.

Seed203 illustrates the slow-failure problem: keeping the1e-8 pole response but exchanging the cart residual reduces stochastic completions47/64 to0/64. Mean undiscounted return falls1052.900 to299.523, while discounted return falls only78.696 to77.824 (about1.1%). The reverse exchange also collapses this seed's completions39/64 to0/64. Neither direction is hidden.

The included plot shows stochastic episode0 of this seed: the original policy stays inside the track for15seconds; the exchanged response drifts to the negative rail in about5.6seconds. This episode was selected after measurement as the lowest indexed original-completion/exchanged-position-failure example, for illustration only; it did not select a training update or an acceptance score.

## 3. What the actual rewards and trajectories say

The existing nonterminal reward is1 -0.2*x^2 -0.02*v^2 -theta^2 -0.05*omega^2 -0.001*u^2, evaluated after the step; task termination overrides it with-10. Independent reconstruction of all observed rewards agrees within2.0e-7. Cart position is already penalized.

Rust's smaller-epsilon policy has57 stochastic position-only failures versus11 for the default. Of those57,31 keep mean absolute pole angle below0.1rad during their last second. Their median position-failure time is3.33seconds and median last-second mean absolute angle0.0948rad. This supports a real pole-regulation/cart-drift failure mode, but not every rail failure is a well-balanced pole: the remaining26 fail that angle criterion.

The reward decomposition also limits the hypothesis. For Rust, the smaller-epsilon policy increases discounted cart-position cost by2.651 per episode while lowering discounted pole-angle cost by0.433 and reducing the terminal/missing-future-reward contribution by4.579. Its aggregate discounted return consequently improves despite more rail failures.

For SB3, smaller epsilon instead lowers discounted position cost by0.645. Its worse discounted return is chiefly associated with an increased terminal/missing-future contribution of2.504, plus larger velocity/angular/action costs. Increased cart-position penalty is not a valid universal explanation for SB3's score regression. Full cost components and every per-seed exchange result are retained.

At dt0.01/gamma0.99, the discount multiplier is approximately0.366 after1second,0.00657 after5seconds, and0.0000432 after10seconds. Discounted reward has a0.690-second half-life. With lambda0.95, the raw GAE trace coefficients have a0.113-second half-life. These are not hard planning cutoffs: the critic can carry longer-horizon consequences. They explain why reliable future-value propagation matters, not why increasing gamma is automatically a fix.

### Analytical countercheck, not a supplied controller

The post-hoc check solves the discounted discrete quadratic control problem using the actual local dynamics, current gamma, and current reward weights. It preserves the fact that cost is charged on the next state, including the resulting state/input cross term. The mathematical optimum's closed-loop spectral radius is0.989425 (<1); the Riccati residual is4.21e-12.

Thus the current local reward/discount are compatible with stabilizing coupled feedback. This result excludes force saturation, noise and terminal/reset boundaries; it does not settle the full nonlinear stochastic task. No LQR actor was installed, supplied, or simulated. It is a countercheck against blaming discounting alone, not the proposed user solution.

## Verification, evidence and limits

The pinned run passed7 diagnostic controls and finished once, without a score-based retry. Its79-member manifest verifies. Offline checking independently reconstructs trajectory continuity, observation-noise bounds, commanded force mapping, ending labels, per-step reward components, discounted/undiscounted totals, and all2,304 unique records. The actual state-step consistency residual is below4.90e-7. Root enumeration is separately checked against direct network evaluation and finite differences.

All1,536 original-policy episodes reproduce C4's durations and ending labels exactly. Floating return values are not universally bitwise identical: maximum undiscounted difference0.0020833 and discounted difference0.0000417. Their precise cross-runtime/kernel cause was not established. These differences are retained, not used to retry policies or claim portable bitwise replay.

The raw archive contains every state/noisy observation/commanded force/reward/critic estimate, all original input archives, executed scripts, versions and logs. Offline analysis and the nominal quadratic countercheck use Python3.13.5/NumPy2.3.5/SciPy1.17.0, separately from pinned trajectory execution. Critic residuals against single realized continuations are not independent conditional value calibration; no new wrong-sign advantage claim is made from them.

The self-contained conversation bundle is `rustrobotics-ppo-cart-centering-evidence.zip`; it includes the unchanged raw Actions archive, per-case CSV/JSON, detailed verification scripts and the illustrative plot. Production code/defaults and master remain unchanged. PR37 remains separate and unmerged. #35/#33 are not closed by this diagnostic.

## Next correction target

Focus the next bounded correction on learning reliable coupled recovery from outward-moving, off-centre states: check future-value/action-credit accuracy there, and representative shared-policy trajectory coverage at a fixed data budget. Do not promote cart-residual swaps, a supplied controller, an epsilon retune, or a gamma sweep merely because this diagnostic exposes slow-mode instability. A supported from-scratch candidate still needs comparison retaining all seeds and fresh held-out #33 acceptance.