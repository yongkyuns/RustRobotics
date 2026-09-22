# Independent simulator cross-check of learned-baseline full-step actors

**22 September 2026. Diagnostic only. The registered native replay remains the required gate.**

Protocol was recorded before outcomes in issue #35 comments `5785029668` and
`5785034261`. This cross-check evaluates the four fixed seed41006/update4140
actors with a separately implemented NumPy simulator and an independent PCG64
random domain. It does not reproduce Rust `StdRng` bytes and does not replace the
native prefix/sham/unchanged-critic controls or the registered 6,144 native
records.

## Fixed actors

| Arm | SHA256 |
|---|---|
| incoming pre4140 | `b439aabece1fbc92be5062680f3e284f44783e96e31e8495d415d31455f7617d` |
| historical original update4140 | `c7de297286bb018e51c4578cf49e24476258579ca55fded3cb6c3ba8fe50c983` |
| learned-uniform full numerical step | `8c34ecd70a3030890ab16d2e228774794a24e2f9cfab2ce58570eb6b0e4d3c90` |
| learned-early-balanced full numerical step | `189783fb048789086448afd8eeb4b3da93dcc78c12cbf1beabf6a0a340bf1af3` |

No actor was selected or modified from evaluation outcomes.

## Independent evaluator

The implementation directly ports the production task equations from baseline
`4739f370558b9443708c920ac30614f86e3c07bb`: nonlinear RK4 cart-pole dynamics,
`dt=.01`, `l=2m`, cart/pole masses `1kg`, `g=9.81`, force limit `20N`, failure at
`|x|>2.4m` or `|theta|>.6rad`, production observation/action/disturbance noise,
and the exact reward formula. Actor snapshots are the existing 4→64→64→1 ReLU
f32 networks with latent standard deviation `.1` and command `20*tanh(z)`.

Randomness intentionally differs from Rust. The fixed root is `2026092218`; each
case uses NumPy `SeedSequence([root,panel_id,rep,stream_tag])` with independent
environment/action/initial-state streams. The same sampled initial condition,
observation-noise sequence, stochastic action innovations, action noise and
disturbance sequence are paired across all four actors. Disturbance magnitudes
are pre-generated even when the Bernoulli event is false; therefore this is a
fresh implementation with the same declared distributions, not an attempt at
Rust RNG stream parity.

Panels use 512 cases each, all capped at 2,048 steps: nominal deterministic,
nominal stochastic and outward stochastic. Outward states use one common sign
for x/v with `|x|~U(.8,1.2)`, `|v|~U(.4,.8)`, `theta~U(-.15,.15)`, and
`omega~U(-.3,.3)`.

## Results

| Arm | nominal det | nominal stochastic | outward stochastic | outward mean discounted return |
|---|---:|---:|---:|---:|
| Incoming | **509/512** | **508/512** | **415/512** | 57.55691 |
| Original update4140 | **509/512** | **509/512** | **382/512** | 57.12924 |
| Learned-uniform | **509/512** | **508/512** | **415/512** | 57.75496 |
| Learned-early-balanced | **509/512** | **508/512** | **412/512** | 57.79334 |

Every failure in this external domain is a position-limit failure; there are no
angle-only or combined failures.

The historical update therefore loses **33 outward completions** relative to the
incoming actor in this independent random implementation. The learned-uniform
update restores all 33 in aggregate, while the early-balanced update restores 30
and remains three completions below incoming.

### Paired discounted-return contrasts

Ordinary descriptive 99% intervals use mean ± `2.586*SE`, as registered.

| Candidate minus incoming | nominal det | nominal stochastic | outward stochastic |
|---|---:|---:|---:|
| Learned-uniform | +0.09950 `[+0.07002,+0.12898]` | +0.10346 `[+0.07124,+0.13568]` | +0.19805 `[+0.17802,+0.21809]` |
| Learned-early-balanced | +0.09075 `[+0.06851,+0.11300]` | +0.09548 `[+0.07096,+0.11999]` | +0.23643 `[+0.21585,+0.25702]` |

| Candidate minus original update4140 | nominal det | nominal stochastic | outward stochastic |
|---|---:|---:|---:|
| Learned-uniform | +0.10244 `[+0.08279,+0.12208]` | +0.08452 `[+0.02072,+0.14833]` | +0.62572 `[+0.55883,+0.69260]` |
| Learned-early-balanced | +0.09369 `[+0.07582,+0.11157]` | +0.07654 `[+0.01429,+0.13878]` | +0.66410 `[+0.60454,+0.72366]` |

The learned-uniform actor passes the registered external local screen: no
completion panel is below incoming and no discounted-return contrast is
confidently harmful. Early-balanced fails only the outward-completion criterion
(412 vs 415) despite positive mean discounted-return contrasts on all panels.
That difference is retained rather than used to select or alter either candidate.

The original update's outward discounted-return change versus incoming is
`-0.42766`, with descriptive paired 99% interval `[-0.48525,-0.37008]`, along
with the 33-completion loss. Thus the independent evaluator captures the same
kind of local recovery regression that motivated the native experiment, without
sharing its RNG implementation.

## Interpretation

This is the first actual task-trajectory evidence for the two completed
learned-baseline numerical actor steps. It supports the earlier local-gradient
result: changing the baseline can avoid the destructive update4140 behavior
without shrinking the actor step to zero. The uniform learned baseline is the
cleaner result on this fixed external screen; the early-balanced variant improves
reward but has three additional outward failures versus incoming.

This remains one exposed training history, one independently implemented
simulator and one fixed external random domain. It does not establish continual
critic tracking, repeated from-scratch reliability, native numerical parity, or
sustained-balancing qualification. The queued native experiment remains decisive
for the registered repository-specific policy-performance claim.

## Reproduction

A complete second execution reproduced `report.json` and `evaluation.csv`
byte-for-byte.

- `external_eval.py` SHA256: `6fb6022d61eca3b6ba1415ce955a736f468e823a9b0a9e10ee9b2830d2b59d89`
- `results/report.json` SHA256: `eadaca6e9a104d28a9ae103a9520240621a4e0fb326901e64911db21c3b56f45`
- `results/evaluation.csv` SHA256: `110c41d2bdb8def6d2fc2749735168dc251fcc3479acdbc879854b4d821546e6`

The conversation evidence bundle `rustrobotics-learned-baseline-external-eval-20260922.zip`
contains the exact four actor bytes, evaluator source, all 6,144 evaluation rows,
report, manifest, execution log and repeat log.