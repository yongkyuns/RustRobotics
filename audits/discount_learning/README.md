# Discount-learning comparison — complete; gamma 0.995 not adopted

[Full results](RESULTS.md), report commit `39f0ef8c1e36d1f325f20a0ccfad295ddcdb06ce`.

Original run **35511524659**, executed source **6516572add5509ccb338e4ecb55d572733f99bd2**, completed on its first attempt on September 20, 2026 at13:39:44UTC. All17 jobs pass. This replaces the earlier queued-status note; verification did not dispatch duplicate training.

Eight paired exposed development seeds41001–41008 train gamma0.99 and0.995 from scratch using the same4096 ordinary-reset recovery-union plus512 half-outward updates. Gamma is set before the first target, not switched in a pretrained critic. All physical settings, reward, experimental lambda1, global normalization, support lengths and Adam schedule remain fixed. Both policies are scored with the SAME gamma0.99 evaluator. The complete protocol remains in issue35 comment5749864155, posted before dispatch.

| Training gamma | Five-minute deterministic /512 | Five-minute stochastic /512 | Sixty-second outward /512 |
|---|---:|---:|---:|
|0.99|508|502|477|
|0.995|509|507|443|

The candidate passes BOTH nominal absolute panels, but fails outward recovery and overall development/absolute qualification. Its outward failures increase35->69, concentrated in seeds41004 and41006. Its paired-history adjusted99% outward completion effect is-6.641percentage points[-32.957,+19.676]; no required positive lower bound. Nominal retention passes, but cannot override failed recovery. These are exposed training seeds, not fresh held-out or hardware qualification.

Verification passes101 native unit/audit tests,7 ordinary balancing+9 ordinary learning controls,Clippy/format/source restoration,and48 offline synthetic/property tests. All17 ZIPs/4344 payload hashes,21504 outcomes,96 regular actor/critic pairs,768 selected optimizer transactions and144 full traces/1243991 steps are checked. All historical gamma0.99 checkpoints and paired initial paths match. Unexported complete gradients,Adam moments and trajectories are not independently regenerated.

A fresh-unpack replay reproduced all EIGHT numerical reports byte-for-byte, explicitly retaining development_screen=False and absolute_screen=False. The downloadable full evidence contains all17 unchanged current raw archives,selected historical comparison inputs,actual executed sources/logs,standalone verification code/tests,tables and replay instructions. Offline verification executes no native binary,network,simulator or training.

**Keep gamma0.99 and all production settings unchanged.** No trained weights,PR38,master,deployed assets or fallback changed or merged. The supported lambda0.95 default is distinct from this experimental lambda1 recipe. Fixed support windows increase candidate bootstrap dependence; that is a limit of this gamma-only comparison, not a proven causal explanation or a promoted next fix.
