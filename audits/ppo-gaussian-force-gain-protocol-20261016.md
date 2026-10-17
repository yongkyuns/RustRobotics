# Gaussian force-amplitude causal probe — issue #42

**Predeclared development-only test.** Four previously exposed squashed-Gaussian
policies (training seeds 201–204), each trained for the exact original 976
PPO updates / 499,712 interactions with unchanged native plant and optimizer.
After training, only frozen **evaluation** force commands are transformed.

All common 64-key ordinary and difficult panels are tested for each seed,
both stochastic and deterministic modes, 1500×0.01-s ticks, with original
observation/actuation noise, physical termination and reward functions.
Every episode is recorded; no best seed, panel or intervention is selected.

Interventions, specified **before outcomes**, applied to the *same learned
Gaussian policy*:

- `ideal`: unchanged bounded Gaussian force F=20*tanh(z) N.
- `boost_1p5`: 1.5 F clipped to ±20 N.
- `boost_2p0`: 2 F clipped to ±20 N.
- `bang_bang_sign`: ±20 N from sign(F); an exact zero maps to 0 N.

The experiment keeps action polarity at each policy evaluation decision,
but the physical trajectories and later observations can diverge after
interventions. More force also changes squared-action reward penalties.
This is a causal evaluation-time intervention, **not proof** of a single
root cause nor a continuous-production PPO repair.

The workflow must reproduce the fixed original Gaussian final counts:
difficult 54/53,56/55,59/58,48/44 out of 64; ordinary 64/64 for
seeds 201–203 and 63/63 for seed 204. Its other modes are not forced
to pass any success threshold: all negative outcomes remain evidence.

All 4×4×2×2×64=**4096** full-episode outcomes are retained, with an
additional full time-step trace for known selected key `4100100015`.
This does not constitute new training-seed confirmation for #33 and
does not justify merging PR #41.
