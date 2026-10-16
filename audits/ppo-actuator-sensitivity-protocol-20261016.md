# PPO #42 — frozen actuator-sensitivity experiment (development only)

Scope: Native categorical Rust/Burn reference, PR #41, already-exposed training
seeds 201–204. This evaluates **frozen final policies** after 976 updates
on the same 64 common ordinary and 64 difficult reset keys as the original
four-seed evidence. There is no policy update during evaluation.

Comparisons use exactly the original noisy nonlinear native plant, reward,
15-second horizon (1500 ticks, 0.01 s per tick), action RNG, observation noise
and independent per-key environmental RNG streams. The actor sees only the
noisy observation; true state is written **solely** to the diagnostic CSV.
No held-out trajectories are used for policy updates.

The predeclared actuator modes (not chosen after seeing stress results) are:

- ideal: original immediate categorical command +/-20 N.
- delay_one_tick: one-step command delay, initially 0 N.
- hold_two_ticks: new policy command applied on ticks 0,2,4,... and held
  for the intervening tick. Policy still observes/chooses each tick.
- slew_five_n_per_tick: commanded force changes at most 5 N per tick;
  starts at 0 N. Noise/disturbance are still applied by PendulumEnv.

For each of four seeds × four actuator modes × two policy modes × two common
panels × 64 keys, save 4096 **complete episode rows**: survival, termination,
original reward, min cart clearance and requested-force sign switches. Trace
all up-to-1500 10-ms transitions for **key 4100100100** in both action modes
and all four actuation modes: noisy observation, pre/post true state, desired
command, actuator output, noisy applied physical force, reward, separate rail
and angle-limit violations, and terminal/truncated flags. Persist failures
and partial data, and the full source SHA. The verifier requires that all
nominal completion counts match the prior four-seed evidence before making
a comparison to actuator variants.

This does NOT claim an actuator model realistic for real motors: 5 N/tick,
one 10-ms delay and two-tick hold are diagnostic examples. It also does
NOT claim four new training seeds, independent held-out acceptance, or an
explanation for continuous PPO's failure. The development key was selected
because it previously failed; do not tune the policy on it or remove it
from future scoring. See #33/#35/#42 for qualification gates.
