# Shared-policy PPO rollout collection

One `PpoTrainerSession` owns one actor, one critic, and their persistent Adam
optimizers. It may collect from several independent environment streams before
one update. Native simulation and the browser worker use this same session.
There are no independently trained policies to average.

## Collection and budgets

`PpoConfig::rollout_steps` remains **steps per environment per update**. For `N`
environments and `T` steps, each update collects exactly `N * T` transitions. It
then runs the existing configured epochs and minibatches on that union. Increasing
`N` while retaining `T` increases both data and optimization work. To compare
sample budgets, count actual `metrics.total_env_steps`, not calls or UI ticks.

The actor and critic stay immutable during collection. Each environment owns its
physical state, current noisy observation, unfinished episode return, continuing
environment RNG, and continuing action RNG. Collection visits environments in a
fixed order. GAE is closed separately at each true terminal, timeout and local
buffer cutoff; it never crosses into another environment. Timeouts bootstrap from
the final observation before reset, true terminals use zero, and buffer cutoffs
retain live episode state. Raw advantages are normalized once over the complete
union, not independently per environment or episode.

This is synchronous, environment-major collection. Multiple environments do not
imply multiple CPU threads or browser workers. The browser uses **one learner
worker**, regardless of environment count. No speedup is claimed. A future
parallel collector must preserve the common-policy and trajectory-boundary
contracts rather than average independently optimized network parameters.

## Compatibility and randomness

`new(config)` and `new_seeded(config, seed)` still create one environment.
`new_with_environments(config, count)` and
`new_seeded_with_environments(config, seed, count)` select the same training path
with more streams. `count == 1` preserves the original initialization and random
stream sequence. Extra streams use a separate fixed seed domain; constructing
more streams does not consume the first stream's randomness or change the initial
models. Additional states contain no models or optimizers. Zero counts and
integer-overflowing rollout sizes are rejected by the explicit session APIs.

The simulator's legacy zero-count convention still maps to one environment. Its
serialized `parallel_trainers`, `total_replicas` and related legacy status names
are retained for existing clients, but now count **environments**, not agents or
OS workers. UI labels describe the new semantics. Existing saved counts greater
than one intentionally stop using the previous parameter-averaging algorithm.
Single-environment hyperparameters, task, network and update equations are not
retuned by this change. Reset the learner to apply changed environment counts.

The WASM creation input accepts the existing flat trainer configuration, plus an
optional `environment_count` (default one) and optional `seed` (default fresh
entropy). A seed enables replay within the same numerical build; it is not an
exact-resume checkpoint. Invalid pool counts fail before publishing a session.

## Optimizer and metric lifecycle

Ticks, polling, readouts and normal collection do not reload weights or recreate
optimizers. Grouped and split update calls have the same numerical evolution in
the same build. The coordinator copies the single learner's metrics, including a
negative best episode return; it does not average metrics from independent agents.
Recent-return metrics follow the deterministic environment-major completion order.

An explicit external `load_shared_state()` remains a **weight-transfer warm
start**: it replaces actor/critic weights and resets Adam moments, while keeping
all live environments, RNG cursors, partial returns and metrics. It is never
called implicitly by the coordinator. Portable snapshots still are not full
training checkpoints. Seeded bitwise replay is not promised across platforms,
compiler versions or numerical backends.

## Scope of the correction

This fixes the native single-replica self-reload defect and removes nonstandard
multi-replica parameter averaging described in issue #36. It does not establish a
learning-quality improvement, solve the standalone Rust/SB3 discrepancy in #35,
or qualify sustained balancing in #33. Those require separate normal
from-scratch comparisons and a frozen held-out protocol. No supplied controller,
critic pretraining, diagnostic return oracle, or evaluation-driven update is used.
