"""One-off guarded source integration; removed before PR submission."""
from pathlib import Path
import hashlib

ROOT = Path('rust_robotics_train')
expected = {
    'src/trainer.rs': '320ec24727074438029be89b8d942a39a68fe1ae',
    'src/env.rs': 'b120dfb28842ca063275830998f5eb0a41c166be',
    'src/model.rs': 'decaf639529e3f7ec1943f26499714e15e423404',
    'README.md': '98167ebe769100a3682c2c11c5256b82bf354cda',
}
for name, sha in expected.items():
    data = (ROOT / name).read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert actual == sha, (name, actual, sha)

def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)

path = ROOT / 'src/env.rs'
s = path.read_text()
s = replace(s, '    pub fn new(model: Model, config: PendulumEnvConfig) -> Self {', '''    pub fn new(model: Model, config: PendulumEnvConfig) -> Self {
        Self::new_with_rng(model, config, &mut rand::thread_rng())
    }

    /// Constructs and resets using only the caller's random stream.
    /// Keep using the `_with_rng` methods for a reproducible trajectory.
    pub fn new_with_rng<R: Rng + ?Sized>(
        model: Model,
        config: PendulumEnvConfig,
        rng: &mut R,
    ) -> Self {''')
s = replace(s, '        env.reset();', '        env.reset_with_rng(rng);')
s = replace(s, '''    pub fn observation(&self) -> [f32; 4] {
        let mut rng = rand::thread_rng();''', '''    pub fn observation(&self) -> [f32; 4] {
        self.observation_with_rng(&mut rand::thread_rng())
    }

    /// Samples observation noise without touching any shared random stream.
    pub fn observation_with_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> [f32; 4] {''')
s = replace(s, '''    pub fn reset(&mut self) -> [f32; 4] {
        let mut rng = rand::thread_rng();''', '''    pub fn reset(&mut self) -> [f32; 4] {
        self.reset_with_rng(&mut rand::thread_rng())
    }

    /// Draws reset state and its observation from the caller's continuing stream.
    /// This does not reseed that stream.
    pub fn reset_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f32; 4] {''')
s = replace(s, '''    pub fn step(&mut self, action: f32) -> StepResult {
        let mut rng = rand::thread_rng();''', '''    pub fn step(&mut self, action: f32) -> StepResult {
        self.step_with_rng(action, &mut rand::thread_rng())
    }

    /// Draws action noise, disturbances and observation noise from one explicit
    /// stream. Dynamics, rewards and boundary semantics match `step`.
    pub fn step_with_rng<R: Rng + ?Sized>(&mut self, action: f32, rng: &mut R) -> StepResult {''')
s = s.replace('sample_symmetric(&mut rng,', 'sample_symmetric(rng,')
s = replace(s, '                    &mut rng,', '                    rng,')
s = replace(s, '        self.observation()\n', '        self.observation_with_rng(rng)\n')
s = replace(s, '            observation: self.observation(),', '            observation: self.observation_with_rng(rng),')
s = replace(s, 'fn sample_symmetric(rng: &mut impl Rng, magnitude: f32) -> f32 {', 'fn sample_symmetric<R: Rng + ?Sized>(rng: &mut R, magnitude: f32) -> f32 {')
path.write_text(s)

path = ROOT / 'src/model.rs'
s = path.read_text()
s = replace(s, 'use rust_robotics_core::{LinearSnapshot, PolicySnapshot, ValueSnapshot};', 'use rust_robotics_core::{LinearSnapshot, PolicySnapshot, ValueSnapshot};\nuse rand::Rng;')
anchor = '    /// Runs the affine/ReLU stack without any task-specific output squash.'
code = '''    /// Eager caller-owned initialization with the same U(-1/sqrt(fan_in),
    /// 1/sqrt(fan_in)) weight/bias law as Burn 0.20.1's default LinearConfig.
    /// No backend seed or lazy random tensor is involved. Parameter IDs are
    /// backend bookkeeping, not part of the numerical reproducibility contract.
    pub(crate) fn new_with_rng<R: Rng + ?Sized>(
        device: &B::Device,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rng: &mut R,
    ) -> Self {
        fn layer<B: Backend, R: Rng + ?Sized>(
            device: &B::Device,
            in_dim: usize,
            out_dim: usize,
            rng: &mut R,
        ) -> Linear<B> {
            assert!(in_dim > 0 && out_dim > 0, "MLP dimensions must be positive");
            let bound = (1.0 / in_dim as f64).sqrt() as f32;
            let weight = (0..in_dim * out_dim)
                .map(|_| rng.gen_range(-bound..bound))
                .collect();
            let bias = (0..out_dim)
                .map(|_| rng.gen_range(-bound..bound))
                .collect();
            linear_from_snapshot(&LinearSnapshot { in_dim, out_dim, weight, bias }, device)
        }
        Self {
            input: layer(device, input_dim, hidden_dim, rng),
            hidden: layer(device, hidden_dim, hidden_dim, rng),
            output: layer(device, hidden_dim, output_dim, rng),
        }
    }

'''
s = replace(s, anchor, code + anchor)
path.write_text(s)

path = ROOT / 'src/trainer.rs'
s = path.read_text()
s = replace(s, '        PolicyNetwork, ValueNetwork,', '        Mlp, PolicyNetwork, ValueNetwork,')
s = replace(s, 'use rand::{seq::SliceRandom, Rng};', 'use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};')
s = replace(s, '    episode_return: f32,', '''    episode_return: f32,
    // Separate continuing streams prevent optimizer draw counts from changing
    // collection noise. No per-tick thread_rng or backend-global seed is used.
    environment_rng: StdRng,
    action_rng: StdRng,
    update_rng: StdRng,''')
s = replace(s, '''    /// Creates a new trainer session with fresh actor and critic networks.
    pub fn new(config: PpoTrainerConfig) -> Self {''', '''    /// Creates a fresh, entropy-seeded session with privately owned randomness.
    pub fn new(config: PpoTrainerConfig) -> Self {
        Self::new_seeded(config, rand::thread_rng().gen())
    }

    /// Creates a reproducible fresh session, including model initialization,
    /// environment resets/noise, action sampling, minibatches and entropy.
    ///
    /// Equal seeds, configurations and call sequences replay numerical state on
    /// the same backend/build. No cross-platform/version bitwise or checkpoint
    /// guarantee is made. This does not seed unrelated sessions or Burn globally.
    pub fn new_seeded(config: PpoTrainerConfig, seed: u64) -> Self {''')
s = replace(s, '''        let env = PendulumEnv::new(Default::default(), config.env);
        let current_observation = env.observation();
        let actor = PolicyNetwork::new(&device, OBS_DIM, config.hidden_dim, config.env.max_force);
        let critic = ValueNetwork::new(&device, OBS_DIM, config.hidden_dim);''', '''        // Fixed child order: actor initialization, critic initialization,
        // environment, actions, optimizer. Initialization draws do not consume
        // live environment/action streams, including when hidden_dim changes.
        let mut root = StdRng::seed_from_u64(seed);
        let mut child = || StdRng::seed_from_u64(root.gen());
        let mut actor_rng = child();
        let mut critic_rng = child();
        let mut environment_rng = child();
        let action_rng = child();
        let update_rng = child();
        let env = PendulumEnv::new_with_rng(Default::default(), config.env, &mut environment_rng);
        let current_observation = env.observation_with_rng(&mut environment_rng);
        let actor = PolicyNetwork {
            mlp: Mlp::new_with_rng(&device, OBS_DIM, config.hidden_dim, 1, &mut actor_rng),
            action_limit: config.env.max_force,
        };
        let critic = ValueNetwork {
            mlp: Mlp::new_with_rng(&device, OBS_DIM, config.hidden_dim, 1, &mut critic_rng),
        };''')
s = replace(s, '            episode_return: 0.0,', '''            episode_return: 0.0,
            environment_rng,
            action_rng,
            update_rng,''')
s = replace(s, '''        self.collect_rollout_with_rng(&mut rand::thread_rng())''', '''        // Use a local cursor to avoid aliasing &mut self with its RNG, then
        // retain the advanced cursor. This is a copy, not a reseed or shared RNG.
        let mut rng = self.action_rng.clone();
        let rollout = self.collect_rollout_with_rng(&mut rng);
        self.action_rng = rng;
        rollout''')
s = replace(s, '            let step = self.env.step(sample.action);', '            let step = self.env.step_with_rng(sample.action, &mut self.environment_rng);')
s = replace(s, '                self.current_observation = self.env.reset();', '                self.current_observation = self.env.reset_with_rng(&mut self.environment_rng);')
s = replace(s, '        self.optimize_with_rng(rollout, &mut rand::thread_rng());', '''        let mut rng = self.update_rng.clone();
        self.optimize_with_rng(rollout, &mut rng);
        self.update_rng = rng;''')
s += '\n#[cfg(test)]\n#[path = "ppo_seed_tests.rs"]\nmod seed_tests;\n'
path.write_text(s)

path = ROOT / 'README.md'
s = path.read_text()
anchor = '## Verification\n'
code = '''## Seeded sessions and explicit environment randomness

```rust
use rust_robotics_train::{PpoTrainerConfig, PpoTrainerSession};
let mut trainer = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), 42);
trainer.train_updates(3);
```

`new_seeded(config, seed)` covers actor/critic initialization, initial environment
state, observation/action noise, random disturbances, episode resets, Gaussian
action sampling, minibatch shuffling and bounded-policy entropy sampling. Five
child StdRng streams have a fixed order: actor initialization, critic initialization,
environment, actions, optimizer. The last three continue across updates. Model
size cannot consume collection randomness; optimizer draw counts cannot consume
action or environment randomness. Shuffling and entropy share the optimizer stream.
Reset and weight transfer do not reseed. No process-global/backend seed is set.

The existing `new(config)` draws a fresh entropy seed and uses this same owned
implementation. Existing config fields, serialized layout, signatures and the
standalone `PolicyNetwork::new` / `ValueNetwork::new` APIs remain available.
Seeded MLP initialization eagerly fills weights and biases from
`U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, matching Burn 0.20.1's default LinearConfig
law, rather than relying on lazily evaluated backend-global random tensors. It
preserves the distribution, not Burn's particular random sequence or parameter IDs.

`PendulumEnv` offers `new_with_rng`, `reset_with_rng`, `step_with_rng` and
`observation_with_rng` for caller-owned streams, including evaluation fixtures.
The original methods remain entropy-backed wrappers. Mixing those original
methods into an explicit-stream trajectory opts out of deterministic replay.
The environment itself stores no RNG and gains no mutex/interior mutability.
Standalone noisy observation calls consume draws; trainer snapshot/config/metrics
readouts do not. Cloning an environment copies its state, not an external RNG.

Replay requires the same seed, configuration, inputs, calls, locked dependencies,
backend and numerical build. Tests compare exact numerical values locally within
a build, including multiple actual optimizer updates and concurrently/interleaved
sessions. Bitwise equality across architectures, compiler or dependency upgrades
is not promised; StdRng's algorithm is not a stable serialized format. Store the
seed/config and source/lockfile revisions for experiments. This is fresh-run
reproducibility, not resumable checkpointing: portable shared weights still omit
RNG/optimizer/environment/partial-episode state. The simulator's replica scheduling
and aggregation are not covered by this single-session contract.

References: [Burn 0.20.1 Linear initialization](https://github.com/tracel-ai/burn/blob/v0.20.1/crates/burn-nn/src/modules/linear.rs)
and [rand 0.8 StdRng portability](https://docs.rs/rand/0.8.5/rand/rngs/struct.StdRng.html).

'''
s = replace(s, anchor, code + anchor)
s = replace(s, '''This is the action/objective increment of issue #5. End-to-end session/environment
seeding, reproducible short-learning qualification and broader checkpoint work
remain open.''', '''The action/objective, rollout-boundary and seeded-session increments of issue #5
are implemented. Reproducible short-learning improvement qualification and broader
checkpoint work remain open.''')
path.write_text(s)
