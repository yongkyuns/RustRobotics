#!/usr/bin/env python3
"""Temporary exact-source transformation; removed before review."""
from pathlib import Path
import hashlib

R = Path('.')
for name, expected in {
    'rust_robotics_train/src/trainer.rs': '515bf5e3f6f2875234d7711200e6f7019aad5123',
    'rust_robotics_sim/src/simulator/ppo_trainer/mod.rs': '6684f149a3a12cac1fae730d6ce9e3629c7af4cf',
    'rust_robotics_sim/src/simulator/ppo_trainer/native.rs': '336aa9134e332c6aaae9adbf3a5ba58e2da355f6',
    'rust_robotics_sim/src/simulator/ppo_trainer/lifecycle_tests.rs': '4ed393813be406f46844ea09a61d4c766ae67ae4',
}.items():
    b = (R / name).read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
    assert actual == expected, (name, actual)

def once(s, a, b):
    assert s.count(a) == 1, (a, s.count(a))
    return s.replace(a, b)

p = R / 'rust_robotics_train/src/trainer.rs'
s = p.read_text()
a = s.index('    fn collect_rollout_with_rng<R:')
b = s.index('    /// Optimizes actor', a)
collector = s[a:b].rstrip()
collector = once(collector, '    fn collect_rollout_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> RolloutBatch {', '''    pub(super) fn collect<R: Rng + ?Sized>(
        &mut self,
        cursor: RolloutCursor<'_>,
        rng: &mut R,
    ) -> RolloutBatch {''')
collector = once(collector, '''        // Keep the existing whole-rollout normalization, not per-episode scaling.
        let advantages = normalize(&advantages);
        self.metrics.last_mean_advantage = mean_slice(&advantages).unwrap_or(0.0);
''', '')
collector = collector.replace('self.current_observation', '*cursor.current_observation').replace('self.episode_return', '*cursor.episode_return').replace('&mut self.environment_rng', 'cursor.environment_rng')
collector = collector.replace('self.env.reset_with_rng', 'cursor.env.reset_with_rng')
collector = once(collector, 'let step = self\n                .env\n                .step_with_rng', 'let step = cursor.env.step_with_rng')
assert 'self.env' not in collector
inference = s[s.index('    /// Computes the unsquashed Gaussian mean'):s.index('\n}\n\nfn gather_observations')]
helper = '''//! Trajectory-local collection under one immutable actor/critic pair.
//!
//! Additional environments own no models or optimizers. This code is also the
//! single-environment collector: there is no second PPO training implementation.
use super::*;

#[derive(Debug, Clone)]
pub(super) struct RolloutEnvironment {
    pub(super) env: PendulumEnv,
    pub(super) current_observation: [f32; 4],
    pub(super) episode_return: f32,
    pub(super) environment_rng: StdRng,
    pub(super) action_rng: StdRng,
}

impl RolloutEnvironment {
    pub(super) fn new(config: PendulumEnvConfig, environment_seed: u64, action_seed: u64) -> Self {
        let mut environment_rng = StdRng::seed_from_u64(environment_seed);
        let env = PendulumEnv::new_with_rng(Default::default(), config, &mut environment_rng);
        let current_observation = env.observation_with_rng(&mut environment_rng);
        Self {
            env,
            current_observation,
            episode_return: 0.0,
            environment_rng,
            action_rng: StdRng::seed_from_u64(action_seed),
        }
    }
}

pub(super) struct RolloutCursor<'a> {
    pub(super) env: &'a mut PendulumEnv,
    pub(super) current_observation: &'a mut [f32; 4],
    pub(super) episode_return: &'a mut f32,
    pub(super) environment_rng: &'a mut StdRng,
}

pub(super) struct RolloutCollector<'a> {
    pub(super) config: &'a PpoTrainerConfig,
    pub(super) device: &'a AutodiffDevice,
    pub(super) actor: &'a PolicyNetwork<AutodiffBackend>,
    pub(super) critic: &'a ValueNetwork<AutodiffBackend>,
    pub(super) metrics: &'a mut PpoMetrics,
    pub(super) recent_episode_returns: &'a mut Vec<f32>,
}

impl RolloutCollector<'_> {
''' + collector + '\n\n' + inference + '\n}\n'
helper = helper.replace('&self.device', 'self.device').replace('&mut self.recent_episode_returns', 'self.recent_episode_returns').replace('&self.recent_episode_returns', 'self.recent_episode_returns').replace('std::mem::take(&mut *cursor.episode_return)', 'std::mem::take(cursor.episode_return)')
(R / 'rust_robotics_train/src/ppo_rollout_pool.rs').write_text(helper)
s = s[:a] + '''    fn collect_rollout_with_rng<R: Rng + ?Sized>(&mut self, rng: &mut R) -> RolloutBatch {
        // All streams see the SAME frozen actor and critic. No optimization is
        // allowed until the complete pool has been collected. Each call below
        // independently closes terminal, timeout and buffer-cutoff GAE traces.
        let mut collector = RolloutCollector {
            config: &self.config,
            device: &self.device,
            actor: &self.actor,
            critic: &self.critic,
            metrics: &mut self.metrics,
            recent_episode_returns: &mut self.recent_episode_returns,
        };
        let mut rollout = collector.collect(
            RolloutCursor {
                env: &mut self.env,
                current_observation: &mut self.current_observation,
                episode_return: &mut self.episode_return,
                environment_rng: &mut self.environment_rng,
            },
            rng,
        );
        for stream in &mut self.additional_environments {
            let RolloutEnvironment {
                env, current_observation, episode_return, environment_rng, action_rng,
            } = stream;
            let next = collector.collect(
                RolloutCursor { env, current_observation, episode_return, environment_rng },
                action_rng,
            );
            rollout.observations.extend(next.observations);
            rollout.latent_actions.extend(next.latent_actions);
            rollout.old_log_probs.extend(next.old_log_probs);
            rollout.returns.extend(next.returns);
            rollout.advantages.extend(next.advantages);
        }
        // Normalize ONCE over the union, never separately per stream/episode.
        rollout.advantages = normalize(&rollout.advantages);
        self.metrics.last_mean_advantage = mean_slice(&rollout.advantages).unwrap_or(0.0);
        rollout
    }

''' + s[b:]
s = once(s, 'const OBS_DIM: usize = 4;', '''#[path = "ppo_rollout_pool.rs"]
mod rollout_pool;
use rollout_pool::{RolloutCollector, RolloutCursor, RolloutEnvironment};

const OBS_DIM: usize = 4;''')
s = once(s, '    update_rng: StdRng,\n}', '''    update_rng: StdRng,
    // Stream zero retains the original fields/seeding contract above. Additional
    // streams contain only environment/episode/RNG state, not independent agents.
    additional_environments: Vec<RolloutEnvironment>,
}''')
s = once(s, '            update_rng,\n', '            update_rng,\n            additional_environments: Vec::new(),\n')
anchor = '    /// Returns the immutable training configuration for this session.'
s = once(s, anchor, '''    /// Creates one learner with several environment streams and fresh randomness.
    /// `ppo.rollout_steps` is the number of transitions PER environment/update.
    /// Collection is synchronous; this does not promise CPU-worker parallelism.
    pub fn new_with_environments(config: PpoTrainerConfig, environments: usize) -> Self {
        Self::new_seeded_with_environments(config, rand::thread_rng().gen(), environments)
    }

    /// Reproducible shared-policy collection. One environment is exactly the
    /// original seeded session, including initialization and every RNG stream.
    /// Additional streams use a separate, fixed seed domain; adding streams does
    /// not alter the initial models or any existing stream's initial randomness.
    ///
    /// Panics for zero environments or an overflowing rollout size. As with
    /// `new_seeded`, replay is within the same numerical backend/build only.
    pub fn new_seeded_with_environments(
        config: PpoTrainerConfig,
        seed: u64,
        environments: usize,
    ) -> Self {
        assert!(environments > 0, "PPO requires at least one environment");
        assert!(config.ppo.rollout_steps.checked_mul(environments).is_some(),
            "PPO pooled rollout size overflows usize");
        let env_config = config.env;
        let mut session = Self::new_seeded(config, seed);
        let mut streams = StdRng::seed_from_u64(seed ^ 0x504f_4f4c_4544_0001);
        for _ in 1..environments {
            session.additional_environments.push(RolloutEnvironment::new(
                env_config, streams.gen(), streams.gen(),
            ));
        }
        session
    }

    /// Number of persistent environments sharing this session's actor/critic.
    pub fn environment_count(&self) -> usize {
        1 + self.additional_environments.len()
    }

''' + anchor)
s = once(s, '    /// Computes the unsquashed Gaussian mean predicted by the actor.', '    #[cfg(test)]\n    /// Computes the unsquashed Gaussian mean predicted by the actor.')
start = s.index("    /// Computes the critic's scalar value estimate")
end = s.index('\n}\n\nfn gather_observations', start)
s = s[:start] + s[end:]
s = s.replace('/// - the current environment state', '/// - one or more independent environment/episode/RNG states')
s = s.replace('    /// Exports both actor and critic state for replica synchronization.', '    /// Exports actor and critic weights for an explicit external warm start.')
s = s.replace('    /// Each update collects a fresh rollout, then performs one optimization pass\n    /// over that rollout using shuffled minibatches.', '    /// Each update collects every environment under one frozen actor/critic,\n    /// then optimizes the union for the configured epochs with shuffled minibatches.')
s += '\n#[cfg(test)]\n#[path = "ppo_pool_tests.rs"]\nmod pool_tests;\n'
p.write_text(s)
base = R / 'rust_robotics_sim/src/simulator/ppo_trainer'
(base / 'mod.rs').write_text('''//! One persistent PPO learner for the native app or one browser worker.
//!
//! The selected environment count controls rollout coverage, not a collection
//! of independently initialized/optimized agents. No parameter averaging or
//! automatic external weight transfer occurs on ticks or readouts.
use rust_robotics_core::{PolicySnapshot, PpoMetrics};
use rust_robotics_train::PpoTrainerConfig;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(not(target_arch = "wasm32"))]
use native::NativePpoReplicaExecutor as PlatformPpoReplicaExecutor;
#[cfg(target_arch = "wasm32")]
use web::WebPpoReplicaExecutor as PlatformPpoReplicaExecutor;

/// Legacy type name retained for the UI readout contract. Counts now describe
/// environments served by the one learner, NOT independent policy replicas.
#[derive(Debug, Clone, Copy, Default)]
pub struct PpoReplicaStatus {
    pub total: usize,
    pub ready: usize,
    pub busy: usize,
}

#[derive(Default)]
pub struct PpoTrainerCoordinator {
    executor: Option<PlatformPpoReplicaExecutor>,
    environment_count: usize,
    snapshot: Option<PolicySnapshot>,
    metrics: Option<PpoMetrics>,
    last_error: Option<String>,
    busy: bool,
    status: PpoReplicaStatus,
}

impl PpoTrainerCoordinator {
    /// Starts one learner with the requested number of environment streams.
    /// The old zero-count UI convention still means one environment.
    pub fn reset(&mut self, config: &PpoTrainerConfig, environments: usize) {
        self.destroy();
        self.environment_count = environments.max(1);
        self.executor = Some(PlatformPpoReplicaExecutor::new(
            config.clone(), self.environment_count,
        ));
        self.refresh_summary();
    }

    pub fn destroy(&mut self) {
        if let Some(mut executor) = self.executor.take() {
            executor.destroy();
        }
        self.environment_count = 0;
        self.snapshot = None;
        self.metrics = None;
        self.last_error = None;
        self.busy = false;
        self.status = PpoReplicaStatus::default();
    }

    pub fn is_initialized(&self) -> bool { self.executor.is_some() }

    /// Advances the learner without rebuilding networks or optimizer state.
    pub fn tick(&mut self, updates: usize) {
        if let Some(executor) = &mut self.executor {
            executor.tick(updates.max(1));
        }
        self.refresh_summary();
    }

    /// Readout/polling never performs a model synchronization or an update.
    pub fn refresh(&mut self) {
        if let Some(executor) = &mut self.executor { executor.poll(); }
        self.refresh_summary();
    }

    pub fn snapshot(&self) -> Option<&PolicySnapshot> { self.snapshot.as_ref() }
    pub fn metrics(&self) -> Option<&PpoMetrics> { self.metrics.as_ref() }
    pub fn last_error(&self) -> Option<&str> { self.last_error.as_deref() }
    pub fn busy(&self) -> bool { self.busy }
    pub fn status(&self) -> PpoReplicaStatus { self.status }

    fn refresh_summary(&mut self) {
        let Some(executor) = &self.executor else { return; };
        self.busy = executor.busy();
        self.status = PpoReplicaStatus {
            total: self.environment_count,
            ready: if executor.ready() { self.environment_count } else { 0 },
            busy: if self.busy { self.environment_count } else { 0 },
        };
        self.last_error = executor.last_error().map(str::to_owned);
        self.snapshot = executor.shared_state().map(|state| state.policy);
        // In particular, preserve a negative best return, rather than max(0, x).
        self.metrics = executor.metrics().cloned();
    }
}

impl Drop for PpoTrainerCoordinator {
    fn drop(&mut self) { self.destroy(); }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lifecycle_tests;
''')
p = base / 'native.rs'
s = p.read_text()
s = once(s, 'pub fn new(config: PpoTrainerConfig) -> Self {', 'pub fn new(config: PpoTrainerConfig, environments: usize) -> Self {')
s = once(s, 'session: PpoTrainerSession::new(config),', 'session: PpoTrainerSession::new_with_environments(config, environments),')
s = once(s, '    pub fn accepts_shared_state(&self) -> bool {\n        true\n    }\n\n', '')
s = once(s, '    pub fn load_shared_state(&mut self, state: &PpoSharedState) {\n        self.session.load_shared_state(state);\n    }\n\n', '')
s = once(s, 'pub(super) fn new_seeded(config: PpoTrainerConfig, seed: u64) -> Self {', 'pub(super) fn new_seeded(config: PpoTrainerConfig, seed: u64, environments: usize) -> Self {')
s = once(s, 'session: PpoTrainerSession::new_seeded(config, seed),', 'session: PpoTrainerSession::new_seeded_with_environments(config, seed, environments),')
p.write_text(s)
p = base / 'web.rs'
s = p.read_text()
s = once(s, 'use serde::Deserialize;', 'use serde::{Deserialize, Serialize};')
s = once(s, '#[derive(Deserialize)]\nstruct PollState', '''#[derive(Serialize)]
struct WorkerConfig<'a> {
    #[serde(flatten)]
    config: &'a PpoTrainerConfig,
    environment_count: usize,
}

#[derive(Deserialize)]
struct PollState''')
s = once(s, 'pub fn new(config: PpoTrainerConfig) -> Self {', 'pub fn new(config: PpoTrainerConfig, environments: usize) -> Self {')
s = once(s, 'executor.reset(config);', 'executor.reset(config, environments);')
s = once(s, 'fn reset(&mut self, config: PpoTrainerConfig) {', 'fn reset(&mut self, config: PpoTrainerConfig, environments: usize) {')
s = once(s, 'serde_wasm_bindgen::to_value(&config)', 'serde_wasm_bindgen::to_value(&WorkerConfig { config: &config, environment_count: environments })')
s = once(s, '    pub fn accepts_shared_state(&self) -> bool {\n        self.ready && !self.busy\n    }\n\n', '')
a = s.index('    pub fn load_shared_state(&mut self, state: &PpoSharedState) {')
b = s.index('    pub fn last_error', a)
s = s[:a] + s[b:]
s = once(s, '    #[wasm_bindgen(js_name = rustRoboticsPpoTrainerLoadSharedState)]\n    fn js_ppo_trainer_load_shared_state(handle: u32, state: JsValue);\n', '')
p.write_text(s)
p = base / 'lifecycle_tests.rs'
s = p.read_text()
s = once(s, '    coordinator.executors = vec![PlatformPpoReplicaExecutor::new_seeded(config, seed)];', '''    coordinator.environment_count = 1;
    coordinator.executor = Some(PlatformPpoReplicaExecutor::new_seeded(config, seed, 1));''')
s = s.replace('coordinator.executors[0].shared_state()', 'coordinator.executor.as_ref().unwrap().shared_state()')
s += '''
#[test]
fn multiple_environments_use_one_persistent_pooled_learner() {
    let mut config = PpoTrainerConfig::default();
    config.ppo.rollout_steps = 16;
    config.ppo.mini_batch_size = 16;
    config.hidden_dim = 8;
    for environments in [2, 8] {
        let mut coordinator = PpoTrainerCoordinator::default();
        coordinator.environment_count = environments;
        coordinator.executor = Some(PlatformPpoReplicaExecutor::new_seeded(
            config.clone(), 201, environments,
        ));
        coordinator.refresh_summary();
        let mut direct = PpoTrainerSession::new_seeded_with_environments(
            config.clone(), 201, environments,
        );
        for _ in 0..3 {
            coordinator.tick(1);
            coordinator.refresh();
            direct.train_updates(1);
            let actual = coordinator.executor.as_ref().unwrap().shared_state().unwrap();
            let expected = direct.shared_state();
            assert_eq!(actual.policy, expected.policy);
            assert_eq!(actual.value, expected.value);
            assert_eq!(coordinator.metrics(), Some(direct.metrics()));
            assert_eq!(coordinator.status().total, environments);
            assert_eq!(coordinator.status().ready, environments);
            assert_eq!(coordinator.status().busy, 0);
        }
        assert_eq!(direct.metrics().total_env_steps, 3 * environments * 16);
        assert_eq!(direct.metrics().total_updates, 3);
    }
}
'''
p.write_text(s)
p = R / 'rust_robotics_sim/src/web_ppo_worker.rs'
s = p.read_text()
a = '#[cfg(target_arch = "wasm32")]\nfn next_session_id()'
s = once(s, a, '''#[cfg(target_arch = "wasm32")]
#[derive(Deserialize)]
struct WorkerTrainerConfig {
    #[serde(flatten)]
    config: PpoTrainerConfig,
    #[serde(default = "one_environment")]
    environment_count: usize,
    // Optional caller-owned seed also makes real worker regressions reproducible.
    // Existing calls omit it and retain entropy-seeded initialization.
    #[serde(default)]
    seed: Option<u64>,
}

#[cfg(target_arch = "wasm32")]
fn one_environment() -> usize { 1 }

''' + a)
s = once(s, '''    let config: PpoTrainerConfig =
        serde_wasm_bindgen::from_value(config).map_err(|err| js_err(err.to_string()))?;
    let session = PpoTrainerSession::new(config);''', '''    let input: WorkerTrainerConfig =
        serde_wasm_bindgen::from_value(config).map_err(|err| js_err(err.to_string()))?;
    if input.environment_count == 0
        || input.config.ppo.rollout_steps.checked_mul(input.environment_count).is_none()
    {
        return Err(js_err("invalid PPO environment count or pooled rollout size"));
    }
    let session = match input.seed {
        Some(seed) => PpoTrainerSession::new_seeded_with_environments(
            input.config, seed, input.environment_count,
        ),
        None => PpoTrainerSession::new_with_environments(input.config, input.environment_count),
    };''')
p.write_text(s)
p = R / 'rust_robotics_core/src/lib.rs'
s = p.read_text()
s = once(s, '''/// Bundle of policy and value snapshots exchanged between trainer replicas.
///
/// The simulator's multi-replica coordinator averages these dense tensors and
/// redistributes the merged result. Keeping the actor and critic bundled
/// together avoids accidental shape mismatches between independently updated
/// models. This is weight-transfer state, not an exact-resume checkpoint:''', '''/// Bundle of actor and critic weights for an explicit external warm start.
///
/// Ordinary training keeps one learner and pools environment rollouts; it does
/// not average independently optimized networks or reload on readout. Keeping
/// both networks bundled supports intentional transfer of compatible models.
/// This is weight-transfer state, not an exact-resume checkpoint:''')
p.write_text(s)
p = R / 'rust_robotics_train/src/algorithm.rs'
s = p.read_text().replace('    pub rollout_steps: usize,', '    /// Transitions per environment per update; pooled size is count * steps.\n    pub rollout_steps: usize,', 1)
p.write_text(s)
p = R / 'rust_robotics_sim/src/simulator/pendulum/ui.rs'
s = p.read_text().replace('Replicas {', 'Environments {').replace('Web: CPU workers.', 'Web: one learner worker; pooled environments.').replace('ui.label("Parallel");', 'ui.label("Environments");').replace('ui.label("Rollout");', 'ui.label("Steps/env");')
p.write_text(s)
p = R / 'rust_robotics_sim/src/simulator/pendulum/training.rs'
s = p.read_text().replace('which may run one or more training replicas', 'which owns one learner and one or more environments').replace('Rebuilds trainer replicas from the current configuration.', 'Rebuilds the learner and its environment streams from the current configuration.')
p.write_text(s)
Path('/tmp/ppo-fix-message').write_text('fix: collect pooled trajectories under one persistent PPO learner')
print('Applied shared-policy collection and coordinator replacement; defaults, losses, optimizer and task unchanged.')
