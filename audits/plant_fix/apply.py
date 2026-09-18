"""Apply reviewed source edits in a disposable checkout, then retain a real patch."""
from pathlib import Path
import shutil, subprocess
BASE='d3f59f9bf38d4038ba7c5008e3a3b41c15e99835'

def edit(name, old, new):
    p=Path(name);text=p.read_text();assert text.count(old)==1,(name,old[:90],text.count(old))
    p.write_text(text.replace(old,new))

shutil.copyfile('audits/plant_fix/cart_pole.rs','rust_robotics_algo/src/cart_pole.rs')
edit('rust_robotics_algo/src/lib.rs','pub mod control;','pub mod cart_pole;\npub mod control;')
trainer='rust_robotics_train/src/trainer.rs'
edit(trainer,'pub struct PpoTrainerConfig {','''pub struct PpoTrainerConfig {
    /// The physical plant used by every rollout stream. Old serialized configs
    /// without this field retain the historical physical parameter defaults.
    #[serde(default)]
    pub plant: rust_robotics_algo::cart_pole::CartPoleParameters,''')
edit(trainer,'            env: PendulumEnvConfig::default(),','            plant: Default::default(),\n            env: PendulumEnvConfig::default(),')
edit(trainer,'PendulumEnv::new_with_rng(Default::default(), config.env, &mut environment_rng)',
    'PendulumEnv::new_with_rng(config.plant.model(), config.env, &mut environment_rng)')
edit(trainer,'        let env_config = config.env;','        let env_config = config.env;\n        let model = config.plant.model();')
edit(trainer,'                .push(RolloutEnvironment::new(\n                    env_config,',
    '                .push(RolloutEnvironment::new(\n                    model,\n                    env_config,')
pool='rust_robotics_train/src/ppo_rollout_pool.rs'
edit(pool,'pub(super) fn new(config: PendulumEnvConfig, environment_seed: u64, action_seed: u64) -> Self {',
    '''pub(super) fn new(model: rust_robotics_algo::inverted_pendulum::Model,
        config: PendulumEnvConfig, environment_seed: u64, action_seed: u64) -> Self {''')
edit(pool,'PendulumEnv::new_with_rng(Default::default(), config, &mut environment_rng)',
    'PendulumEnv::new_with_rng(model, config, &mut environment_rng)')
# Add the newly explicit physical default to the two complete frozen recipe literals.
for p in Path('rust_robotics_train/tests').glob('*.rs'):
    text=p.read_text()
    text=text.replace('PpoTrainerConfig {\n        env:', 'PpoTrainerConfig {\n        plant: Default::default(),\n        env:')
    p.write_text(text)

env='rust_robotics_train/src/env.rs'
edit(env,'    control::StateSpace,','    cart_pole::CartPoleParameters,')
edit(env,'#[derive(Debug, Clone, Copy, Serialize, Deserialize)]\npub struct PendulumEnvConfig',
    '#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]\npub struct PendulumEnvConfig')
edit(env,'    steps: usize,\n}', '    steps: usize,\n    last_applied_force: f32,\n}')
edit(env,'''        let mut env = Self {
            model,
            config,
            state: vector![0.0, 0.0, 0.0, 0.0],
            steps: 0,
        };''','''        let mut env = Self::from_state(model, config, Vector4::zeros(), 0);''')
edit(env,'    pub fn model(&self) -> Model {','''    /// Build an environment around a known physical state without drawing noise
    /// or resetting. Used by the live viewer to share the exact transition path.
    /// This is state transfer, not an optimizer/RNG resume checkpoint.
    pub fn from_state(model: Model, config: PendulumEnvConfig, state: Vector4, steps: usize) -> Self {
        CartPoleParameters::from(model).validate();
        assert!(config.dt.is_finite() && config.dt > 0.0, "invalid pendulum dt");
        assert!(config.max_force.is_finite() && config.max_force > 0.0, "invalid force limit");
        Self { model, config, state, steps, last_applied_force: 0.0 }
    }

    /// Actual force after action clipping, actuation noise and disturbances.
    pub fn last_applied_force(&self) -> f32 { self.last_applied_force }

    pub fn model(&self) -> Model {''')
edit(env,'''        let (a, b) = self.model.model(self.config.dt);
        self.state = a * self.state + b * applied_force;''','''        self.last_applied_force = applied_force;
        self.state = CartPoleParameters::from(self.model).step(self.state, applied_force, self.config.dt);''')
edit(env,'''        self.steps = 0;
        self.observation_with_rng(rng)''','''        self.steps = 0;
        self.last_applied_force = 0.0;
        self.observation_with_rng(rng)''')

# Live policy execution now delegates the complete transition/noise/reward/boundary
# contract to PendulumEnv. Classical controllers keep their independent model tuning.
domain='rust_robotics_sim/src/simulator/pendulum/domain.rs'
edit(domain,'use rust_robotics_train::PpoTrainerConfig;',
    'use rust_robotics_train::{PpoTrainerConfig, PendulumEnv, PendulumEnvConfig};\nuse rust_robotics_algo::cart_pole::CartPoleParameters;')
edit(domain,'    pub(crate) visual_episode_steps: usize,','''    pub(crate) visual_episode_steps: usize,
    pub(crate) policy_observation: Option<[f32; 4]>,
    pub(crate) active_training_environment: Option<(CartPoleParameters, PendulumEnvConfig)>,
    pub(crate) policy_environment_stale: bool,''')
edit(domain,'            visual_episode_steps: 0,','''            visual_episode_steps: 0,
            policy_observation: None,
            active_training_environment: None,
            policy_environment_stale: false,''')
p=Path(domain);s=p.read_text()
start=s.index('    /// Continuous-time nonlinear cart-pole dynamics.')
end=s.index('    #[cfg(target_arch = "wasm32")]\n    fn policy_trainer_snapshot', start)
s=s[:start]+'''    /// Classical controllers and PPO use the same physical RK4 plant.
    fn integrate_plant_rk4(&self, state: State, control: f32, dt: f32) -> State {
        CartPoleParameters::from(self.model).step(state, control, dt)
    }

'''+s[end:];p.write_text(s)
edit(domain,'            last_error: self.trainer_backend.last_error().map(str::to_owned),','''            last_error: if self.policy_environment_stale {
                Some("Plant/noise changed: restart PPO training before using this policy.".to_owned())
            } else { self.trainer_backend.last_error().map(str::to_owned) },''')
edit(domain,'        if let Some(action) = patch.trainer_action {','''        self.validate_training_environment();
        if let Some(action) = patch.trainer_action {''')
edit(domain,'''    pub fn step_with_noise(&mut self, dt: f32, noise: NoiseConfig) {
        let mut measured_state''','''    pub fn step_with_noise(&mut self, dt: f32, noise: NoiseConfig) {
        self.configure_training_noise(noise);
        self.validate_training_environment();
        if self.controller.kind() == ControllerKind::Policy {
            self.step_policy_with_rng(dt, &mut rand::thread_rng());
            return;
        }
        let mut measured_state''')
edit(domain,'    pub fn set_policy_controller(&mut self, snapshot: &PolicySnapshot) {','''    /// The exact PPO environment path, including pre-reset observations and
    /// termination after Stop. The caller owns noise draws; no GUI clock is used.
    pub(crate) fn step_policy_with_rng<R: Rng + ?Sized>(&mut self, dt: f32, rng: &mut R) {
        self.validate_training_environment();
        if self.policy_environment_stale {
            self.last_control_error = Some("Plant/noise changed: restart PPO training.".to_owned());
            return;
        }
        assert_eq!(dt, self.trainer_config.env.dt, "PPO live timestep must match training");
        let mut env = PendulumEnv::from_state(self.model, self.trainer_config.env,
            self.state, self.visual_episode_steps);
        let observation = self.policy_observation.take().unwrap_or_else(|| env.observation_with_rng(rng));
        let Controller::Policy(policy) = &self.controller else { unreachable!("PPO path only"); };
        let result = env.step_with_rng(policy.act(observation), rng);
        self.state = env.state();
        self.visual_episode_steps += 1;
        self.policy_observation = Some(result.observation);
        self.last_control_error = None;
        self.data.add(self.data.time_last() + dt, vec![self.state[0], self.state[1],
            self.state[2], self.state[3], env.last_applied_force()]);
        if result.done {
            self.policy_observation = Some(env.reset_with_rng(rng));
            self.state = env.state();
            self.visual_episode_steps = 0;
            self.time_init = 0.0;
            self.data.clear();
        }
    }

    pub fn set_policy_controller(&mut self, snapshot: &PolicySnapshot) {''')
edit(domain,'''            self.state.clone_from(data);''','''            self.state.clone_from(data);
            self.policy_observation = None;''')
edit(domain,'''        self.state = vector![0., 0., rand(0.4), 0.];
        self.time_init = 0.0;''','''        if self.controller.kind() == ControllerKind::Policy {
            let mut rng = rand::thread_rng();
            let mut env = PendulumEnv::from_state(self.model, self.trainer_config.env, self.state, 0);
            self.policy_observation = Some(env.reset_with_rng(&mut rng));
            self.state = env.state();
        } else {
            self.state = vector![0., 0., rand(0.4), 0.];
            self.policy_observation = None;
        }
        self.time_init = 0.0;''')

training='rust_robotics_sim/src/simulator/pendulum/training.rs'
edit(training,'use super::domain::{ControllerKind, InvertedPendulum, PENDULUM_FIXED_DT};',
    '''use super::domain::{ControllerKind, InvertedPendulum, NoiseConfig, PENDULUM_FIXED_DT};
use rust_robotics_algo::cart_pole::CartPoleParameters;
use rust_robotics_train::PendulumEnvConfig;''')
edit(training,'impl InvertedPendulum {','''impl InvertedPendulum {
    /// Keep physical settings explicit and reject a stale trainer/policy rather
    /// than silently deploying it on a different task. Restart is user-visible.
    pub(crate) fn validate_training_environment(&mut self) {
        self.trainer_config.plant = CartPoleParameters::from(self.model);
        let requested = (self.trainer_config.plant, self.trainer_config.env);
        if self.active_training_environment.is_some_and(|active| active != requested) {
            self.training_active = false;
            self.trainer_backend.destroy();
            self.active_training_environment = None;
            self.policy_observation = None;
            self.policy_environment_stale = true;
            self.last_control_error = Some("Plant/noise changed: restart PPO training.".to_owned());
        }
    }

    /// The global noise control affects BOTH PPO training and live inference.
    /// The original PPO amplitudes are the scale-one reference; no hidden second
    /// PPO disturbance model lives in the viewer.
    pub(crate) fn configure_training_noise(&mut self, noise: NoiseConfig) {
        let scale = if noise.enabled { noise.scale.max(0.0) } else { 0.0 };
        let base = PendulumEnvConfig::default();
        let env = &mut self.trainer_config.env;
        let previous = *env;
        env.observation_position_noise_m = base.observation_position_noise_m * scale;
        env.observation_velocity_noise_mps = base.observation_velocity_noise_mps * scale;
        env.observation_angle_noise_rad = base.observation_angle_noise_rad * scale;
        env.observation_angular_velocity_noise_radps = base.observation_angular_velocity_noise_radps * scale;
        env.action_noise_force_n = base.action_noise_force_n * scale;
        env.disturbance_force_n = base.disturbance_force_n * scale;
        env.disturbance_probability_per_step = if scale > 0.0 { base.disturbance_probability_per_step } else { 0.0 };
        if previous != *env { self.policy_observation = None; }
        self.validate_training_environment();
    }
''')
edit(training,'''    pub fn tick_training(&mut self) {
        if self.training_active''','''    pub fn tick_training(&mut self) {
        self.validate_training_environment();
        if self.policy_environment_stale { return; }
        if self.training_active''')
edit(training,'''    pub(crate) fn reset_trainer(&mut self) {
        self.trainer_config.env.dt = PENDULUM_FIXED_DT;''','''    pub(crate) fn reset_trainer(&mut self) {
        self.trainer_config.env.dt = PENDULUM_FIXED_DT;
        self.trainer_config.plant = CartPoleParameters::from(self.model);
        self.active_training_environment = Some((self.trainer_config.plant, self.trainer_config.env));
        self.policy_environment_stale = false;
        self.last_control_error = None;
        self.policy_observation = None;''')
edit(training,'''    pub(crate) fn start_training(&mut self) {
        self.controller_selection''','''    pub(crate) fn start_training(&mut self) {
        self.validate_training_environment();
        self.controller_selection''')
edit(training,'''    pub(crate) fn sync_policy_selection(&mut self) {
        if self.controller_selection''','''    pub(crate) fn sync_policy_selection(&mut self) {
        self.validate_training_environment();
        if self.policy_environment_stale { return; }
        if self.controller_selection''')

runtime='rust_robotics_sim/src/simulator/runtime.rs'
edit(runtime,'''        self.pendulum_noise_enabled = enabled;''','''        self.pendulum_noise_enabled = enabled;
        let noise = self.pendulum_noise_config();
        for pendulum in &mut self.simulations.pendulums { pendulum.configure_training_noise(noise); }''')
edit(runtime,'''        self.pendulum_noise_scale = scale.clamp(0.0, 3.0);''','''        self.pendulum_noise_scale = scale.clamp(0.0, 3.0);
        let noise = self.pendulum_noise_config();
        for pendulum in &mut self.simulations.pendulums { pendulum.configure_training_noise(noise); }''')
edit(runtime,'''                        .for_each(InvertedPendulum::tick_training);''','''                        .for_each(|pendulum| { pendulum.configure_training_noise(noise); pendulum.tick_training(); });''')
edit(runtime,'''        if self.mode == SimMode::InvertedPendulum && self.paused {
            self.simulations''','''        if self.mode == SimMode::InvertedPendulum && self.paused {
            let noise = self.pendulum_noise_config();
            self.simulations''')
edit(runtime,'''                .for_each(InvertedPendulum::tick_training);''','''                .for_each(|pendulum| { pendulum.configure_training_noise(noise); pendulum.tick_training(); });''')
ui='rust_robotics_sim/src/simulator/pendulum/ui.rs'
edit(ui,'''    fn show_controller_summary(&self, ui: &mut Ui, available_policy: Option<&PolicySnapshot>) {
        match''','''    fn show_controller_summary(&self, ui: &mut Ui, available_policy: Option<&PolicySnapshot>) {
        if self.policy_environment_stale {
            ui.label("Plant/noise changed. Restart PPO training; old policy execution is disabled.");
        }
        match''')
edit(ui,'''        ui.set_width(controller_width);''','''        if self.policy_environment_stale {
            ui.label("Plant/noise changed. Restart PPO training; old policy execution is disabled.");
        }
        ui.set_width(controller_width);''')

# Regression files are part of the final source commit, not disposable checks.
shutil.copyfile('audits/plant_fix/contract_tests.rs','rust_robotics_sim/src/simulator/pendulum/contract_tests.rs')
with open('rust_robotics_sim/src/simulator/pendulum/mod.rs','a') as f:
    f.write('\n#[cfg(test)]\nmod contract_tests;\n')
shutil.copyfile('audits/plant_fix/plant_tests.rs','rust_robotics_train/src/plant_tests.rs')
with open(trainer,'a') as f: f.write('\n#[cfg(test)]\n#[path = "plant_tests.rs"]\nmod plant_tests;\n')
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
print('SHARED PLANT PATCH APPLIED: ordinary PPO equations and hyperparameters unchanged')
