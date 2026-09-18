"""Reproduce the first reviewed patch, then apply explicit contract corrections.
The historical preparation is pinned; the final production commit contains only
ordinary source and permanent tests, not this generator or its audit history.
"""
from pathlib import Path
import subprocess
original = subprocess.check_output(['git','show',
    '08fa3d57e01cc176d90fef2c30875461eec5669e:audits/plant_fix/apply.py']).decode()
exec(compile(original, 'original-plant-patch.py', 'exec'))

# RNG reference tests independently reproduce draw order, not the old physical
# discretization. Shared-plant dynamics have independent Lagrange/RK4 tests.
seed = 'rust_robotics_train/src/ppo_seed_tests.rs'
edit(seed, 'use rust_robotics_algo::{control::StateSpace, prelude::Vector4};',
     'use rust_robotics_algo::{cart_pole::CartPoleParameters, prelude::Vector4};')
edit(seed, '''        // Reuse only the unchanged plant matrix, not production noise helpers.
        let (a, b) = env.model().model(c.dt);
        let next = a * Vector4::from_column_slice(&x) + b * (clipped + action_noise + disturbance);''',
    '''        // Reuse only the deterministic plant, never production noise helpers.
        // Independent physics checks live in cart_pole::tests.
        let next = CartPoleParameters::from(env.model()).step(
            Vector4::from_column_slice(&x), clipped + action_noise + disturbance, c.dt);''')
# Guarantee a single negative terminal episode independently of whether the
# integrator moves position during the first step. Keep the original assertion
# that the first negative return becomes best (rather than max(0, return)).
rollout = 'rust_robotics_train/src/ppo_rollout_tests.rs'
edit(rollout, '''    env.max_angle_rad = 0.0;
    let mut session = session(env, 2);
    let expected = trace(&session, 2).0;''',
    '''    env.max_angle_rad = 0.0;
    env.reset_angle_range_rad = 0.1;
    let mut session = session(env, 1);
    let expected = trace(&session, 1).0;''')

# Centralize activation. Publishing a new snapshot while already in Policy mode
# must not reset the trajectory, but entering Policy mode must use its reset law.
domain = 'rust_robotics_sim/src/simulator/pendulum/domain.rs'
edit(domain, '''    pub fn set_policy_controller(&mut self, snapshot: &PolicySnapshot) {
        self.controller = Controller::policy(snapshot.clone());
    }''', '''    pub fn set_policy_controller(&mut self, snapshot: &PolicySnapshot) {
        let entering_policy = self.controller.kind() != ControllerKind::Policy;
        self.controller = Controller::policy(snapshot.clone());
        self.controller_selection = ControllerKind::Policy;
        if entering_policy { self.reset_state(); }
    }''')
# Expose the seeded wrapper internally so tests exercise the same path used by
# the application, not merely a hand-copied dynamics helper.
edit(domain, '''    pub fn step_with_noise(&mut self, dt: f32, noise: NoiseConfig) {
        self.configure_training_noise(noise);''', '''    pub fn step_with_noise(&mut self, dt: f32, noise: NoiseConfig) {
        self.step_with_rng(dt, noise, &mut rand::thread_rng());
    }

    pub(crate) fn step_with_rng<R: Rng + ?Sized>(&mut self, dt: f32, noise: NoiseConfig, rng: &mut R) {
        self.configure_training_noise(noise);''')
edit(domain, '            self.step_policy_with_rng(dt, &mut rand::thread_rng());',
     '            self.step_policy_with_rng(dt, rng);')
edit(domain, '''        assert_eq!(dt, self.trainer_config.env.dt, "PPO live timestep must match training");''',
    '''        if self.active_training_environment.is_some() && self.trainer_backend.snapshot().is_none() {
            // A web reset may not have published its new snapshot yet. Never
            // execute retained old weights on the newly configured environment.
            self.last_control_error = Some("Waiting for the matching PPO policy snapshot.".to_owned());
            return;
        }
        assert_eq!(dt, self.trainer_config.env.dt, "PPO live timestep must match training");''')
# Original patch has already been rustfmt'd; use the formatted anchor if needed.

runtime = 'rust_robotics_sim/src/simulator/runtime.rs'
p = Path(runtime); text = p.read_text()
for method, call in [('set_pendulum_controller_kind', '            pendulum.select_controller_kind(kind);'),
                     ('patch_pendulum', '            pendulum.apply_patch(patch);')]:
    start = text.index('    pub(crate) fn ' + method)
    end = text.index('\n    }', start) + len('\n    }')
    section = text[start:end]
    assert section.count('        if let Some(pendulum)') == 1
    section = section.replace('        if let Some(pendulum)',
        '        let noise = self.pendulum_noise_config();\n        if let Some(pendulum)')
    assert section.count(call) == 1
    section = section.replace(call, '            pendulum.configure_training_noise(noise);\n' + call)
    text = text[:start] + section + text[end:]
p.write_text(text)

# Adjust the explicit-state fixture for activation now doing a real reset.
contract = Path('rust_robotics_sim/src/simulator/pendulum/contract_tests.rs')
text = contract.read_text()
anchor = '        sim.set_policy_controller(&policy);'
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + '\n        sim.state = state;\n        sim.policy_observation = None;')
anchor = '            sim.step_policy_with_rng(PENDULUM_FIXED_DT, &mut b);'
assert text.count(anchor) == 1
text = text.replace(anchor, '''            sim.step_with_rng(PENDULUM_FIXED_DT,
                NoiseConfig { enabled: true, scale: 1.0 }, &mut b);''')
text += '''

#[test]
fn activating_policy_resets_but_publishing_new_weights_does_not() {
    let mut sim = InvertedPendulum { state: Vector4::new(3.0, 5.0, 0.8, 6.0),
        visual_episode_steps: 17, ..Default::default() };
    let policy = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), 17).snapshot();
    sim.set_policy_controller(&policy);
    assert_eq!(sim.controller_selection, super::domain::ControllerKind::Policy);
    assert_eq!(sim.visual_episode_steps, 0);
    assert!(sim.state[0].abs() <= sim.trainer_config.env.reset_position_range_m);
    assert!(sim.state[2].abs() <= sim.trainer_config.env.reset_angle_range_rad);
    let state = sim.state;
    let observation = sim.policy_observation;
    sim.set_policy_controller(&policy);
    assert_eq!(sim.state, state);
    assert_eq!(sim.policy_observation, observation);
}

#[test]
fn pending_replacement_snapshot_does_not_run_old_policy() {
    let mut sim = InvertedPendulum::default();
    sim.start_training();
    sim.trainer_backend.destroy();
    // The environment contract remains active but there is no published policy:
    // this models the browser worker creation interval without mocking physics.
    let state = sim.state;
    sim.step_policy_with_rng(PENDULUM_FIXED_DT, &mut StdRng::seed_from_u64(33));
    assert_eq!(sim.state, state);
    assert!(sim.last_control_error().unwrap().contains("Waiting"));
}
'''
contract.write_text(text)
subprocess.run(['cargo','fmt','--all'],check=True)
subprocess.run(['git','diff','--check'],check=True)
print('ACTIVATION AND LINEAR-FIXTURE MIGRATION APPLIED; no learning thresholds changed')
