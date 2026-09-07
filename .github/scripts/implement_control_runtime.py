from pathlib import Path

# Tighten the public MPC diagnostic wording while retaining the verified API.
mpc = Path("rust_robotics_algo/src/control/inverted_pendulum/mpc.rs")
s = mpc.read_text()
s = s.replace(
    "/// Returns the underlying solver status as a stable diagnostic string.",
    "/// Returns the underlying solver status as a diagnostic string.",
)
mpc.write_text(s)

p = Path("rust_robotics_sim/src/simulator/pendulum/domain.rs")
s = p.read_text()

marker = "fn mpc_stage_cost_model(mut model: Model) -> Model {\n"
cache = """#[derive(Debug, Clone)]
struct LqrRuntimeCache {
    model: Model,
    dt: f32,
    prepared: rb::control::lqr::PreparedLqr<4, 1>,
}

"""
if "struct LqrRuntimeCache" not in s:
    s = s.replace(marker, cache + marker, 1)

old_card = """    pub(crate) controller_params: ControllerParamsSnapshot,
    pub(crate) policy_trainer: PolicyTrainerSnapshot,
}"""
new_card = """    pub(crate) controller_params: ControllerParamsSnapshot,
    pub(crate) policy_trainer: PolicyTrainerSnapshot,
    pub(crate) control_error: Option<String>,
}"""
if old_card in s:
    s = s.replace(old_card, new_card, 1)

old_fields = """    pub(crate) parallel_trainers: usize,
}"""
new_fields = """    pub(crate) parallel_trainers: usize,
    lqr_cache: Option<LqrRuntimeCache>,
    pub(crate) last_control_error: Option<String>,
}"""
if old_fields not in s:
    raise SystemExit("InvertedPendulum field marker not found")
s = s.replace(old_fields, new_fields, 1)

old_default = """            training_updates_per_tick: 1,
            parallel_trainers: 1,
        }"""
new_default = """            training_updates_per_tick: 1,
            parallel_trainers: 1,
            lqr_cache: None,
            last_control_error: None,
        }"""
if old_default not in s:
    raise SystemExit("default marker not found")
s = s.replace(old_default, new_default, 1)

old_card_build = """            controller_params,
            policy_trainer: self.policy_trainer_snapshot(),
        }"""
new_card_build = """            controller_params,
            policy_trainer: self.policy_trainer_snapshot(),
            control_error: self.last_control_error.clone(),
        }"""
if old_card_build in s:
    s = s.replace(old_card_build, new_card_build, 1)

old_control = """        let control_command = self.controller.control(measured_state, dt);
        let applied_control = if noise.is_active() {"""
new_control = """        let control_command = match &mut self.controller {
            Controller::LQR(model) => {
                let needs_prepare = self.lqr_cache.as_ref().is_none_or(|cache| {
                    cache.model != *model || cache.dt.to_bits() != dt.to_bits()
                });
                if needs_prepare {
                    self.lqr_cache = Some(LqrRuntimeCache {
                        model: *model,
                        dt,
                        prepared: model.prepare(dt),
                    });
                }
                self.last_control_error = None;
                self.lqr_cache
                    .as_ref()
                    .expect("LQR cache is prepared above")
                    .prepared
                    .control(measured_state)[0]
            }
            Controller::PID(pid) => {
                self.lqr_cache = None;
                self.last_control_error = None;
                pid.control(-measured_state[2], dt)
            }
            Controller::MPC(model) => {
                self.lqr_cache = None;
                match try_mpc_control(measured_state, *model, dt) {
                    Ok(control) => {
                        self.last_control_error = None;
                        control
                    }
                    Err(error) => {
                        self.last_control_error = Some(error.to_string());
                        0.0
                    }
                }
            }
            Controller::Policy(policy) => {
                self.lqr_cache = None;
                self.last_control_error = None;
                policy.act([
                    measured_state[0],
                    measured_state[1],
                    measured_state[2],
                    measured_state[3],
                ])
            }
        };
        let applied_control = if noise.is_active() {"""
if old_control not in s:
    raise SystemExit("step control marker not found")
s = s.replace(old_control, new_control, 1)

old_reset = """        self.controller.reset_state();
        self.data.clear();"""
new_reset = """        self.controller.reset_state();
        self.lqr_cache = None;
        self.last_control_error = None;
        self.data.clear();"""
if old_reset not in s:
    raise SystemExit("reset marker not found")
s = s.replace(old_reset, new_reset, 1)

getter_marker = "    pub(crate) fn visual_episode_done(&self) -> bool {"
getter = """    #[cfg(test)]
    pub(crate) fn lqr_cache_matches(&self, model: Model, dt: f32) -> bool {
        self.lqr_cache.as_ref().is_some_and(|cache| {
            cache.model == model && cache.dt.to_bits() == dt.to_bits()
        })
    }

    pub fn last_control_error(&self) -> Option<&str> {
        self.last_control_error.as_deref()
    }

"""
if "pub(crate) fn lqr_cache_matches" not in s:
    s = s.replace(getter_marker, getter + getter_marker, 1)

p.write_text(s)

tests = Path("rust_robotics_sim/src/simulator/pendulum/tests.rs")
t = tests.read_text()
additions = """
#[test]
fn pendulum_runtime_prepares_and_reuses_lqr_configuration() {
    let model = Model::default();
    let mut sim = InvertedPendulum {
        state: vector![0.1, 0.0, 0.05, 0.0],
        controller: Controller::lqr(model),
        ..Default::default()
    };

    assert!(!sim.lqr_cache_matches(model, PENDULUM_FIXED_DT));
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.lqr_cache_matches(model, PENDULUM_FIXED_DT));

    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.lqr_cache_matches(model, PENDULUM_FIXED_DT));
}

#[test]
fn lqr_runtime_reprepares_when_model_changes() {
    let original = Model::default();
    let mut sim = InvertedPendulum {
        controller: Controller::lqr(original),
        ..Default::default()
    };
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.lqr_cache_matches(original, PENDULUM_FIXED_DT));

    let mut updated = original;
    updated.Q[10] *= 1.5;
    sim.controller = Controller::lqr(updated);
    sim.step(PENDULUM_FIXED_DT);

    assert!(!sim.lqr_cache_matches(original, PENDULUM_FIXED_DT));
    assert!(sim.lqr_cache_matches(updated, PENDULUM_FIXED_DT));
}

#[test]
fn successful_mpc_runtime_clears_control_error() {
    let mut sim = InvertedPendulum {
        state: vector![0.0, 0.0, 0.05, 0.0],
        controller: Controller::mpc(Model::default()),
        last_control_error: Some("stale error".to_owned()),
        ..Default::default()
    };

    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.last_control_error().is_none());
}
"""
if "pendulum_runtime_prepares_and_reuses_lqr_configuration" not in t:
    t += additions
tests.write_text(t)
