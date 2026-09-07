"""Capture the old MPC behavior, then generate and harden the prepared runtime."""
from pathlib import Path
import math
import re
import runpy
import subprocess

BASELINE_SHA = "7086dc7b6d3e18681b0ec9b034358a6cd69986b6"
mpc = Path("rust_robotics_algo/src/control/inverted_pendulum/mpc.rs")
domain = Path("rust_robotics_sim/src/simulator/pendulum/domain.rs")
tests = Path("rust_robotics_sim/src/simulator/pendulum/tests.rs")
original = mpc.read_text()
if "pub struct PreparedMpc" in original:
    raise SystemExit("Expected the pre-refactor MPC implementation")

cases = r'''
    fn reference_models() -> [Model; 2] {
        let tuned = Model {
            l_bar: 1.5,
            m_cart: 1.2,
            m_ball: 0.8,
            Q: diag![1.0, 1.0, 10.0, 1.0],
            R: diag![0.1],
            ..Model::default()
        };
        [Model::default(), tuned]
    }

    fn reference_states() -> [Vector4; 5] {
        [
            vector![0.0, 0.0, 0.0, 0.0],
            vector![0.0, 0.1, 0.1, 0.0],
            vector![0.5, -0.2, 0.05, 0.1],
            vector![-0.5, 0.3, -0.1, -0.2],
            vector![2.0, 1.0, 0.6, 0.3],
        ]
    }
'''
emitter = r'''
    #[test]
    fn emit_pre_refactor_mpc_baseline() {
        let mut index = 0;
        for model in reference_models() {
            for dt in [0.01, 0.05, 0.1] {
                for state in reference_states() {
                    let control = try_mpc_control(state, model, dt)
                        .expect("pre-refactor MPC reference must solve");
                    assert!(control.is_finite());
                    println!("PREPARED_MPC_BASELINE {index} {control:?}");
                    index += 1;
                }
            }
        }
    }
'''
# Run the unchanged numerical implementation before invoking the refactor.
try:
    mpc.write_text(original + "\n#[cfg(test)]\nmod prepared_mpc_reference {\n    use super::*;\n" + cases + emitter + "}\n")
    result = subprocess.run(
        ["cargo", "test", "-p", "rust_robotics_algo", "emit_pre_refactor_mpc_baseline",
         "--", "--nocapture", "--test-threads=1"],
        capture_output=True, text=True, check=False,
    )
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    result.check_returncode()
finally:
    mpc.write_text(original)

matches = re.findall(r"PREPARED_MPC_BASELINE (\d+) (\S+)", result.stdout)
if len(matches) != 30 or [int(index) for index, _ in matches] != list(range(30)):
    raise SystemExit(f"Expected 30 ordered baseline outputs, got {matches!r}")
values = [float(value) for _, value in matches]
if not all(math.isfinite(value) for value in values):
    raise SystemExit("Non-finite baseline output")
print(f"Captured {len(values)} independent MPC reference outputs from {BASELINE_SHA}", flush=True)

runpy.run_path(".github/scripts/implement_prepared_mpc.py", run_name="__main__")


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one marker, got {count}: {old!r}")
    return text.replace(old, new, 1)


# Count actual construction, not just a matching key, in test builds only.
d = domain.read_text()
d = replace_once(d,
    "    pub(crate) mpc_cache: Option<MpcRuntimeCache>,\n",
    "    pub(crate) mpc_cache: Option<MpcRuntimeCache>,\n"
    "    #[cfg(test)]\n"
    "    pub(crate) mpc_preparations: usize,\n")
d = replace_once(d,
    "            mpc_cache: None,\n",
    "            mpc_cache: None,\n"
    "            #[cfg(test)]\n"
    "            mpc_preparations: 0,\n")
d = replace_once(d,
    "                        prepared: PreparedMpc::new(*model, dt),\n                    });\n",
    "                        prepared: PreparedMpc::new(*model, dt),\n                    });\n"
    "                    #[cfg(test)]\n"
    "                    {\n"
    "                        self.mpc_preparations += 1;\n"
    "                    }\n")
domain.write_text(d)

# Replace the two generated tests, preserving all pre-existing tests.
t = tests.read_text()
marker = "\n#[test]\nfn pendulum_runtime_prepares_and_reuses_mpc_configuration() {"
if t.count(marker) != 1:
    raise SystemExit("Generated MPC test marker not found uniquely")
t = t[:t.index(marker)]
t += r'''

fn default_mpc_model() -> Model {
    match Controller::mpc(Model::default()) {
        Controller::MPC(model) => model,
        _ => unreachable!("MPC constructor must select MPC"),
    }
}

fn mpc_sim(model: Model) -> InvertedPendulum {
    InvertedPendulum {
        state: vector![0.0, 0.0, 0.05, 0.0],
        controller: Controller::MPC(model),
        controller_selection: ControllerKind::Mpc,
        ..Default::default()
    }
}

#[test]
fn pendulum_runtime_prepares_and_reuses_mpc_configuration() {
    let model = default_mpc_model();
    let mut sim = mpc_sim(model);

    assert!(!sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 0);
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 1);
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 1, "unchanged configuration must be reused");
    assert!(sim.last_control_error().is_none());
}

#[test]
fn mpc_runtime_reprepares_when_model_changes() {
    let original = default_mpc_model();
    let mut sim = mpc_sim(original);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 1);

    let mut updated = original;
    updated.Q[10] *= 1.5;
    sim.controller = Controller::MPC(updated);
    sim.step(PENDULUM_FIXED_DT);

    assert!(!sim.mpc_cache_matches(original, PENDULUM_FIXED_DT));
    assert!(sim.mpc_cache_matches(updated, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 2);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 2);
    assert!(sim.last_control_error().is_none());
}

#[test]
fn mpc_runtime_reprepares_when_timestep_changes() {
    let model = default_mpc_model();
    let mut sim = mpc_sim(model);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 1);

    let updated_dt = PENDULUM_FIXED_DT * 0.5;
    sim.step(updated_dt);
    assert!(!sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert!(sim.mpc_cache_matches(model, updated_dt));
    assert_eq!(sim.mpc_preparations, 2);
    sim.step(updated_dt);
    assert_eq!(sim.mpc_preparations, 2);
    assert!(sim.last_control_error().is_none());
}

#[test]
fn switching_away_from_mpc_discards_its_prepared_configuration() {
    let model = default_mpc_model();
    for controller in [
        Controller::lqr(Model::default()),
        Controller::pid(),
        Controller::policy(policy_snapshot(0.0)),
    ] {
        let mut sim = mpc_sim(model);
        sim.step(PENDULUM_FIXED_DT);
        assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
        assert_eq!(sim.mpc_preparations, 1);

        sim.controller = controller;
        sim.step(PENDULUM_FIXED_DT);
        assert!(sim.mpc_cache.is_none());
        assert_eq!(sim.mpc_preparations, 1);

        sim.controller = Controller::MPC(model);
        sim.step(PENDULUM_FIXED_DT);
        assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
        assert!(sim.lqr_cache.is_none());
        assert_eq!(sim.mpc_preparations, 2);
        assert!(sim.last_control_error().is_none());
    }
}

#[test]
fn resetting_mpc_clears_prepared_configuration_and_error() {
    let model = default_mpc_model();
    let mut sim = mpc_sim(model);
    sim.step(PENDULUM_FIXED_DT);
    assert_eq!(sim.mpc_preparations, 1);
    sim.last_control_error = Some("stale failure".to_owned());

    sim.reset_state();
    assert!(sim.mpc_cache.is_none());
    assert!(sim.lqr_cache.is_none());
    assert!(sim.last_control_error().is_none());

    // Keep the rollout deterministic despite reset's randomized starting angle.
    sim.state = vector![0.0, 0.0, 0.05, 0.0];
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    assert_eq!(sim.mpc_preparations, 2);
    assert!(sim.last_control_error().is_none());
}

#[test]
fn cached_mpc_rollout_matches_fresh_preparation_each_step() {
    let model = default_mpc_model();
    let mut cached = mpc_sim(model);
    let mut fresh = mpc_sim(model);
    for _ in 0..12 {
        fresh.mpc_cache = None;
        cached.step(PENDULUM_FIXED_DT);
        fresh.step(PENDULUM_FIXED_DT);
        assert!(cached.last_control_error().is_none());
        assert!(fresh.last_control_error().is_none());
        assert!((cached.state - fresh.state).norm() < 1e-6);
    }
    assert_eq!(cached.mpc_preparations, 1);
    assert_eq!(fresh.mpc_preparations, 12);
}
'''
tests.write_text(t)

# Persist independent, pre-refactor fixtures as a normal regression test.
s = mpc.read_text()
s = replace_once(s,
    "/// horizon matrix assembly, and dense-to-CSC conversion on every control tick.\n",
    "/// horizon matrix assembly, and dense-to-CSC conversion on every control tick.\n"
    "/// A fresh Clarabel solver is still constructed for each solve; this type does\n"
    "/// not cache solver factorization or warm-start the optimization.\n")
expected = ",\n".join("        " + repr(value) for value in values)
regression = r'''
    #[test]
    fn prepared_mpc_preserves_pre_refactor_outputs() {
        let expected: [f32; 30] = [
__EXPECTED__
        ];
        let mut index = 0;
        for model in reference_models() {
            for dt in [0.01, 0.05, 0.1] {
                let prepared = PreparedMpc::new(model, dt);
                for state in reference_states() {
                    let actual = prepared.try_control(state).expect("prepared MPC reference solve");
                    assert!(actual.is_finite());
                    assert!(actual.abs() <= 50.0001, "actuator bound: {actual}");
                    assert!(
                        (actual - expected[index]).abs() < 1e-4,
                        "case {index}: actual={actual}, expected={}", expected[index]
                    );
                    index += 1;
                }
            }
        }
        assert_eq!(index, expected.len());
    }
'''.replace("__EXPECTED__", expected)
s += ("\n#[cfg(test)]\nmod prepared_mpc_regression_tests {\n    use super::*;\n"
      + f"\n    // Reference outputs captured from the pre-refactor implementation at {BASELINE_SHA}.\n"
      + cases + regression + "}\n")
mpc.write_text(s)
print("Prepared MPC generation, independent fixtures, and cache lifecycle tests are ready for validation", flush=True)
