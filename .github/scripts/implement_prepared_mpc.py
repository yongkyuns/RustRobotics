from pathlib import Path

mpc = Path("rust_robotics_algo/src/control/inverted_pendulum/mpc.rs")
s = mpc.read_text()

start = s.index("/// MPC control using quadratic programming with the Clarabel solver.\n")
end = s.index("/// Compatibility helper that preserves the historical zero-force fallback.\n")

prepared = r'''/// MPC problem data that is constant for a fixed model and timestep.
///
/// Preparing the problem once avoids repeating the terminal Riccati solve,
/// horizon matrix assembly, and dense-to-CSC conversion on every control tick.
pub struct PreparedMpc {
    p: clarabel::algebra::CscMatrix<f64>,
    q: Vec<f64>,
    a: clarabel::algebra::CscMatrix<f64>,
    b_template: Vec<f64>,
}

impl PreparedMpc {
    /// Builds the static quadratic-program data for a fixed model and timestep.
    pub fn new(model: Model, dt: f32) -> Self {
        use crate::control::LQR;
        use clarabel::algebra::*;

        let (Ad, Bd) = model.model(dt);
        let umin = -50.0_f64;
        let umax = 50.0_f64;
        let Q = model.Q;
        let R = model.R;
        let QN = model.solve_DARE(Ad, Bd);

        let P_mat = block_diag!(kron!(eye!(N), Q), QN, kron!(eye!(N), R));

        // The pendulum MPC reference is the zero state, so q is identically zero.
        let q = vec![0.0; N_VARS];

        let Ax = kron!(eye!(N + 1), -eye!(NX)) + kron!(eye!({ N + 1 }, -1), Ad);
        let Bu = kron!(vstack!(zeros!(1, N), eye!(N)), Bd);
        let Aeq = hstack!(Ax, Bu);

        let zeros_xu = zeros!({ N * NU }, { (N + 1) * NX });
        let eye_u = eye!(N * NU);
        let Aineq_upper = hstack!(zeros_xu, eye_u);
        let Aineq_lower = hstack!(zeros_xu, -eye_u);
        let A_mat = vstack!(Aeq, vstack!(Aineq_upper, Aineq_lower));

        let mut p_row_indices = Vec::new();
        let mut p_col_ptrs = vec![0];
        let mut p_values = Vec::new();
        for j in 0..N_VARS {
            for i in 0..=j {
                let val = P_mat[(i, j)] as f64;
                if val.abs() > 1e-12 {
                    p_row_indices.push(i);
                    p_values.push(val);
                }
            }
            p_col_ptrs.push(p_row_indices.len());
        }
        let p = CscMatrix::new(N_VARS, N_VARS, p_col_ptrs, p_row_indices, p_values);

        let n_constraints = N_EQ + N_INEQ;
        let mut a_row_indices = Vec::new();
        let mut a_col_ptrs = vec![0];
        let mut a_values = Vec::new();
        for j in 0..N_VARS {
            for i in 0..n_constraints {
                let val = A_mat[(i, j)] as f64;
                if val.abs() > 1e-12 {
                    a_row_indices.push(i);
                    a_values.push(val);
                }
            }
            a_col_ptrs.push(a_row_indices.len());
        }
        let a = CscMatrix::new(
            n_constraints,
            N_VARS,
            a_col_ptrs,
            a_row_indices,
            a_values,
        );

        let mut b_template = vec![0.0; n_constraints];
        for k in 0..N {
            b_template[N_EQ + k] = umax;
            b_template[N_EQ + N * NU + k] = -umin;
        }

        Self {
            p,
            q,
            a,
            b_template,
        }
    }

    /// Solves the prepared MPC problem for the current measured state.
    pub fn try_control(&self, x: Vector4) -> Result<f32, MpcSolveError> {
        use clarabel::solver::*;

        let mut b = self.b_template.clone();
        for i in 0..NX {
            b[i] = -(x[i] as f64);
        }

        let cones = [ZeroConeT(N_EQ), NonnegativeConeT(N_INEQ)];
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .build()
            .unwrap();
        let mut solver = DefaultSolver::new(&self.p, &self.q, &self.a, &b, &cones, settings);
        solver.solve();

        let u_idx = (N + 1) * NX;
        match solver.solution.status {
            SolverStatus::Solved | SolverStatus::AlmostSolved => {
                Ok(solver.solution.x[u_idx] as f32)
            }
            ref status => Err(MpcSolveError::from_status(status)),
        }
    }
}

/// MPC control using quadratic programming with the Clarabel solver.
///
/// This one-shot API prepares the fixed QP structure and solves it once. Runtime
/// callers with a fixed model and timestep should reuse [`PreparedMpc`] instead.
pub fn try_mpc_control(x: Vector4, model: Model, dt: f32) -> Result<f32, MpcSolveError> {
    PreparedMpc::new(model, dt).try_control(x)
}

'''
s = s[:start] + prepared + s[end:]

# Add prepared-vs-one-shot regression tests before the existing nominal test.
test_marker = "    #[test]\n    fn fallible_api_reports_a_valid_solution() {"
test_insert = r'''    #[test]
    fn prepared_api_matches_one_shot_api_across_states() {
        let model = Model::default();
        let dt = 0.1;
        let prepared = PreparedMpc::new(model, dt);
        let states = [
            vector![0.0, 0.1, 0.1, 0.0],
            vector![0.5, -0.2, 0.05, 0.1],
            vector![-0.5, 0.3, -0.1, -0.2],
        ];

        for x in states {
            let expected = try_mpc_control(x, model, dt).expect("one-shot MPC solve");
            let actual = prepared.try_control(x).expect("prepared MPC solve");
            assert!((actual - expected).abs() < 1e-5, "actual={actual}, expected={expected}");
        }
    }

'''
if "prepared_api_matches_one_shot_api_across_states" not in s:
    s = s.replace(test_marker, test_insert + test_marker, 1)
mpc.write_text(s)

# Adopt PreparedMpc in the simulator runtime cache.
domain = Path("rust_robotics_sim/src/simulator/pendulum/domain.rs")
d = domain.read_text()

lqr_cache = r'''#[derive(Debug, Clone)]
pub(crate) struct LqrRuntimeCache {
    model: Model,
    dt: f32,
    prepared: rb::control::lqr::PreparedLqr<4, 1>,
}
'''
mpc_cache = lqr_cache + r'''

pub(crate) struct MpcRuntimeCache {
    model: Model,
    dt: f32,
    prepared: PreparedMpc,
}
'''
if "struct MpcRuntimeCache" not in d:
    if lqr_cache not in d:
        raise SystemExit("LQR cache marker not found")
    d = d.replace(lqr_cache, mpc_cache, 1)

old_fields = r'''    pub(crate) lqr_cache: Option<LqrRuntimeCache>,
    pub(crate) last_control_error: Option<String>,
'''
new_fields = r'''    pub(crate) lqr_cache: Option<LqrRuntimeCache>,
    pub(crate) mpc_cache: Option<MpcRuntimeCache>,
    pub(crate) last_control_error: Option<String>,
'''
if old_fields not in d:
    raise SystemExit("runtime fields marker not found")
d = d.replace(old_fields, new_fields, 1)

old_default = r'''            lqr_cache: None,
            last_control_error: None,
'''
new_default = r'''            lqr_cache: None,
            mpc_cache: None,
            last_control_error: None,
'''
if old_default not in d:
    raise SystemExit("runtime default marker not found")
d = d.replace(old_default, new_default, 1)

old_lqr = r'''            Controller::LQR(model) => {
                let needs_prepare = self.lqr_cache.as_ref().is_none_or(|cache| {
'''
new_lqr = r'''            Controller::LQR(model) => {
                self.mpc_cache = None;
                let needs_prepare = self.lqr_cache.as_ref().is_none_or(|cache| {
'''
if old_lqr not in d:
    raise SystemExit("LQR runtime marker not found")
d = d.replace(old_lqr, new_lqr, 1)

old_pid = r'''            Controller::PID(pid) => {
                self.lqr_cache = None;
                self.last_control_error = None;
'''
new_pid = r'''            Controller::PID(pid) => {
                self.lqr_cache = None;
                self.mpc_cache = None;
                self.last_control_error = None;
'''
if old_pid not in d:
    raise SystemExit("PID runtime marker not found")
d = d.replace(old_pid, new_pid, 1)

old_mpc = r'''            Controller::MPC(model) => {
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
'''
new_mpc = r'''            Controller::MPC(model) => {
                self.lqr_cache = None;
                let needs_prepare = self.mpc_cache.as_ref().is_none_or(|cache| {
                    cache.model != *model || cache.dt.to_bits() != dt.to_bits()
                });
                if needs_prepare {
                    self.mpc_cache = Some(MpcRuntimeCache {
                        model: *model,
                        dt,
                        prepared: PreparedMpc::new(*model, dt),
                    });
                }
                match self
                    .mpc_cache
                    .as_ref()
                    .expect("MPC cache is prepared above")
                    .prepared
                    .try_control(measured_state)
                {
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
'''
if old_mpc not in d:
    raise SystemExit("MPC runtime marker not found")
d = d.replace(old_mpc, new_mpc, 1)

old_policy = r'''            Controller::Policy(policy) => {
                self.lqr_cache = None;
                self.last_control_error = None;
'''
new_policy = r'''            Controller::Policy(policy) => {
                self.lqr_cache = None;
                self.mpc_cache = None;
                self.last_control_error = None;
'''
if old_policy not in d:
    raise SystemExit("policy runtime marker not found")
d = d.replace(old_policy, new_policy, 1)

old_getter = r'''    pub fn last_control_error(&self) -> Option<&str> {
'''
new_getter = r'''    #[cfg(test)]
    pub(crate) fn mpc_cache_matches(&self, model: Model, dt: f32) -> bool {
        self.mpc_cache
            .as_ref()
            .is_some_and(|cache| cache.model == model && cache.dt.to_bits() == dt.to_bits())
    }

    pub fn last_control_error(&self) -> Option<&str> {
'''
if old_getter not in d:
    raise SystemExit("runtime getter marker not found")
d = d.replace(old_getter, new_getter, 1)

old_reset = r'''        self.lqr_cache = None;
        self.last_control_error = None;
'''
new_reset = r'''        self.lqr_cache = None;
        self.mpc_cache = None;
        self.last_control_error = None;
'''
# Replace the reset occurrence only; earlier branch snippets were already changed above.
reset_pos = d.find(old_reset, d.find("fn reset_state(&mut self)"))
if reset_pos == -1:
    raise SystemExit("reset runtime cache marker not found")
d = d[:reset_pos] + d[reset_pos:].replace(old_reset, new_reset, 1)
domain.write_text(d)

# Add simulator cache reuse/invalidation tests.
tests = Path("rust_robotics_sim/src/simulator/pendulum/tests.rs")
t = tests.read_text()
add = r'''
#[test]
fn pendulum_runtime_prepares_and_reuses_mpc_configuration() {
    let model = super::domain::mpc_stage_cost_model(Model::default());
    let mut sim = InvertedPendulum {
        state: vector![0.0, 0.0, 0.05, 0.0],
        controller: Controller::MPC(model),
        ..Default::default()
    };

    assert!(!sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(model, PENDULUM_FIXED_DT));
}

#[test]
fn mpc_runtime_reprepares_when_model_changes() {
    let original = super::domain::mpc_stage_cost_model(Model::default());
    let mut sim = InvertedPendulum {
        state: vector![0.0, 0.0, 0.05, 0.0],
        controller: Controller::MPC(original),
        ..Default::default()
    };
    sim.step(PENDULUM_FIXED_DT);
    assert!(sim.mpc_cache_matches(original, PENDULUM_FIXED_DT));

    let mut updated = original;
    updated.Q[10] *= 1.5;
    sim.controller = Controller::MPC(updated);
    sim.step(PENDULUM_FIXED_DT);

    assert!(!sim.mpc_cache_matches(original, PENDULUM_FIXED_DT));
    assert!(sim.mpc_cache_matches(updated, PENDULUM_FIXED_DT));
}
'''
if "pendulum_runtime_prepares_and_reuses_mpc_configuration" not in t:
    t += add
tests.write_text(t)
