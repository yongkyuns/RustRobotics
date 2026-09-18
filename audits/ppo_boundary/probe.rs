//! Disposable, observation-only boundary probe. No alternate PPO implementation.
use super::*;
use std::{cell::RefCell, fmt::Write as _, fs, path::Path};

#[derive(Default)]
pub(super) struct Row {
    pub(super) index: usize,
    pub(super) observation: [f32; 4],
    pub(super) mean: f32,
    pub(super) value: f32,
    pub(super) latent: f32,
    pub(super) log_prob: f32,
    pub(super) reward: f32,
    pub(super) terminated: bool,
    pub(super) truncated: bool,
    pub(super) final_observation: [f32; 4],
    pub(super) next_observation: [f32; 4],
    pub(super) path_end: bool,
    pub(super) bootstrap: f32,
}

#[derive(Default)]
struct Trace {
    rows: Vec<Row>,
    minibatches: Vec<String>,
}

thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

pub(super) fn record_transition(row: Row) {
    TRACE.with(|t| {
        if let Some(t) = t.borrow_mut().as_mut() {
            t.rows.push(row);
        }
    });
}

pub(super) fn record_bootstrap(value: f32) {
    TRACE.with(|t| {
        if let Some(t) = t.borrow_mut().as_mut() {
            let row = t.rows.last_mut().expect("a transition precedes its bootstrap");
            row.path_end = true;
            row.bootstrap = value;
        }
    });
}

pub(super) fn record_next(observation: [f32; 4]) {
    TRACE.with(|t| {
        if let Some(t) = t.borrow_mut().as_mut() {
            t.rows.last_mut().expect("a transition precedes reset").next_observation = observation;
        }
    });
}

fn state_json(session: &PpoTrainerSession) -> String {
    let state = session.shared_state();
    let mut actor = Vec::new();
    let mut critic = Vec::new();
    for layer in [state.policy.input, state.policy.hidden, state.policy.output] {
        actor.extend(layer.weight);
        actor.extend(layer.bias);
    }
    for layer in [state.value.input, state.value.hidden, state.value.output] {
        critic.extend(layer.weight);
        critic.extend(layer.bias);
    }
    assert!(actor.iter().chain(&critic).all(|x| x.is_finite()));
    format!("{{\"actor\":{actor:?},\"critic\":{critic:?}}}")
}

pub(super) fn record_minibatch(
    session: &PpoTrainerSession,
    indices: &[usize],
    policy_loss: f32,
    value_loss: f32,
) {
    TRACE.with(|t| {
        if let Some(t) = t.borrow_mut().as_mut() {
            assert!(policy_loss.is_finite() && value_loss.is_finite());
            t.minibatches.push(format!(
                "{{\"indices\":{indices:?},\"policy_loss\":{policy_loss:?},\"value_loss\":{value_loss:?},\"state\":{}}}",
                state_json(session)
            ));
        }
    });
}

fn row_json(r: &Row) -> String {
    assert!(r.observation.iter().chain(&r.final_observation).chain(&r.next_observation).all(|x| x.is_finite()));
    assert!([r.mean, r.value, r.latent, r.log_prob, r.reward, r.bootstrap].iter().all(|x| x.is_finite()));
    format!(
        "{{\"index\":{},\"observation\":{:?},\"mean\":{:?},\"value\":{:?},\"latent\":{:?},\"log_prob\":{:?},\"reward\":{:?},\"terminated\":{},\"truncated\":{},\"final_observation\":{:?},\"next_observation\":{:?},\"path_end\":{},\"bootstrap\":{:?}}}",
        r.index, r.observation, r.mean, r.value, r.latent, r.log_prob, r.reward,
        r.terminated, r.truncated, r.final_observation, r.next_observation, r.path_end, r.bootstrap
    )
}

fn emit_case(out: &Path, name: &str, seed: u64, count: usize, updates: usize, config: PpoTrainerConfig) {
    let mut traced = PpoTrainerSession::new_seeded_with_environments(config.clone(), seed, count);
    let mut plain = PpoTrainerSession::new_seeded_with_environments(config.clone(), seed, count);
    assert_eq!(state_json(&traced), state_json(&plain));
    let mut json = format!(
        "{{\"schema\":1,\"name\":\"{name}\",\"seed\":{seed},\"environment_count\":{count},\"config\":{{\"hidden_dim\":{},\"action_std\":{:?},\"max_force\":{:?},\"rollout_steps\":{},\"mini_batch_size\":{},\"epochs\":{},\"gamma\":{:?},\"gae_lambda\":{:?},\"clip_epsilon\":{:?},\"value_loss_coef\":{:?},\"learning_rate\":{:?},\"max_steps\":{}}},\"initial\":{},\"updates\":[",
        config.hidden_dim, config.action_std, config.env.max_force, config.ppo.rollout_steps,
        config.ppo.mini_batch_size, config.ppo.epochs_per_update, config.ppo.gamma,
        config.ppo.gae_lambda, config.ppo.clip_epsilon, config.ppo.value_loss_coef,
        config.ppo.learning_rate, config.env.max_steps, state_json(&traced)
    );
    for update in 0..updates {
        TRACE.with(|t| *t.borrow_mut() = Some(Trace::default()));
        let batch = traced.collect_rollout();
        traced.optimize(&batch);
        traced.metrics.total_updates += 1;
        let trace = TRACE.with(|t| t.borrow_mut().take().expect("active trace"));
        // Observers and split collect/optimize calls must not alter production
        // behavior, metrics, optimizer history, or subsequent random streams.
        plain.train_updates(1);
        assert_eq!(state_json(&traced), state_json(&plain));
        assert_eq!(traced.metrics(), plain.metrics());
        assert_eq!(trace.rows.len(), count * config.ppo.rollout_steps);
        assert_eq!(trace.minibatches.len(), config.ppo.epochs_per_update * batch.observations.len().div_ceil(config.ppo.mini_batch_size));
        if update > 0 { json.push(','); }
        write!(json,
            "{{\"rows\":[{}],\"batch\":{{\"observations\":{:?},\"latent_actions\":{:?},\"old_log_probs\":{:?},\"returns\":{:?},\"advantages\":{:?}}},\"minibatches\":[{}],\"total_env_steps\":{},\"total_updates\":{}}}",
            trace.rows.iter().map(row_json).collect::<Vec<_>>().join(","),
            batch.observations, batch.latent_actions, batch.old_log_probs, batch.returns, batch.advantages,
            trace.minibatches.join(","), traced.metrics.total_env_steps, traced.metrics.total_updates
        ).unwrap();
    }
    let next = traced.collect_rollout();
    let control = plain.collect_rollout();
    assert_eq!(next.observations, control.observations);
    assert_eq!(next.latent_actions, control.latent_actions);
    assert_eq!(next.old_log_probs, control.old_log_probs);
    assert_eq!(next.returns, control.returns);
    assert_eq!(next.advantages, control.advantages);
    json.push_str("],\"observer_control\":\"bitwise_equal_including_next_rollout\"}\n");
    fs::write(out.join(format!("{name}.json")), json).unwrap();
    println!("BOUNDARY TRACE COMPLETE {name}: {updates} updates, {count} environments");
}

#[test]
#[ignore = "explicit boundary diagnostic, not a learning-quality gate"]
fn emit_ppo_boundary_trace() {
    let out = std::env::var("PPO_BOUNDARY_OUT").expect("PPO_BOUNDARY_OUT");
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    emit_case(out, "default-201", 201, 1, 8, PpoTrainerConfig::default());
    emit_case(out, "default-204", 204, 1, 8, PpoTrainerConfig::default());
    let mut boundary = PpoTrainerConfig::default();
    boundary.env.max_steps = 3;
    boundary.env.max_position_m = 1000.0;
    boundary.env.max_angle_rad = 1000.0;
    boundary.ppo.rollout_steps = 7;
    boundary.ppo.mini_batch_size = 8;
    boundary.ppo.epochs_per_update = 2;
    emit_case(out, "timeouts-pool-201", 201, 3, 4, boundary.clone());
    boundary.env.max_angle_rad = 0.0;
    emit_case(out, "terminals-pool-201", 201, 3, 4, boundary);
}
