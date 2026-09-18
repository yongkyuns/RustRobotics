//! Audit-only configuration and observation hooks. No alternate optimizer.
use super::*;
use std::io::Write;

impl PpoTrainerSession {
    pub(crate) fn audit_new_batch(seed: u64, arm: u32) -> Self {
        let mut config = PpoTrainerConfig::default();
        let environments = match arm {
            0 => 1,
            1 => { config.ppo.rollout_steps = 4096; config.ppo.mini_batch_size = 1024; 1 },
            2 => { config.ppo.mini_batch_size = 1024; 8 },
            _ => panic!("invalid audit arm"),
        };
        Self::new_seeded_with_environments(config, seed, environments)
    }
}

pub(super) fn record_epoch(session: &PpoTrainerSession, batch: &RolloutBatch, epoch: usize) {
    let Ok(path) = std::env::var("PPO_BATCH_TRACE") else { return; };
    type B = <AutodiffBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;
    let obs = obs_tensor::<B>(&session.device, &batch.observations);
    let distribution = SquashedGaussian::new(session.config.action_std, session.actor.action_limit);
    let log_probs = distribution.log_prob_tensor(
        session.actor.valid().latent_mean(obs.clone()),
        scalar_tensor::<B>(&session.device, &batch.latent_actions),
    );
    let policy_loss = tensor_scalar(&clipped_surrogate(
        log_probs, scalar_tensor::<B>(&session.device, &batch.old_log_probs),
        scalar_tensor::<B>(&session.device, &batch.advantages), session.config.ppo.clip_epsilon,
    ));
    let mse = tensor_scalar(&(session.critic.valid().forward(obs)
        - scalar_tensor::<B>(&session.device, &batch.returns)).square().mean());
    let outward = batch.observations.iter().filter(|o| o[0].abs() >= 0.5 && o[0] * o[1] > 0.0).count();
    assert!(policy_loss.is_finite() && mse.is_finite());
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
    writeln!(file, "{{\"update\":{},\"epoch\":{},\"steps\":{},\"episodes\":{},\"samples\":{},\"outward\":{},\"clipped_surrogate\":{:?},\"target_mse\":{:?}}}",
        session.metrics.total_updates + 1, epoch, session.metrics.total_env_steps,
        session.metrics.total_episodes, batch.observations.len(), outward, -policy_loss, mse).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    fn equal(a: &PpoTrainerSession, b: &PpoTrainerSession) {
        assert_eq!(a.snapshot(), b.snapshot());
        assert_eq!(a.shared_state().value, b.shared_state().value);
        assert_eq!(a.metrics(), b.metrics());
    }
    #[test]
    fn unchanged_arm_is_ordinary_session() {
        let mut a = PpoTrainerSession::new_seeded(PpoTrainerConfig::default(), 201);
        let mut b = PpoTrainerSession::audit_new_batch(201, 0);
        for _ in 0..4 { equal(&a,&b); a.train_updates(1); b.train_updates(1); }
        equal(&a,&b);
        let a=a.collect_rollout(); let b=b.collect_rollout();
        assert_eq!(a.observations,b.observations); assert_eq!(a.returns,b.returns);
        assert_eq!(a.advantages,b.advantages); assert_eq!(a.latent_actions,b.latent_actions);
    }
    #[test]
    fn equal_large_arm_budgets_and_initialization() {
        let mut a=PpoTrainerSession::audit_new_batch(201,1);
        let mut b=PpoTrainerSession::audit_new_batch(201,2);
        equal(&a,&b);
        assert_eq!(a.environment_count(),1); assert_eq!(b.environment_count(),8);
        let x=a.collect_rollout(); let y=b.collect_rollout();
        assert_eq!(x.observations.len(),4096); assert_eq!(y.observations.len(),4096);
        assert_eq!(x.observations[..512],y.observations[..512]);
        assert_eq!(a.metrics.total_env_steps,b.metrics.total_env_steps);
        assert_eq!(x.observations.len()/a.config.ppo.mini_batch_size*a.config.ppo.epochs_per_update,16);
    }
    #[test]
    fn observer_does_not_change_training_or_rng() {
        let out=std::env::var("PPO_BATCH_OUT").unwrap();
        let path=std::path::Path::new(&out).join("observer-control.jsonl");
        let mut a=PpoTrainerSession::audit_new_batch(204,2);
        let mut b=PpoTrainerSession::audit_new_batch(204,2);
        std::env::set_var("PPO_BATCH_TRACE", &path);
        a.train_updates(2);
        std::env::remove_var("PPO_BATCH_TRACE");
        b.train_updates(2); equal(&a,&b);
        let x=a.collect_rollout(); let y=b.collect_rollout();
        assert_eq!(x.observations,y.observations);assert_eq!(x.returns,y.returns);
        assert_eq!(x.advantages,y.advantages);assert_eq!(x.latent_actions,y.latent_actions);
        let text=std::fs::read_to_string(path).unwrap();assert_eq!(text.lines().count(),10);
    }
    #[test]
    fn large_arm_grouping_keeps_optimizer_history() {
        for arm in [1,2] {
            let mut a=PpoTrainerSession::audit_new_batch(203,arm);
            let mut b=PpoTrainerSession::audit_new_batch(203,arm);
            a.train_updates(2); b.train_updates(1);let _=b.shared_state();b.train_updates(1);equal(&a,&b);
        }
    }
}
